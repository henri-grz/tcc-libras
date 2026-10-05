"""DIAG05 - Diagnosticos do exp05: o modelo treinou ou nao?

O exp05 parou na epoca 16 em todas as execucoes, com paciencia 15. Ou seja, a melhor
loss de validacao foi a da epoca 1, e o modelo avaliado foi praticamente o inicial.
Este script separa duas explicacoes:
- o problema e inviavel com 1 a 2 exemplos de treino por classe entre sinalizadores
- o treino ou a representacao tem defeito

Tres diagnosticos. Nenhum altera o protocolo do exp05:
1. historico de treino: reroda o exp05 (mesmos folds e seeds) guardando loss e
   acuracia por epoca. A variante "early_stopping" deve reproduzir o exp05. Roda tambem
   uma variante "epocas_fixas", sem early stopping, so para ver o que acontece. Essa
   variante NAO e resultado reportavel: o numero de epocas nao foi escolhido por
   validacao.
2. teste de sobreajuste: treina e avalia nos mesmos 33 videos. Um pipeline saudavel
   deve decorar o treino.
3. 1-NN em LOSO sobre a representacao corrigida. Sem treino e sem early stopping.
   Mede se a representacao carrega informacao que generaliza entre sinalizadores.

Nao e um experimento do TCC. E diagnostico do exp05, e nenhum numero daqui substitui
o resultado principal.

Uso:
    python -m experiments.diag_exp05 --smoke
    python -m experiments.diag_exp05
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import banner, base_parser  # noqa: E402

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.config import load_config
from src.dataset import build_dataset, standardize
from src.evaluate import save_result
from src.models import build_model
from src.seeding import set_seed
from src.splits import assert_no_group_leakage, loso_folds
from src.train import History, get_device, predict, train_model

NAME = "diag_exp05"
MODEL = "lstm_corrected"


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) para X ~ Binomial(n, p). Chance de acertar k ou mais por sorteio."""
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def summarize_history(h: History) -> dict:
    best = h.best_epoch
    return {
        "best_epoch": best + 1,
        "stopped_epoch": h.stopped_epoch + 1,
        "train_acc_na_melhor_epoca": h.train_acc[best],
        "train_acc_max": max(h.train_acc),
        "train_acc_ultima": h.train_acc[-1],
        "val_loss_primeira": h.val_loss[0] if h.val_loss else None,
        "val_loss_min": min(h.val_loss) if h.val_loss else None,
        "val_loss_ultima": h.val_loss[-1] if h.val_loss else None,
    }


def main() -> None:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument(
        "--fixed-epochs", type=int, default=None,
        help="epocas da variante sem early stopping e do teste de sobreajuste "
             "(padrao: training.epochs do YAML)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr_cfg = cfg.training
    seeds = args.seeds or list(tr_cfg["seeds"])
    if args.smoke:
        seeds = seeds[:1]
    epochs = 5 if args.smoke else int(tr_cfg["epochs"])
    fixed_epochs = 5 if args.smoke else (args.fixed_epochs or int(tr_cfg["epochs"]))
    patience = int(tr_cfg["early_stopping_patience"])
    batch_size = int(tr_cfg["batch_size"])
    lr = float(tr_cfg["learning_rate"])
    device = get_device(prefer_cuda=not args.cpu)

    banner("DIAG05 - diagnosticos do exp05", "o modelo treinou ou nao?")

    samples = build_dataset(cfg)
    n_cls = len(samples.class_names)
    chance = 1.0 / n_cls
    print(samples.describe())

    folds = loso_folds(
        samples.y, samples.groups, samples.signers,
        val_size=float(tr_cfg["val_size"]), seed=seeds[0],
    )

    # --- 1. historico de treino ---------------------------------------------
    banner(
        "1. historico de treino",
        f"early_stopping: igual ao exp05 | epocas_fixas: {fixed_epochs} epocas, sem early stopping",
    )
    variants = {
        "early_stopping": dict(epochs=epochs, patience=patience, monitor="val_loss"),
        # paciencia = epocas: nunca para antes; restaura os pesos de menor loss de treino
        "epocas_fixas": dict(epochs=fixed_epochs, patience=fixed_epochs, monitor="train_loss"),
    }
    diag1: dict[str, list[dict]] = {v: [] for v in variants}

    for seed in seeds:
        for fold in folds:
            assert_no_group_leakage(samples.groups, fold.train, fold.test)
            assert_no_group_leakage(samples.groups, fold.train, fold.val)
            X_tr, X_val, X_te = standardize(
                samples.X[fold.train], samples.X[fold.val], samples.X[fold.test]
            )
            y_tr, y_val, y_te = samples.y[fold.train], samples.y[fold.val], samples.y[fold.test]

            for variant, kw in variants.items():
                set_seed(seed)
                model = build_model(MODEL, input_size=samples.dim, num_classes=n_cls)
                hist = train_model(
                    model, X_tr, y_tr, X_val, y_val,
                    batch_size=batch_size, learning_rate=lr, device=device, verbose=False, **kw,
                )
                pred = predict(model, X_te, device=device)
                s = summarize_history(hist)
                run = {
                    "seed": seed,
                    "fold": fold.name,
                    "test_acc": float((pred == y_te).mean()),
                    "classes_preditas_distintas": int(len(np.unique(pred))),
                    **s,
                    "historico": hist.to_dict(),
                }
                diag1[variant].append(run)
                print(
                    f"  {variant:<15} seed={seed:<5} {fold.name:<18} "
                    f"melhor_epoca={s['best_epoch']:>3}  "
                    f"train_acc(melhor/max)={s['train_acc_na_melhor_epoca'] * 100:5.1f}%/"
                    f"{s['train_acc_max'] * 100:5.1f}%  "
                    f"val_loss(1a/min/ult)={s['val_loss_primeira']:.2f}/"
                    f"{s['val_loss_min']:.2f}/{s['val_loss_ultima']:.2f}  "
                    f"teste={run['test_acc'] * 100:5.1f}%  "
                    f"classes_preditas={run['classes_preditas_distintas']}"
                )

    for variant, runs in diag1.items():
        accs = np.array([r["test_acc"] for r in runs])
        print(f"\n  {variant}: acuracia de teste media {accs.mean() * 100:.2f}% "
              f"(chance {chance * 100:.2f}%)")

    # --- 2. teste de sobreajuste --------------------------------------------
    banner("2. teste de sobreajuste", f"treina e avalia nos {len(samples)} videos, {fixed_epochs} epocas")
    set_seed(seeds[0])
    (X_all,) = standardize(samples.X)
    model = build_model(MODEL, input_size=samples.dim, num_classes=n_cls)
    hist = train_model(
        model, X_all, samples.y,
        epochs=fixed_epochs, patience=fixed_epochs, monitor="train_loss",
        batch_size=batch_size, learning_rate=lr, device=device, verbose=False,
    )
    overfit_acc = float((predict(model, X_all, device=device) == samples.y).mean())
    # acuracia de treino durante o treino e medida com dropout ligado; a final, sem
    first_100 = next((i + 1 for i, a in enumerate(hist.train_acc) if a >= 1.0), None)
    print(f"  acuracia nos proprios dados de treino: {overfit_acc * 100:.2f}%")
    print(f"  primeira epoca com 100% de acerto no treino (com dropout): {first_100}")
    print(f"  loss de treino final: {hist.train_loss[-1]:.4f}")
    diag2 = {
        "acuracia_no_treino": overfit_acc,
        "primeira_epoca_100": first_100,
        "historico": hist.to_dict(),
    }

    # --- 3. 1-NN em LOSO ------------------------------------------------------
    banner(
        "3. 1-NN em LOSO",
        "treino = os dois outros sinalizadores (treino + validacao do exp05, o 1-NN nao usa validacao)",
    )
    representations = {
        # sequencia inteira achatada: compara frame a frame
        "sequencia_achatada": lambda a: a.reshape(len(a), -1),
        # media no tempo: descarta a ordem temporal, so a postura media
        "media_temporal": lambda a: a.mean(axis=1),
    }
    diag3: dict[str, dict] = {}
    for rep_name, rep in representations.items():
        correct, total, per_fold = 0, 0, []
        for fold in folds:
            pool = np.concatenate([fold.train, fold.val])
            assert_no_group_leakage(samples.groups, pool, fold.test)
            X_tr, X_te = standardize(samples.X[pool], samples.X[fold.test])
            knn = KNeighborsClassifier(n_neighbors=1).fit(rep(X_tr), samples.y[pool])
            pred = knn.predict(rep(X_te))
            hits = int((pred == samples.y[fold.test]).sum())
            correct += hits
            total += len(fold.test)
            per_fold.append({"fold": fold.name, "acertos": hits, "n": int(len(fold.test))})
            print(f"  {rep_name:<20} {fold.name:<18} acertos={hits}/{len(fold.test)}")
        p = binom_sf(correct, total, chance)
        print(
            f"  {rep_name:<20} total {correct}/{total} = {correct / total * 100:.2f}%  "
            f"| p(acaso faria >= isso) = {p:.4f}"
        )
        diag3[rep_name] = {
            "acertos": correct, "n": total, "acuracia": correct / total,
            "p_valor_binomial": p, "por_fold": per_fold,
        }

    # limiar de acertos para superar o acaso com p < 0,05 nos 33 videos
    n_total = len(samples)
    k_min = next(k for k in range(n_total + 1) if binom_sf(k, n_total, chance) < 0.05)

    banner("como ler")
    print(
        f"- O acaso acerta em media {n_total * chance:.1f} de {n_total} videos. Para dizer que\n"
        f"  algo supera o acaso com p < 0,05, sao precisos pelo menos {k_min}/{n_total} acertos.\n"
        "- Diagnostico 2 abaixo de ~100%: o pipeline nao consegue nem decorar 33 videos.\n"
        "  Procure defeito na representacao ou no treino antes de qualquer conclusao.\n"
        "- Diagnostico 2 perto de 100% e diagnostico 1 com train_acc alto e val_loss\n"
        "  subindo: a rede decora o treino e nao generaliza para outro sinalizador.\n"
        "- Diagnostico 3 acima do limiar: a representacao carrega informacao entre\n"
        "  sinalizadores, e o gargalo e o treino da LSTM com tao poucos exemplos.\n"
        "- Diagnostico 3 no nivel do acaso: com 1 a 2 exemplos por classe, nem o\n"
        "  vizinho mais proximo generaliza entre pessoas. Isso apoia a explicacao de\n"
        "  que o dataset nao basta para reconhecimento independente de sinalizador."
    )

    save_result(
        NAME,
        {
            "descricao": "diagnosticos do exp05: historico de treino, sobreajuste e 1-NN em LOSO",
            "smoke": bool(args.smoke),
            "modelo": MODEL,
            "seeds": seeds,
            "epochs_early_stopping": epochs,
            "epochs_fixas": fixed_epochs,
            "chance_level": chance,
            "limiar_acertos_p05": {"k": k_min, "n": n_total},
            "diag1_historico": diag1,
            "diag2_sobreajuste": diag2,
            "diag3_knn_loso": diag3,
        },
        cfg.results_dir,
    )


if __name__ == "__main__":
    main()
