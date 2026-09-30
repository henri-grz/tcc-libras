"""EXP05 - Pipeline corrigido avaliado com leave-one-signer-out.

Este e o resultado principal do TCC: o numero honesto.

Correcoes aplicadas:
- amostra = video, sequencia = frames, vetor por frame com layout fixo (C1)
- normalizacao invariante por frame: origem nos ombros, escala pela distancia entre eles
- reamostragem temporal para T frames fixos, sem corte manual (S14)
- padronizacao ajustada apenas no treino (C2.1)
- split agrupado por video e, no protocolo principal, por sinalizador (C2.2)
- conjunto de validacao separado e early stopping na loss de validacao (S3, S4)
- dropout efetivo (S1)
- 3 seeds x 3 folds, com media e desvio padrao (S10)
- metricas por classe com nome correto (S6) e matriz de confusao salva (S11)

Compara duas arquiteturas: LSTM corrigida e BiLSTM corrigida.

Uso:
    python -m experiments.exp05_pipeline_corrigido_loso --smoke
    python -m experiments.exp05_pipeline_corrigido_loso
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import banner, base_parser  # noqa: E402

from src.config import load_config
from src.dataset import build_dataset, standardize
from src.evaluate import (
    aggregate_runs,
    compute_metrics,
    plot_confusion_matrix,
    print_summary,
    save_result,
)
from src.models import build_model, count_parameters
from src.seeding import set_seed
from src.splits import assert_no_group_leakage, loso_folds
from src.train import get_device, predict, train_model

NAME = "exp05_pipeline_corrigido_loso"
MODELS = ("lstm_corrected", "bilstm_corrected")


def main() -> None:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr_cfg = cfg.training
    seeds = args.seeds or list(tr_cfg["seeds"])
    if args.smoke:
        seeds = seeds[:1]
    epochs = 5 if args.smoke else int(tr_cfg["epochs"])
    device = get_device(prefer_cuda=not args.cpu)

    banner(
        "EXP05 - pipeline corrigido, leave-one-signer-out",
        "resultado principal do TCC",
    )

    samples = build_dataset(cfg)
    print(samples.describe())

    folds = loso_folds(
        samples.y, samples.groups, samples.signers,
        val_size=float(tr_cfg["val_size"]), seed=seeds[0],
    )
    if not folds:
        raise SystemExit(
            "nenhum fold LOSO gerado. Verifique a coluna 'articulador' no metadata.csv."
        )
    print(f"\nfolds LOSO: {len(folds)}")

    all_results: dict[str, dict] = {}

    for model_name in args.models:
        banner(f"modelo: {model_name}")
        runs: list[dict] = []
        first_cm: list[list[int]] | None = None

        for seed in seeds:
            for fold in folds:
                set_seed(seed)
                assert_no_group_leakage(samples.groups, fold.train, fold.test)
                assert_no_group_leakage(samples.groups, fold.train, fold.val)

                X_tr, X_val, X_te = standardize(
                    samples.X[fold.train], samples.X[fold.val], samples.X[fold.test]
                )
                y_tr = samples.y[fold.train]
                y_val = samples.y[fold.val]
                y_te = samples.y[fold.test]

                model = build_model(
                    model_name, input_size=samples.dim, num_classes=len(samples.class_names)
                )
                hist = train_model(
                    model, X_tr, y_tr, X_val, y_val,
                    epochs=epochs,
                    batch_size=int(tr_cfg["batch_size"]),
                    learning_rate=float(tr_cfg["learning_rate"]),
                    patience=int(tr_cfg["early_stopping_patience"]),
                    monitor="val_loss",
                    device=device,
                    verbose=False,
                )

                metrics = compute_metrics(
                    y_te, predict(model, X_te, device=device), samples.class_names
                )
                print(
                    f"  seed={seed} {fold.name:<22} {fold.sizes():<34} "
                    f"acc={metrics['accuracy'] * 100:6.2f}%  "
                    f"f1_macro={metrics['f1_macro'] * 100:6.2f}%  "
                    f"epocas={hist.stopped_epoch + 1}"
                )
                runs.append({"seed": seed, "fold": fold.name, **metrics})
                if first_cm is None:
                    first_cm = metrics["confusion_matrix"]

        agg = aggregate_runs(runs)
        params = count_parameters(
            build_model(model_name, input_size=samples.dim, num_classes=len(samples.class_names))
        )
        print(
            f"\n{model_name}: acuracia {agg['accuracy']['mean'] * 100:.2f}% "
            f"+- {agg['accuracy']['std'] * 100:.2f}  |  "
            f"f1_macro {agg['f1_macro']['mean'] * 100:.2f}% "
            f"+- {agg['f1_macro']['std'] * 100:.2f}  |  {params:,} parametros"
        )
        all_results[model_name] = {"aggregate": agg, "n_params": params, "runs": runs}

        if first_cm is not None:
            plot_confusion_matrix(
                first_cm, samples.class_names,
                cfg.results_dir / f"{NAME}_{model_name}_confusao.png",
                title=f"EXP05 {model_name} (fold 1)",
            )

    banner("EXP05 conclusao")
    chance = 1.0 / len(samples.class_names)
    print(f"nivel de chance: {chance * 100:.2f}%\n")
    for name, res in all_results.items():
        a = res["aggregate"]["accuracy"]
        f = res["aggregate"]["f1_macro"]
        print(
            f"{name:<20} acuracia {a['mean'] * 100:6.2f}% +- {a['std'] * 100:5.2f}   "
            f"f1_macro {f['mean'] * 100:6.2f}% +- {f['std'] * 100:5.2f}"
        )

    best = max(all_results.items(), key=lambda kv: kv[1]["aggregate"]["accuracy"]["mean"])
    print_summary(all_results[best[0]]["runs"][0], f"detalhe do primeiro fold de {best[0]}")
    print(
        "\nEste e o numero que vai no resumo do TCC, com o desvio padrao e o protocolo\n"
        "explicitos. Reporte sempre acuracia balanceada junto da acuracia."
    )

    save_result(
        NAME,
        {
            "descricao": "pipeline corrigido avaliado com leave-one-signer-out",
            "correcoes_aplicadas": ["C1", "C2.1", "C2.2", "C2.3", "S1", "S3", "S4", "S6", "S10", "S11", "S14"],
            "smoke": bool(args.smoke),
            "protocolo": {
                "folds_loso": len(folds),
                "seeds": seeds,
                "epochs": epochs,
                "sequence_length": samples.seq_len,
                "feature_dim": samples.dim,
                "classes": samples.class_names,
                "articuladores": sorted(set(samples.signers.tolist())),
            },
            "chance_level": chance,
            "melhor_modelo": best[0],
            "resultados": all_results,
        },
        cfg.results_dir,
    )


if __name__ == "__main__":
    main()
