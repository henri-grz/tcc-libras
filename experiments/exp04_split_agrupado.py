"""EXP04 - Contrafactual: mesma representacao, mesmo modelo, split correto.

Este e o experimento que isola a variavel. Tudo aqui e identico ao exp01, exceto:
- o split e agrupado por video (nenhum video aparece no treino e no teste)
- a padronizacao usa apenas estatisticas do treino
- as janelas sao contiguas dentro de cada conjunto

A diferenca de acuracia entre exp01 e exp04 e a medida direta do vazamento. Como so uma
variavel mudou, a queda pode ser atribuida a ela.

Uso:
    python -m experiments.exp04_split_agrupado --smoke
    python -m experiments.exp04_split_agrupado
"""

from __future__ import annotations

import json

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import banner, base_parser  # noqa: E402

from src.baseline_original import grouped_pipeline, restaurar_melhor_epoca
from src.config import load_config
from src.dataset import load_metadata
from src.evaluate import compute_metrics, plot_confusion_matrix, print_summary, save_result
from src.models import build_model
from src.seeding import set_seed
from src.splits import grouped_split
from src.train import get_device, predict, train_model

NAME = "exp04_split_agrupado"


def main() -> None:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--max-sequences", type=int, default=None,
                    help="sobrepoe reproducao.max_sequences do YAML")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    args = ap.parse_args()

    cfg = load_config(args.config)
    bo = cfg.baseline_original
    restore_best = restaurar_melhor_epoca(bo)
    epochs = 3 if args.smoke else int(bo["epochs"])
    max_seq = 600 if args.smoke else (
        args.max_sequences if args.max_sequences is not None else int(cfg.reproducao["max_sequences"])
    )
    seeds = args.seeds[:1] if args.smoke else args.seeds

    banner(
        "EXP04 - representacao do pipeline original com split agrupado por video",
        "a queda em relacao ao exp01 mede o vazamento, porque so o split mudou",
    )

    meta = load_metadata(cfg)
    class_names = [c for c in cfg.classes if c in set(meta["sinal"])]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_video = np.array([class_to_idx[s] for s in meta["sinal"]])
    groups = meta["video_id"].to_numpy()

    device = get_device(prefer_cuda=not args.cpu)
    runs: list[dict] = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n--- seed {seed} ---")

        tr_idx, te_idx = grouped_split(y_video, groups, test_size=0.3, seed=seed)
        train_videos = groups[tr_idx].tolist()
        test_videos = groups[te_idx].tolist()
        assert not (set(train_videos) & set(test_videos)), "vazamento de video"
        print(f"videos treino={len(train_videos)} teste={len(test_videos)}")

        X_tr, y_tr, X_te, y_te = grouped_pipeline(
            cfg,
            train_videos=train_videos,
            test_videos=test_videos,
            sequence_length=int(bo["sequence_length"]),
            max_sequences=max_seq,
            random_state=seed,
        )

        model = build_model(
            "lstm_original", input_size=X_tr.shape[2], num_classes=len(class_names)
        )
        train_model(
            model, X_tr, y_tr, epochs=epochs, batch_size=int(bo["batch_size"]),
            learning_rate=float(bo["learning_rate"]),
            patience=int(bo["early_stopping_patience"]),
            monitor="train_loss", device=device, verbose=False,
            restore_best=restore_best,
        )

        metrics = compute_metrics(y_te, predict(model, X_te, device=device), class_names)
        print(
            f"acuracia={metrics['accuracy'] * 100:.2f}%  "
            f"balanceada={metrics['balanced_accuracy'] * 100:.2f}%  "
            f"chance={metrics['chance_level'] * 100:.2f}%"
        )
        runs.append({"seed": seed, "n_test_videos": len(test_videos), **metrics})

    accs = np.array([r["accuracy"] for r in runs])
    print_summary(runs[0], "EXP04 detalhe da primeira seed")

    # Compara com o exp01 (variante com todos os videos), se ele ja foi rodado.
    exp01_path = cfg.results_dir / "exp01_reproduzir_baseline.json"
    if not exp01_path.exists():
        exp01_path = None
    delta = None
    if exp01_path is not None:
        with open(exp01_path, encoding="utf-8") as f:
            exp01_acc = json.load(f)["metrics"]["accuracy"]
        delta = float(exp01_acc - accs.mean())

    banner("EXP04 conclusao")
    print(f"acuracia media com split agrupado : {accs.mean() * 100:.2f}%")
    print(f"nivel de chance                   : {runs[0]['chance_level'] * 100:.2f}%")
    if delta is not None:
        print(f"acuracia do exp01 (com vazamento) : {(accs.mean() + delta) * 100:.2f}%")
        print(f"queda atribuivel ao vazamento     : {delta * 100:.2f} pontos percentuais")
    else:
        print("rode o exp01 antes para obter a comparacao direta")

    save_result(
        NAME,
        {
            "descricao": "representacao do pipeline original com split agrupado por video",
            "unica_variavel_alterada": "split e escopo da padronizacao",
            "smoke": bool(args.smoke),
            "pesos_finais": "melhor_epoca" if restore_best else "ultima_epoca",
            "ordem_pontos": bo.get("ordem_pontos", "por_frame"),
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "chance_level": float(runs[0]["chance_level"]),
            "queda_vs_exp01_pp": None if delta is None else delta * 100,
            "runs": runs,
        },
        cfg.results_dir,
    )
    plot_confusion_matrix(
        runs[0]["confusion_matrix"], class_names,
        cfg.results_dir / f"{NAME}_confusao.png", title="EXP04 split agrupado",
    )


if __name__ == "__main__":
    main()
