"""EXP01 - Reproducao fiel do baseline de Prestes (2024).

Objetivo: obter, com o pipeline original preservado, uma acuracia proxima dos 82,83%
(7 sinais) ou 70,76% (11 sinais) relatados no trabalho.

Nada aqui e corrigido. O experimento reproduz:
- amostra = landmark individual, input_size = 3
- StandardScaler no dataset completo antes do split
- train_test_split com shuffle no nivel do ponto
- sequencias criadas depois do split, janela de 200, rotulo do ultimo ponto
- LSTMOriginal com os defeitos de dropout
- early stopping na loss de TREINO, paciencia 5

Este e o numero que os experimentos 02 a 04 reavaliam sob controles de vazamento.

Uso:
    python -m experiments.exp01_reproduzir_baseline --smoke
    python -m experiments.exp01_reproduzir_baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import banner, base_parser  # noqa: E402

from src.baseline_original import build_pointlevel, leaky_pipeline
from src.config import load_config
from src.evaluate import compute_metrics, plot_confusion_matrix, print_summary, save_result
from src.models import build_model, count_parameters
from src.seeding import set_seed
from src.train import get_device, predict, train_model

NAME = "exp01_reproduzir_baseline"


def main() -> None:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--max-sequences", type=int, default=None,
                    help="sobrepoe reproducao.max_sequences do YAML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bo = cfg.baseline_original
    seed = int(bo["random_state"])
    set_seed(seed)

    epochs = 3 if args.smoke else int(bo["epochs"])
    max_seq = 600 if args.smoke else (
        args.max_sequences if args.max_sequences is not None else int(cfg.reproducao["max_sequences"])
    )

    banner(
        "EXP01 - reproducao fiel do baseline",
        "pipeline original preservado, incluindo as falhas C1, C2.1, C2.2, C2.3, S1, S4",
    )

    pld = build_pointlevel(cfg)
    print(pld.describe())

    X_tr, y_tr, X_te, y_te = leaky_pipeline(
        pld,
        sequence_length=int(bo["sequence_length"]),
        test_size=float(bo["test_size"]),
        random_state=seed,
        max_sequences=max_seq,
    )
    print(f"\nsequencias de treino: {X_tr.shape}, teste: {X_te.shape}")
    print(f"input_size do baseline: {X_tr.shape[2]}  <- so x, y, z (falha C1)")

    model = build_model("lstm_original", input_size=X_tr.shape[2], num_classes=len(pld.class_names))
    print(f"parametros: {count_parameters(model):,}")

    # Early stopping na loss de treino: comportamento original (falha S4).
    hist = train_model(
        model,
        X_tr,
        y_tr,
        epochs=epochs,
        batch_size=int(bo["batch_size"]),
        learning_rate=float(bo["learning_rate"]),
        patience=int(bo["early_stopping_patience"]),
        monitor="train_loss",
        device=get_device(prefer_cuda=not args.cpu),
    )

    y_pred = predict(model, X_te, device=get_device(prefer_cuda=not args.cpu))
    metrics = compute_metrics(y_te, y_pred, pld.class_names)
    print_summary(metrics, "EXP01 resultado (pipeline original, com vazamento)")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        pld.class_names,
        cfg.results_dir / f"{NAME}_confusao.png",
        title="EXP01 baseline original",
    )
    save_result(
        NAME,
        {
            "descricao": "reproducao fiel do pipeline de Prestes (2024), sem correcoes",
            "falhas_replicadas": ["C1", "C2.1", "C2.2", "C2.3", "S1", "S2", "S3", "S4"],
            "config": {
                "sequence_length": int(bo["sequence_length"]),
                "test_size": float(bo["test_size"]),
                "random_state": seed,
                "epochs": epochs,
                "max_sequences": max_seq,
                "input_size": int(X_tr.shape[2]),
            },
            "smoke": bool(args.smoke),
            "history": hist.to_dict(),
            "metrics": metrics,
        },
        cfg.results_dir,
    )

    print(
        "\nLeitura: guarde este numero. Ele e o alvo dos experimentos 02, 03 e 04, "
        "que testam se ele mede reconhecimento de sinal ou vazamento."
    )


if __name__ == "__main__":
    main()
