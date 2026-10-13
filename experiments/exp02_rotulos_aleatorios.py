"""EXP02 - Teste de rotulo aleatorio. Prova o vazamento de dados.

Logica do teste: cada video recebe um sinal sorteado aleatoriamente. Os rotulos passam a
nao ter nenhuma relacao com o conteudo dos videos. Em um pipeline sem vazamento, a
acuracia de teste deve cair ao nivel de chance (1/n_classes).

Se o pipeline original ainda assim atingir acuracia alta, isso so pode vir de
memorizacao de coordenadas que aparecem simultaneamente no treino e no teste. Ou seja,
vazamento. E o teste classico de randomizacao de rotulos.

Este e o experimento mais importante do TCC. E tambem o mais barato de rodar.

Uso:
    python -m experiments.exp02_rotulos_aleatorios --smoke
    python -m experiments.exp02_rotulos_aleatorios
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import banner, base_parser  # noqa: E402

from src.baseline_original import (
    build_pointlevel,
    leaky_pipeline,
    random_label_permutation,
    restaurar_melhor_epoca,
)
from src.config import load_config
from src.evaluate import compute_metrics, print_summary, save_result
from src.models import build_model
from src.seeding import set_seed
from src.train import get_device, predict, train_model

NAME = "exp02_rotulos_aleatorios"


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
        "EXP02 - rotulos aleatorios no pipeline original",
        "acuracia acima do nivel de chance com rotulos aleatorios = prova de vazamento",
    )

    runs: list[dict] = []
    device = get_device(prefer_cuda=not args.cpu)

    for seed in seeds:
        set_seed(seed)
        print(f"\n--- seed {seed} ---")

        permutation = random_label_permutation(cfg, seed=seed)
        pld = build_pointlevel(cfg, label_permutation=permutation)

        X_tr, y_tr, X_te, y_te = leaky_pipeline(
            pld,
            sequence_length=int(bo["sequence_length"]),
            test_size=float(bo["test_size"]),
            random_state=int(bo["random_state"]),
            max_sequences=max_seq,
        )

        model = build_model(
            "lstm_original", input_size=X_tr.shape[2], num_classes=len(pld.class_names)
        )
        train_model(
            model,
            X_tr,
            y_tr,
            epochs=epochs,
            batch_size=int(bo["batch_size"]),
            learning_rate=float(bo["learning_rate"]),
            patience=int(bo["early_stopping_patience"]),
            monitor="train_loss",
            device=device,
            verbose=False,
            restore_best=restore_best,
        )

        y_pred = predict(model, X_te, device=device)
        metrics = compute_metrics(y_te, y_pred, pld.class_names)
        print(
            f"acuracia={metrics['accuracy'] * 100:.2f}%  "
            f"chance={metrics['chance_level'] * 100:.2f}%  "
            f"f1_macro={metrics['f1_macro'] * 100:.2f}%"
        )
        runs.append({"seed": seed, **metrics})

    accs = np.array([r["accuracy"] for r in runs])
    chance = runs[0]["chance_level"]
    ratio = float(accs.mean() / chance) if chance > 0 else float("nan")

    print_summary(runs[0], "EXP02 detalhe da primeira seed")
    banner("EXP02 conclusao")
    print(f"acuracia media com rotulos aleatorios : {accs.mean() * 100:.2f}% (+- {(accs.std(ddof=1) if len(accs) > 1 else 0.0) * 100:.2f} pp)")
    print(f"nivel de chance                       : {chance * 100:.2f}%")
    print(f"razao acuracia/chance                 : {ratio:.2f}x")
    if ratio > 1.5:
        print(
            "\nVEREDITO: vazamento confirmado. Com rotulos sem relacao com o conteudo, o\n"
            "pipeline original ainda separa as classes. Isso e memorizacao de coordenadas\n"
            "presentes nos dois lados do split, nao reconhecimento de sinal."
        )
    else:
        print(
            "\nVEREDITO: nao houve inflacao significativa nesta configuracao. Verifique se\n"
            "o dataset e grande o suficiente e se o split realmente embaralhou pontos."
        )

    save_result(
        NAME,
        {
            "descricao": "teste de randomizacao de rotulos no nivel do video, pipeline original",
            "hipotese": "acuracia acima do nivel de chance indica vazamento de dados",
            "smoke": bool(args.smoke),
            "pesos_finais": "melhor_epoca" if restore_best else "ultima_epoca",
            "ordem_pontos": bo.get("ordem_pontos", "por_frame"),
            "chance_level": chance,
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "ratio_over_chance": ratio,
            "vazamento_confirmado": bool(ratio > 1.5),
            "runs": runs,
        },
        cfg.results_dir,
    )


if __name__ == "__main__":
    main()
