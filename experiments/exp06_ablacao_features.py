"""EXP06 - Ablacao de features: maos vs maos + pose.

Unica ablacao dentro do escopo dos 60 dias. Responde se os landmarks de pose agregam
algo sobre os landmarks de mao, usando o pipeline corrigido e o protocolo LOSO.

A ablacao e feita fatiando o vetor de features ja construido, sem re-extrair nada do
video. Isso e possivel porque o layout do vetor e determinado por `FeatureSpec`.

Usa o mesmo protocolo de treino do exp05 (`--protocolo`, padrao em
training.protocolo_loso do YAML). Ver docs/04_revisao_protocolo_exp05.md.

Uso:
    python -m experiments.exp06_ablacao_features --smoke
    python -m experiments.exp06_ablacao_features
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import (  # noqa: E402
    add_protocol_arg,
    banner,
    base_parser,
    stopping_kwargs,
)

from src.config import load_config
from src.dataset import build_dataset, standardize
from src.evaluate import aggregate_runs, compute_metrics, save_result
from src.models import build_model
from src.seeding import set_seed
from src.splits import assert_no_group_leakage, loso_folds, protocol_train_val
from src.train import get_device, predict, train_model

NAME = "exp06_ablacao_features"

# Configuracoes de ablacao: nome legivel -> blocos do vetor de features.
ABLATIONS: dict[str, list[str]] = {
    "maos": ["left_hand", "right_hand", "flags"],
    "maos_pose": ["pose", "left_hand", "right_hand", "flags"],
    "so_pose": ["pose"],
}


def main() -> None:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--model", default="lstm_corrected")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument(
        "--ablations", nargs="+", default=list(ABLATIONS), choices=list(ABLATIONS)
    )
    add_protocol_arg(ap)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr_cfg = cfg.training
    protocolo = args.protocolo or str(tr_cfg["protocolo_loso"])
    run_name = f"{NAME}_{protocolo}"
    seeds = args.seeds or list(tr_cfg["seeds"])
    if args.smoke:
        seeds = seeds[:1]
    epochs = 5 if args.smoke else int(tr_cfg["epochs"])
    device = get_device(prefer_cuda=not args.cpu)

    banner(
        "EXP06 - ablacao de features com protocolo LOSO",
        f"protocolo de treino: {protocolo}",
    )

    full = build_dataset(cfg)
    print(full.describe())
    print(f"\nblocos disponiveis: {list(full.spec.block_slices())}")

    folds = loso_folds(
        full.y, full.groups, full.signers,
        val_size=float(tr_cfg["val_size"]), seed=seeds[0],
    )
    if not folds:
        raise SystemExit("nenhum fold LOSO gerado")

    results: dict[str, dict] = {}

    for ablation in args.ablations:
        blocks = ABLATIONS[ablation]
        subset = full.select_features(blocks)
        banner(f"ablacao: {ablation}", f"blocos={blocks}  D={subset.dim}")

        runs: list[dict] = []
        for seed in seeds:
            for fold in folds:
                set_seed(seed)
                tr_idx, val_idx = protocol_train_val(fold, protocolo)
                assert_no_group_leakage(subset.groups, tr_idx, fold.test)
                assert_no_group_leakage(subset.groups, tr_idx, val_idx)

                X_tr, X_val, X_te = standardize(
                    subset.X[tr_idx], subset.X[val_idx], subset.X[fold.test]
                )
                model = build_model(
                    args.model, input_size=subset.dim, num_classes=len(subset.class_names)
                )
                train_model(
                    model, X_tr, subset.y[tr_idx], X_val, subset.y[val_idx],
                    batch_size=int(tr_cfg["batch_size"]),
                    learning_rate=float(tr_cfg["learning_rate"]),
                    device=device, verbose=False,
                    **stopping_kwargs(protocolo, tr_cfg, epochs),
                )
                metrics = compute_metrics(
                    subset.y[fold.test],
                    predict(model, X_te, device=device),
                    subset.class_names,
                )
                print(
                    f"  seed={seed} {fold.name:<22} acc={metrics['accuracy'] * 100:6.2f}%  "
                    f"f1_macro={metrics['f1_macro'] * 100:6.2f}%"
                )
                runs.append({"seed": seed, "fold": fold.name, **metrics})

        agg = aggregate_runs(runs)
        print(
            f"\n{ablation}: acuracia {agg['accuracy']['mean'] * 100:.2f}% "
            f"+- {agg['accuracy']['std'] * 100:.2f}"
        )
        results[ablation] = {"blocks": blocks, "dim": subset.dim, "aggregate": agg, "runs": runs}

    banner("EXP06 conclusao")
    print(f"{'ablacao':<12} {'D':>5}  {'acuracia':>18}  {'f1_macro':>18}")
    for name, res in results.items():
        a = res["aggregate"]["accuracy"]
        f = res["aggregate"]["f1_macro"]
        print(
            f"{name:<12} {res['dim']:>5}  "
            f"{a['mean'] * 100:9.2f}% +- {a['std'] * 100:5.2f}  "
            f"{f['mean'] * 100:9.2f}% +- {f['std'] * 100:5.2f}"
        )

    save_result(
        run_name,
        {
            "descricao": "ablacao de blocos de features sob protocolo LOSO",
            "modelo": args.model,
            "protocolo_treino": protocolo,
            "parada": stopping_kwargs(protocolo, tr_cfg, epochs),
            "revisao": "docs/04_revisao_protocolo_exp05.md",
            "smoke": bool(args.smoke),
            "seeds": seeds,
            "resultados": results,
        },
        cfg.results_dir,
    )


if __name__ == "__main__":
    main()
