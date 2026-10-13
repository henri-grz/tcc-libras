"""EXP03 - Classificador de ponto unico. Testa se a LSTM contribui.

Logica do teste: um classificador que ve apenas UM ponto 3D (tres numeros) nao tem
acesso a nenhuma informacao temporal, nem a configuracao de mao, nem a trajetoria.
Se ele chegar perto da acuracia do pipeline original, a modelagem temporal da LSTM nao
esta contribuindo, e o numero medido reflete classificacao estatica de nuvem de pontos
sob vazamento, e nao reconhecimento de sinal isolado.

Rodam dois modelos, ambos sobre o mesmo split embaralhado do pipeline original:
- regressao logistica (referencia trivial, treina em segundos)
- MLP raso de duas camadas

Serve tambem como a referencia trivial que o trabalho original nao inclui (falha S18).

Uso:
    python -m experiments.exp03_ponto_unico --smoke
    python -m experiments.exp03_ponto_unico
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import banner, base_parser  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.baseline_original import build_pointlevel
from src.config import load_config
from src.evaluate import compute_metrics, print_summary, save_result
from src.models import build_model
from src.seeding import set_seed
from src.splits import leaky_pointlevel_split
from src.train import get_device, predict, train_model

NAME = "exp03_ponto_unico"


def main() -> None:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--max-points", type=int, default=None,
                    help="sobrepoe reproducao.max_points do YAML (0 = todos os pontos)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bo = cfg.baseline_original
    seed = int(bo["random_state"])
    set_seed(seed)

    epochs = 3 if args.smoke else 30
    max_points = 20_000 if args.smoke else (
        args.max_points if args.max_points is not None else int(cfg.reproducao["max_points"])
    )

    banner(
        "EXP03 - classificador sobre um unico ponto 3D",
        "se chegar perto do pipeline original, a modelagem temporal nao esta contribuindo",
    )

    pld = build_pointlevel(cfg)
    print(pld.describe())

    # Mesmo pre-processamento e mesmo split embaralhado do pipeline original.
    X = StandardScaler().fit_transform(pld.X).astype(np.float32)
    y = pld.y
    train_idx, test_idx = leaky_pointlevel_split(len(X), float(bo["test_size"]), seed)

    # max_points = 0 desativa a subamostragem, como max_sequences no exp01
    if max_points and len(train_idx) > max_points:
        rng = np.random.RandomState(seed)
        train_idx = rng.choice(train_idx, size=max_points, replace=False)
    n_test_cap = max(max_points // 5, 1)
    if max_points and len(test_idx) > n_test_cap:
        rng = np.random.RandomState(seed + 1)
        test_idx = rng.choice(test_idx, size=n_test_cap, replace=False)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    print(f"\ntreino: {X_tr.shape}  teste: {X_te.shape}  (cada amostra e UM ponto 3D)")

    results: dict[str, dict] = {}

    # --- regressao logistica -------------------------------------------------
    print("\n--- regressao logistica sobre 1 ponto ---")
    # `multi_class` foi removido no scikit-learn 1.7+; o comportamento multinomial
    # e o padrao. Nao reintroduzir o argumento.
    logreg = LogisticRegression(max_iter=300, n_jobs=-1)
    logreg.fit(X_tr, y_tr)
    m_lr = compute_metrics(y_te, logreg.predict(X_te), pld.class_names)
    print(f"acuracia={m_lr['accuracy'] * 100:.2f}%  chance={m_lr['chance_level'] * 100:.2f}%")
    results["logistic_regression"] = m_lr

    # --- MLP raso ------------------------------------------------------------
    print("\n--- MLP raso sobre 1 ponto ---")
    device = get_device(prefer_cuda=not args.cpu)
    mlp = build_model("single_point_mlp", input_size=X_tr.shape[1], num_classes=len(pld.class_names))
    train_model(
        mlp, X_tr, y_tr, epochs=epochs, batch_size=512, learning_rate=1e-3,
        patience=5, monitor="train_loss", device=device, verbose=False,
    )
    m_mlp = compute_metrics(y_te, predict(mlp, X_te, device=device), pld.class_names)
    print(f"acuracia={m_mlp['accuracy'] * 100:.2f}%  chance={m_mlp['chance_level'] * 100:.2f}%")
    results["single_point_mlp"] = m_mlp

    print_summary(m_mlp, "EXP03 detalhe do MLP de ponto unico")

    best = max(m_lr["accuracy"], m_mlp["accuracy"])
    banner("EXP03 conclusao")
    print(f"melhor acuracia com UM unico ponto 3D : {best * 100:.2f}%")
    print(f"nivel de chance                       : {m_lr['chance_level'] * 100:.2f}%")
    print(
        "\nCompare com o resultado do exp01. Se os valores forem proximos, a conclusao e\n"
        "que, nesta representacao, a LSTM de quatro camadas do pipeline original nao\n"
        "agrega sobre um classificador que nao usa informacao temporal."
    )

    save_result(
        NAME,
        {
            "descricao": "classificacao a partir de um unico ponto 3D, split embaralhado do pipeline original",
            "hipotese": "acuracia proxima do pipeline original indica que a LSTM nao contribui",
            "smoke": bool(args.smoke),
            "n_train_points": int(len(X_tr)),
            "n_test_points": int(len(X_te)),
            "melhor_acuracia_ponto_unico": float(best),
            "chance_level": float(m_lr["chance_level"]),
            "models": results,
        },
        cfg.results_dir,
    )


if __name__ == "__main__":
    main()
