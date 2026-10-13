"""Curva de controle do exp01: teto de memorizacao por kNN sobre um unico ponto 3D.

Companheiro do exp01, previsto em docs/03_protocolo_curva_exp01.md. Nao e um
experimento novo: e a referencia contra a qual a curva de `max_sequences` da LSTM
deve ser lida.

Eixo x identico ao do exp01: numero de pontos 3D ROTULADOS vistos no treino. No exp01
cada janela de 200 pontos carrega exatamente um rotulo (o do ultimo ponto), logo N
janelas equivalem a N pontos rotulados. A comparacao e direta.

Dois splits, mesma grade:
- embaralhado no nivel do ponto, como o pipeline original (falhas C2.1 e C2.2)
- agrupado por articulador, sem vazamento

A distancia entre as duas curvas e a medida do vazamento, isolada de qualquer escolha
de arquitetura, de otimizador ou de numero de epocas.

Uso:
    python -m experiments.exp01_curva_controle_knn
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import banner, base_parser  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.baseline_original import build_pointlevel
from src.config import load_config
from src.evaluate import save_result
from src.seeding import set_seed
from src.splits import assert_no_group_leakage, leaky_pointlevel_split

NAME = "exp01_curva_controle_knn"

GRADE = [8_000, 32_000, 128_000, None]  # None = todo o treino disponivel
N_TESTE = 20_000


def _articulador(video_id: str) -> str:
    """Ultimo campo do video_id. Ver extraction.filename_regex no YAML."""
    m = re.search(r"[_\- ]([0-9A-Za-z]+)$", video_id)
    return m.group(1) if m else video_id


def main() -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.baseline_original["random_state"])
    set_seed(seed)

    banner(
        "EXP01 curva de controle - kNN sobre UM ponto 3D",
        "teto de memorizacao. eixo x = pontos rotulados, igual ao max_sequences do exp01",
    )

    pld = build_pointlevel(cfg)
    print(pld.describe())
    rng = np.random.RandomState(seed)

    # --- split embaralhado no ponto, como o pipeline original ----------------
    tr_leak, te_leak = leaky_pointlevel_split(
        len(pld.X), float(cfg.baseline_original["test_size"]), seed
    )
    X_leak = StandardScaler().fit_transform(pld.X).astype(np.float32)  # falha C2.1

    # --- split agrupado por articulador, sem vazamento -----------------------
    arts = np.array([_articulador(g) for g in pld.groups])
    holdout = sorted(set(arts.tolist()))[-1]
    te_grp = np.where(arts == holdout)[0]
    tr_grp = np.where(arts != holdout)[0]
    assert_no_group_leakage(pld.groups, tr_grp, te_grp)
    sc = StandardScaler().fit(pld.X[tr_grp])  # so o treino
    X_grp = sc.transform(pld.X).astype(np.float32)

    cenarios = {
        "embaralhado_no_ponto": (X_leak, tr_leak, te_leak, "split do pipeline original (C2.2)"),
        "agrupado_por_articulador": (X_grp, tr_grp, te_grp, f"holdout articulador {holdout}"),
    }

    curvas: dict[str, list[dict]] = {}
    for nome, (X, tr, te, desc) in cenarios.items():
        te_sub = te if len(te) <= N_TESTE else rng.choice(te, size=N_TESTE, replace=False)
        print(f"\n--- {nome} ({desc}) ---")
        print(f"treino disponivel: {len(tr):,}  teste avaliado: {len(te_sub):,}")
        pontos: list[dict] = []
        for n in GRADE:
            n_eff = len(tr) if n is None else min(n, len(tr))
            idx = tr if n is None else rng.choice(tr, size=n_eff, replace=False)
            linha = {"n_treino": int(n_eff), "subamostrado": n is not None}
            for k in (1, 5):
                t0 = time.time()
                knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(X[idx], pld.y[idx])
                acc = float(knn.score(X[te_sub], pld.y[te_sub]))
                linha[f"acuracia_k{k}"] = acc
                print(f"  n={n_eff:>7,}  k={k}  acuracia={acc * 100:6.2f}%  ({time.time() - t0:.0f}s)")
            pontos.append(linha)
        curvas[nome] = pontos

    banner("leitura")
    for p_leak, p_grp in zip(curvas["embaralhado_no_ponto"], curvas["agrupado_por_articulador"]):
        gap = (p_leak["acuracia_k1"] - p_grp["acuracia_k1"]) * 100
        print(
            f"n={p_leak['n_treino']:>7,}  vazado={p_leak['acuracia_k1'] * 100:6.2f}%  "
            f"agrupado={p_grp['acuracia_k1'] * 100:6.2f}%  vazamento={gap:+6.2f} pp"
        )

    save_result(
        NAME,
        {
            "descricao": "teto de memorizacao por kNN sobre 1 ponto 3D, dois splits, grade de tamanho",
            "papel": "curva de controle do exp01, ver docs/03_protocolo_curva_exp01.md",
            "n_teste_avaliado": int(N_TESTE),
            "chance_level": 1.0 / len(pld.class_names),
            "grade": [n if n is not None else "completo" for n in GRADE],
            "curvas": curvas,
        },
        cfg.results_dir,
    )


if __name__ == "__main__":
    main()
