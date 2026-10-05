"""Particionamento dos dados. Correcao da falha C2, item 2.

O baseline usava `train_test_split` com shuffle no nivel do ponto, o que coloca pontos
do mesmo frame e do mesmo video em treino e teste ao mesmo tempo. Aqui todo split e
agrupado por video, e o protocolo principal e leave-one-signer-out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


@dataclass
class Split:
    """Indices de um particionamento."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    name: str = ""

    def sizes(self) -> str:
        return f"treino={len(self.train)} val={len(self.val)} teste={len(self.test)}"


def grouped_split(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split agrupado, estratificado quando possivel.

    Tenta `StratifiedGroupKFold` para manter a proporcao de classes. Se o numero de
    grupos por classe for pequeno demais, cai para `GroupShuffleSplit`, que garante
    apenas a separacao por grupo. Em nenhum caso um video aparece nos dois lados.
    """
    n_splits = max(2, int(round(1.0 / test_size)))
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
        if len(np.unique(y[train_idx])) == len(np.unique(y)):
            return train_idx, test_idx
    except ValueError:
        pass

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


def grouped_train_val_test(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> Split:
    """Tres conjuntos disjuntos por video.

    O baseline nao tinha conjunto de validacao (falha S3) e fazia early stopping na
    loss de treino (falha S4). Aqui a validacao existe e sai de dentro do treino.
    """
    train_pool, test_idx = grouped_split(y, groups, test_size=test_size, seed=seed)
    rel_train, rel_val = grouped_split(
        y[train_pool], groups[train_pool], test_size=val_size, seed=seed
    )
    return Split(
        train=train_pool[rel_train],
        val=train_pool[rel_val],
        test=test_idx,
        name=f"grouped(seed={seed})",
    )


def loso_folds(
    y: np.ndarray,
    groups: np.ndarray,
    signers: np.ndarray,
    val_size: float = 0.2,
    seed: int = 42,
) -> list[Split]:
    """Leave-one-signer-out. Um fold por articulador.

    Este e o protocolo correto para reconhecimento de sinais: o modelo e testado em um
    sinalizador que nunca viu. A V-Librasil tem 3 articuladores por sinal, logo 3 folds.
    O baseline nao fazia nada disso (falha S10).
    """
    folds: list[Split] = []
    for signer in sorted(set(signers.tolist())):
        test_idx = np.where(signers == signer)[0]
        pool = np.where(signers != signer)[0]
        if len(test_idx) == 0 or len(pool) == 0:
            continue
        rel_train, rel_val = grouped_split(
            y[pool], groups[pool], test_size=val_size, seed=seed
        )
        folds.append(
            Split(
                train=pool[rel_train],
                val=pool[rel_val],
                test=test_idx,
                name=f"loso(holdout={signer})",
            )
        )
    return folds


PROTOCOLOS_LOSO = ("revisado", "original")


def protocol_train_val(split: Split, protocolo: str) -> tuple[np.ndarray, np.ndarray]:
    """Indices de treino e de validacao de um fold, conforme o protocolo de treino.

    - "original": treino e validacao como saem de `loso_folds`; a validacao alimenta o
      early stopping. E o protocolo da primeira rodada do exp05.
    - "revisado": a validacao volta para o treino e nao ha validacao. O treino usa
      epocas fixas. Ver docs/04_revisao_protocolo_exp05.md.

    Em nenhum dos dois o teste e tocado.
    """
    if protocolo == "original":
        return split.train, split.val
    if protocolo == "revisado":
        return np.concatenate([split.train, split.val]), np.array([], dtype=np.int64)
    raise ValueError(f"protocolo desconhecido: {protocolo}. disponiveis: {PROTOCOLOS_LOSO}")


def leaky_pointlevel_split(
    n_rows: int, test_size: float = 0.1, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduz o split do baseline: shuffle no nivel do ponto.

    NAO USAR em experimento corrigido. Existe apenas para reproduzir fielmente o
    trabalho original (exp01) e para medir o vazamento (exp02, exp03).
    """
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(n_rows)
    n_test = int(round(n_rows * test_size))
    return idx[n_test:], idx[:n_test]


def assert_no_group_leakage(
    groups: np.ndarray, train: np.ndarray, test: np.ndarray
) -> None:
    """Falha alto se algum grupo aparecer nos dois lados.

    Chamar isso em todo experimento corrigido. E a salvaguarda contra a falha que
    inviabilizou o trabalho original.
    """
    overlap = set(groups[train].tolist()) & set(groups[test].tolist())
    if overlap:
        raise AssertionError(
            f"vazamento de grupo detectado em {len(overlap)} grupo(s): "
            f"{sorted(overlap)[:5]}"
        )
