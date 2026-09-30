"""Reproducao fiel do pipeline de Prestes (2024).

Este modulo replica de proposito as falhas do trabalho original. Nao corrigir nada
aqui: e o baseline que os experimentos 01 a 04 reproduzem e reavaliam.

Replicado:
- uma amostra por landmark individual, `input_size = 3` (falha C1)
- ordem de leitura: primeiro `hand_landmarks`, depois `pose_landmarks`, como no
  Apendice B p. 60-61
- `StandardScaler` ajustado no dataset completo antes do split (falha C2.1)
- `train_test_split` com shuffle no nivel do ponto (falha C2.2)
- `create_sequences` com janela deslizante de 200 e rotulo do ultimo ponto (falha C2.3)

Diferenca deliberada: os dados vem do cache `.npz`, nao dos JSON agregados. O bug de
rotulagem C3 do Apendice A nao e reproduzido, porque reproduzi-lo tornaria impossivel
saber qual e o rotulo verdadeiro de cada amostra. Isso e conservador: favorece o
baseline, e ainda assim ele nao se sustenta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .dataset import load_metadata, load_video_arrays
from .splits import leaky_pointlevel_split


@dataclass
class PointLevelData:
    """Dataset no formato do baseline: uma linha por landmark."""

    X: np.ndarray  # (n_points, 3)
    y: np.ndarray  # (n_points,) int
    groups: np.ndarray  # (n_points,) video_id, para o exp04
    class_names: list[str]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def describe(self) -> str:
        return (
            f"pontos: {len(self):,}\n"
            f"videos: {len(set(self.groups.tolist()))}\n"
            f"classes: {len(self.class_names)}\n"
            f"pontos por classe: "
            f"{pd.Series(self.y).value_counts().sort_index().to_dict()}"
        )


def build_pointlevel(
    cfg: Config,
    video_ids: list[str] | None = None,
    label_permutation: dict[str, str] | None = None,
) -> PointLevelData:
    """Monta o dataset no nivel do ponto, como no Apendice B.

    Args:
        video_ids: restringe aos videos indicados (usado pelo exp04).
        label_permutation: mapa video_id -> sinal, para embaralhar rotulos no nivel do
            video (usado pelo exp02).
    """
    meta = load_metadata(cfg)
    if video_ids is not None:
        meta = meta[meta["video_id"].isin(video_ids)]

    class_names = [c for c in cfg.classes if c in set(meta["sinal"])]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gs: list[np.ndarray] = []

    for row in meta.itertuples(index=False):
        arrays = load_video_arrays(cfg, row.npz)
        sinal = label_permutation.get(row.video_id, row.sinal) if label_permutation else row.sinal
        label = class_to_idx[sinal]

        # Ordem do original: maos primeiro, depois pose. Landmark ausente era
        # simplesmente omitido, entao replicamos removendo as linhas com NaN.
        per_frame: list[np.ndarray] = []
        for name in ("left_hand", "right_hand", "pose"):
            a = arrays[name]  # (F, K, 3)
            flat = a.reshape(-1, 3)
            per_frame.append(flat[np.isfinite(flat).all(axis=1)])

        pts = np.concatenate(per_frame, axis=0).astype(np.float32)
        if pts.size == 0:
            continue
        xs.append(pts)
        ys.append(np.full(len(pts), label, dtype=np.int64))
        gs.append(np.full(len(pts), row.video_id, dtype=object))

    return PointLevelData(
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        groups=np.concatenate(gs, axis=0),
        class_names=class_names,
    )


def create_sequences(
    data: np.ndarray, labels: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Replica exata de `create_sequences` do Apendice B, p. 61.

    Janela deslizante de passo 1, rotulo do ultimo ponto da janela. Aplicada sobre um
    array ja embaralhado pelo split, e por isso cada "sequencia" mistura classes.
    """
    sequences, sequence_labels = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i : i + sequence_length])
        sequence_labels.append(labels[i + sequence_length - 1])
    return np.array(sequences, dtype=np.float32), np.array(sequence_labels, dtype=np.int64)


def _fit_scaler_on_all(X: np.ndarray) -> np.ndarray:
    """Replica `scaler.fit_transform(X)` aplicado ao dataset completo (falha C2.1)."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(X).astype(np.float32)


def leaky_pipeline(
    pld: PointLevelData,
    sequence_length: int = 200,
    test_size: float = 0.1,
    random_state: int = 42,
    max_sequences: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """O pipeline original, com todas as falhas preservadas.

    Ordem exata do original: padroniza tudo, embaralha pontos, divide, e so depois cria
    as sequencias.

    Args:
        max_sequences: subamostra as janelas para caber na memoria. O original gerava
            uma janela por ponto, o que com milhoes de pontos e impraticavel. A
            subamostragem e uniforme e nao altera a natureza do vazamento.
    """
    X = _fit_scaler_on_all(pld.X)
    train_idx, test_idx = leaky_pointlevel_split(len(X), test_size, random_state)

    X_train_seq, y_train_seq = create_sequences(X[train_idx], pld.y[train_idx], sequence_length)
    X_test_seq, y_test_seq = create_sequences(X[test_idx], pld.y[test_idx], sequence_length)

    if max_sequences:
        X_train_seq, y_train_seq = _subsample(X_train_seq, y_train_seq, max_sequences, random_state)
        X_test_seq, y_test_seq = _subsample(
            X_test_seq, y_test_seq, max(max_sequences // 5, 1), random_state
        )

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


def grouped_pipeline(
    cfg: Config,
    train_videos: list[str],
    test_videos: list[str],
    sequence_length: int = 200,
    max_sequences: int | None = None,
    random_state: int = 42,
    label_permutation: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mesma representacao do baseline, mas com split agrupado por video (exp04).

    A unica coisa que muda em relacao a `leaky_pipeline` e o split: os videos de teste
    nunca aparecem no treino, a padronizacao usa apenas estatisticas do treino, e as
    janelas sao contiguas dentro de cada conjunto. Isolar essa unica variavel e o que
    permite atribuir a queda de acuracia ao vazamento.
    """
    from sklearn.preprocessing import StandardScaler

    tr = build_pointlevel(cfg, video_ids=train_videos, label_permutation=label_permutation)
    te = build_pointlevel(cfg, video_ids=test_videos, label_permutation=label_permutation)

    scaler = StandardScaler().fit(tr.X)  # so o treino
    X_tr = scaler.transform(tr.X).astype(np.float32)
    X_te = scaler.transform(te.X).astype(np.float32)

    X_train_seq, y_train_seq = create_sequences(X_tr, tr.y, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_te, te.y, sequence_length)

    if max_sequences:
        X_train_seq, y_train_seq = _subsample(X_train_seq, y_train_seq, max_sequences, random_state)
        X_test_seq, y_test_seq = _subsample(
            X_test_seq, y_test_seq, max(max_sequences // 5, 1), random_state
        )

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


def _subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def random_label_permutation(cfg: Config, seed: int = 42) -> dict[str, str]:
    """Sorteia um sinal aleatorio para cada video, mantendo o agrupamento (exp02).

    O ponto do teste: os rotulos passam a nao ter relacao nenhuma com o conteudo dos
    videos. Se o pipeline original ainda assim atingir acuracia alta, isso so pode vir
    de memorizacao de coordenadas que aparecem nos dois lados do split, ou seja,
    vazamento. E o teste classico de rotulo aleatorio.
    """
    meta = load_metadata(cfg)
    rng = np.random.RandomState(seed)
    classes = [c for c in cfg.classes if c in set(meta["sinal"])]
    return {
        vid: classes[int(rng.randint(len(classes)))]
        for vid in meta["video_id"].tolist()
    }
