"""Construcao do dataset corrigido: uma amostra = um video.

Esta e a correcao central. No baseline, uma amostra era um landmark individual e a
"sequencia" era uma janela deslizante sobre um array embaralhado. Aqui:

    X: (N_videos, T, D)   com T fixo e D com layout determinado
    y: (N_videos,)
    groups:  video_id     -> usado para split agrupado
    signers: articulador  -> coluna da V-LIBRASIL, usada nos folds por sinalizador
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .landmarks_spec import FeatureSpec, build_frame_matrix, resample_to_length


@dataclass
class Samples:
    """Dataset em nivel de video, pronto para split agrupado."""

    X: np.ndarray  # (N, T, D)
    y: np.ndarray  # (N,) int
    groups: np.ndarray  # (N,) video_id
    signers: np.ndarray  # (N,) articulador
    class_names: list[str]
    spec: FeatureSpec

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[2])

    @property
    def seq_len(self) -> int:
        return int(self.X.shape[1])

    def describe(self) -> str:
        counts = pd.Series(self.y).value_counts().sort_index()
        lines = [
            f"amostras (videos): {len(self)}",
            f"sequencia: T={self.seq_len}, D={self.dim}",
            f"classes: {len(self.class_names)}",
            f"articuladores: {sorted(set(self.signers.tolist()))}",
            "distribuicao por classe:",
        ]
        for idx, n in counts.items():
            lines.append(f"  {self.class_names[int(idx)]}: {n}")
        return "\n".join(lines)

    def select_features(self, blocks: list[str]) -> "Samples":
        """Devolve uma copia com apenas os blocos de features indicados.

        Usado pela ablacao do exp06 (maos vs maos+pose) sem re-extrair nada.
        """
        slices = self.spec.block_slices()
        missing = [b for b in blocks if b not in slices]
        if missing:
            raise KeyError(f"blocos inexistentes: {missing}. disponiveis: {list(slices)}")
        cols = np.concatenate([np.arange(slices[b].start, slices[b].stop) for b in blocks])
        return Samples(
            X=self.X[:, :, cols].copy(),
            y=self.y.copy(),
            groups=self.groups.copy(),
            signers=self.signers.copy(),
            class_names=list(self.class_names),
            spec=self.spec,
        )


def load_metadata(cfg: Config) -> pd.DataFrame:
    if not cfg.metadata_csv.exists():
        raise SystemExit(
            f"{cfg.metadata_csv} nao existe. Rode `python -m src.extract_landmarks` "
            "ou `python -m src.make_synthetic` primeiro."
        )
    return pd.read_csv(cfg.metadata_csv, dtype={"articulador": str})


def load_pessoas(cfg: Config) -> dict[str, str] | None:
    """Mapeamento anonimo video -> pessoa (P1 a P5), feito por inspecao visual.

    A coluna `articulador` do nome do arquivo nao identifica a pessoa. Devolve None se o
    arquivo nao existir.
    """
    if not cfg.pessoas_csv.exists():
        return None
    df = pd.read_csv(cfg.pessoas_csv, dtype=str)
    return dict(zip(df["video_id"], df["pessoa"]))


def load_video_arrays(cfg: Config, npz_rel: str) -> dict[str, np.ndarray]:
    path = cfg.cache_dir / npz_rel
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def build_dataset(
    cfg: Config,
    spec: FeatureSpec | None = None,
    sequence_length: int | None = None,
    normalize: bool | None = None,
) -> Samples:
    """Monta o dataset em nivel de video a partir do cache de landmarks."""
    meta = load_metadata(cfg)
    spec = spec or FeatureSpec.from_config(cfg.features)
    seq_len = sequence_length or int(cfg.features["sequence_length"])
    do_norm = cfg.features["normalize_anchor"] if normalize is None else normalize

    # A ordem das classes vem da configuracao, nao do sistema de arquivos.
    # Isso corrige a falha C3 na leitura.
    class_names = [c for c in cfg.classes if c in set(meta["sinal"])]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    xs, ys, groups, signers = [], [], [], []

    for row in meta.itertuples(index=False):
        arrays = load_video_arrays(cfg, row.npz)
        matrix = build_frame_matrix(
            pose=arrays["pose"],
            left_hand=arrays["left_hand"],
            right_hand=arrays["right_hand"],
            spec=spec,
            normalize=bool(do_norm),
        )
        xs.append(resample_to_length(matrix, seq_len))
        ys.append(class_to_idx[row.sinal])
        groups.append(row.video_id)
        signers.append(str(row.articulador))

    return Samples(
        X=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.int64),
        groups=np.asarray(groups),
        signers=np.asarray(signers),
        class_names=class_names,
        spec=spec,
    )


def standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Padroniza por feature usando estatisticas SO do treino.

    Corrige a falha C2, item 1: o baseline chamava `scaler.fit_transform(X)` no
    dataset completo antes do split, vazando media e desvio do teste.

    Args:
        train: (N, T, D).
        others: demais conjuntos a transformar com as estatisticas do treino.
    """
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    def apply(a: np.ndarray) -> np.ndarray:
        shape = a.shape
        return ((a.reshape(-1, shape[-1]) - mean) / std).reshape(shape).astype(np.float32)

    return (apply(train), *(apply(o) for o in others))
