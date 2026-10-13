"""Especificacao do vetor de features por frame.

Este modulo e a correcao da falha C1 do pipeline original. Nele, cada landmark
individual era uma amostra e `input_size` valia 3, o que impede o modelo de saber qual
ponto esta olhando. Aqui, um frame vira um vetor de tamanho fixo com posicoes
determinadas: cada landmark sempre ocupa o mesmo par de indices, e mao ausente e
representada por zeros mais um flag de presenca.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Contagens do MediaPipe Holistic.
N_POSE = 33
N_HAND = 21
COORDS = 3  # x, y, z

# Indices de pose usados na ancoragem (ombros e quadris).
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24


@dataclass(frozen=True)
class FeatureSpec:
    """Layout do vetor de features por frame.

    A ordem e sempre: pose, mao esquerda, mao direita, flags de presenca.
    """

    pose_indices: tuple[int, ...]
    use_pose: bool = True
    use_hands: bool = True
    use_presence_flags: bool = True

    @classmethod
    def from_config(cls, features_cfg: dict) -> "FeatureSpec":
        return cls(
            pose_indices=tuple(features_cfg.get("pose_indices", range(N_POSE))),
            use_pose=bool(features_cfg.get("use_pose", True)),
            use_hands=bool(features_cfg.get("use_hands", True)),
            use_presence_flags=bool(features_cfg.get("use_presence_flags", True)),
        )

    @property
    def n_pose(self) -> int:
        return len(self.pose_indices) if self.use_pose else 0

    @property
    def dim(self) -> int:
        """Dimensao total do vetor por frame."""
        d = self.n_pose * COORDS
        if self.use_hands:
            d += 2 * N_HAND * COORDS
        if self.use_presence_flags and self.use_hands:
            d += 2
        return d

    def slice_names(self) -> list[str]:
        """Nomes dos blocos do vetor, na ordem, para relatorio e ablacao."""
        names: list[str] = []
        if self.use_pose:
            for i in self.pose_indices:
                names += [f"pose{i}_{c}" for c in "xyz"]
        if self.use_hands:
            for side in ("lh", "rh"):
                for i in range(N_HAND):
                    names += [f"{side}{i}_{c}" for c in "xyz"]
        if self.use_presence_flags and self.use_hands:
            names += ["lh_present", "rh_present"]
        return names

    def block_slices(self) -> dict[str, slice]:
        """Fatias nomeadas do vetor, usadas pela ablacao do exp06."""
        out: dict[str, slice] = {}
        cur = 0
        if self.use_pose:
            n = self.n_pose * COORDS
            out["pose"] = slice(cur, cur + n)
            cur += n
        if self.use_hands:
            n = N_HAND * COORDS
            out["left_hand"] = slice(cur, cur + n)
            cur += n
            out["right_hand"] = slice(cur, cur + n)
            cur += n
        if self.use_presence_flags and self.use_hands:
            out["flags"] = slice(cur, cur + 2)
            cur += 2
        return out


def anchor_and_scale(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula origem e escala por frame a partir da pose.

    Origem: ponto medio dos ombros. Escala: distancia entre ombros, com fallback
    para o comprimento do torso e, em ultimo caso, 1.0. Isso torna a representacao
    invariante a posicao do sinalizador no quadro e a distancia da camera, o que o
    pipeline original nao fazia (aplicava z-score global sobre coordenadas de imagem).

    Args:
        pose: array (F, 33, 3), com NaN onde nao houve deteccao.

    Returns:
        origin (F, 3) e scale (F, 1).
    """
    ls = pose[:, POSE_LEFT_SHOULDER, :]
    rs = pose[:, POSE_RIGHT_SHOULDER, :]
    origin = np.nanmean(np.stack([ls, rs], axis=0), axis=0)  # (F, 3)

    shoulder_dist = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=1)  # so x, y

    lh = pose[:, POSE_LEFT_HIP, :]
    rh = pose[:, POSE_RIGHT_HIP, :]
    hip_mid = np.nanmean(np.stack([lh, rh], axis=0), axis=0)
    torso = np.linalg.norm(origin[:, :2] - hip_mid[:, :2], axis=1)

    scale = np.where(np.isfinite(shoulder_dist) & (shoulder_dist > 1e-6), shoulder_dist, torso)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

    # Frames sem pose nenhuma: origem zero para nao propagar NaN na subtracao.
    origin = np.nan_to_num(origin, nan=0.0)
    return origin.astype(np.float32), scale.astype(np.float32)[:, None]


def build_frame_matrix(
    pose: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    spec: FeatureSpec,
    normalize: bool = True,
) -> np.ndarray:
    """Converte landmarks brutos de um video em matriz (F, D).

    Args:
        pose: (F, 33, 3) com NaN onde ausente.
        left_hand: (F, 21, 3) com NaN onde ausente.
        right_hand: (F, 21, 3) com NaN onde ausente.
        spec: layout do vetor.
        normalize: aplica ancoragem e escala por frame.

    Returns:
        (F, spec.dim) float32, sem NaN. Landmark ausente vira zero, e a ausencia
        de cada mao e sinalizada pelos flags.
    """
    n_frames = pose.shape[0]

    if normalize:
        origin, scale = anchor_and_scale(pose)
        o = origin[:, None, :]
        s = scale[:, None, :]
        pose = (pose - o) / s
        left_hand = (left_hand - o) / s
        right_hand = (right_hand - o) / s

    parts: list[np.ndarray] = []

    if spec.use_pose:
        sel = pose[:, list(spec.pose_indices), :]
        parts.append(sel.reshape(n_frames, -1))

    if spec.use_hands:
        parts.append(left_hand.reshape(n_frames, -1))
        parts.append(right_hand.reshape(n_frames, -1))

    matrix = np.concatenate(parts, axis=1) if parts else np.zeros((n_frames, 0), np.float32)

    if spec.use_presence_flags and spec.use_hands:
        lh_present = np.isfinite(left_hand[:, 0, 0]).astype(np.float32)[:, None]
        rh_present = np.isfinite(right_hand[:, 0, 0]).astype(np.float32)[:, None]
        matrix = np.concatenate([matrix, lh_present, rh_present], axis=1)

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix.astype(np.float32)


def resample_to_length(matrix: np.ndarray, target_len: int) -> np.ndarray:
    """Reamostra a sequencia para `target_len` frames por interpolacao linear.

    Substitui o corte manual da posicao neutra do trabalho original (falha S14),
    que nao era reprodutivel. Aqui todo video vira uma sequencia de comprimento
    fixo, independente da duracao original.
    """
    n_frames, dim = matrix.shape
    if n_frames == 0:
        return np.zeros((target_len, dim), np.float32)
    if n_frames == 1:
        return np.repeat(matrix, target_len, axis=0).astype(np.float32)

    src = np.linspace(0.0, 1.0, n_frames)
    dst = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, dim), np.float32)
    for j in range(dim):
        out[:, j] = np.interp(dst, src, matrix[:, j])
    return out
