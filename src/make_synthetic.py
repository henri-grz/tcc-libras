"""Gera um cache de landmarks sintetico, com a mesma estrutura do cache real.

Serve para validar todo o pipeline no dia 1, antes de ter os videos da V-Librasil
baixados e processados. Nenhum resultado gerado a partir daqui vai para o TCC: e
apenas teste de fumaca.

Cada classe recebe uma trajetoria de mao caracteristica, e cada articulador recebe um
deslocamento e uma escala proprios, para que o protocolo leave-one-signer-out seja de
fato dificil, como no dado real.

Uso:
    python -m src.make_synthetic
    python -m src.make_synthetic --signers 3 --repeats 1
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import Config, load_config
from .landmarks_spec import COORDS, N_HAND, N_POSE


def _base_pose(rng: np.random.RandomState, n_frames: int, signer_scale: float) -> np.ndarray:
    """Corpo aproximadamente estatico, em coordenadas normalizadas de imagem."""
    pose = np.full((n_frames, N_POSE, COORDS), np.nan, np.float32)
    template = np.zeros((N_POSE, COORDS), np.float32)

    # Pontos minimos que a ancoragem usa: ombros (11, 12) e quadris (23, 24).
    template[11] = (0.42, 0.42, 0.0)
    template[12] = (0.58, 0.42, 0.0)
    template[23] = (0.45, 0.75, 0.0)
    template[24] = (0.55, 0.75, 0.0)
    template[0] = (0.50, 0.25, 0.0)  # nariz
    for i in range(N_POSE):
        if not template[i].any():
            template[i] = (0.5 + rng.uniform(-0.1, 0.1), 0.5 + rng.uniform(-0.2, 0.2), 0.0)

    center = np.array([0.5, 0.5, 0.0], np.float32)
    template = center + (template - center) * signer_scale

    jitter = rng.normal(0, 0.004, size=(n_frames, N_POSE, COORDS)).astype(np.float32)
    pose[:] = template[None, :, :] + jitter
    return pose


def _hand_trajectory(
    rng: np.random.RandomState,
    n_frames: int,
    class_idx: int,
    n_classes: int,
    signer_offset: np.ndarray,
) -> np.ndarray:
    """Trajetoria da mao caracteristica da classe, mais ruido e vies do sinalizador."""
    phase = 2 * np.pi * class_idx / max(n_classes, 1)
    t = np.linspace(0.0, 1.0, n_frames)

    # Centro alvo distinto por classe, mais orbita de raio e frequencia proprios.
    cx = 0.5 + 0.16 * np.cos(phase)
    cy = 0.45 + 0.16 * np.sin(phase)
    radius = 0.03 + 0.02 * ((class_idx % 3) + 1) / 3.0
    freq = 1.0 + (class_idx % 4)

    px = cx + radius * np.cos(2 * np.pi * freq * t + phase)
    py = cy + radius * np.sin(2 * np.pi * freq * t + phase)
    pz = -0.05 + 0.02 * np.sin(2 * np.pi * t)

    wrist = np.stack([px, py, pz], axis=1).astype(np.float32) + signer_offset[None, :]

    # 21 landmarks espalhados em volta do pulso, com formato fixo por classe.
    shape_rng = np.random.RandomState(1000 + class_idx)
    finger_offsets = shape_rng.uniform(-0.035, 0.035, size=(N_HAND, COORDS)).astype(np.float32)
    finger_offsets[0] = 0.0  # o pulso e a origem da mao

    hand = wrist[:, None, :] + finger_offsets[None, :, :]
    hand += rng.normal(0, 0.003, size=hand.shape).astype(np.float32)
    return hand.astype(np.float32)


def generate(cfg: Config, n_signers: int = 3, repeats: int = 1, seed: int = 7) -> pd.DataFrame:
    """Escreve `.npz` sinteticos e o metadata.csv no cache."""
    classes = cfg.classes
    rows: list[dict] = []

    for class_idx, sinal in enumerate(classes):
        for signer in range(1, n_signers + 1):
            for rep in range(repeats):
                rng = np.random.RandomState(seed + class_idx * 100 + signer * 10 + rep)
                n_frames = int(rng.randint(60, 140))

                signer_scale = 1.0 + (signer - 2) * 0.08
                signer_offset = np.array(
                    [(signer - 2) * 0.03, (signer - 2) * 0.02, 0.0], np.float32
                )

                pose = _base_pose(rng, n_frames, signer_scale)
                right = _hand_trajectory(rng, n_frames, class_idx, len(classes), signer_offset)

                # Metade das classes usa as duas maos, metade so a direita.
                if class_idx % 2 == 0:
                    left = _hand_trajectory(
                        rng, n_frames, (class_idx + 3) % len(classes), len(classes), signer_offset
                    )
                else:
                    left = np.full((n_frames, N_HAND, COORDS), np.nan, np.float32)

                # Falhas de deteccao esporadicas, para exercitar os flags de presenca.
                drop = rng.rand(n_frames) < 0.05
                right[drop] = np.nan
                pose[rng.rand(n_frames) < 0.02] = np.nan

                suffix = f"_{rep + 1}" if repeats > 1 else ""
                video_id = f"{sinal}_{signer}{suffix}"
                out_path = cfg.landmarks_dir / f"{video_id}.npz"
                np.savez_compressed(
                    out_path,
                    pose=pose,
                    pose_visibility=np.ones((n_frames, N_POSE), np.float32),
                    left_hand=left,
                    right_hand=right,
                )

                rows.append(
                    {
                        "video_id": video_id,
                        "sinal": sinal,
                        "articulador": str(signer),
                        "n_frames": n_frames,
                        "npz": str(out_path.relative_to(cfg.cache_dir)),
                        "source_video": "SINTETICO",
                    }
                )

    df = pd.DataFrame(rows).sort_values(["sinal", "articulador"]).reset_index(drop=True)
    df.to_csv(cfg.metadata_csv, index=False)
    print(f"[sintetico] {len(df)} videos falsos gerados em {cfg.landmarks_dir}")
    print(f"[sintetico] metadata: {cfg.metadata_csv}")
    print("[sintetico] ATENCAO: dado sintetico, nenhum resultado daqui vai para o TCC")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera cache de landmarks sintetico.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--signers", type=int, default=3, help="numero de articuladores")
    ap.add_argument("--repeats", type=int, default=1, help="videos por sinal por articulador")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    generate(load_config(args.config), args.signers, args.repeats, args.seed)


if __name__ == "__main__":
    main()
