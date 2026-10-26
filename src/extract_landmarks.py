"""Extracao de landmarks com MediaPipe Holistic. Roda UMA VEZ sobre os videos.

Correcoes em relacao ao pipeline original:

- Um arquivo `.npz` por video, nunca JSON agregado. Isso elimina a falha C3, em que
  frames de videos diferentes eram concatenados com `extend()` e o nome do arquivo
  vinha do indice de `os.listdir`.
- Rotulo e articulador vem do nome do arquivo por regex explicito, registrados em
  `metadata.csv`. Nada depende da ordem do sistema de arquivos.
- Landmark ausente e gravado como NaN, nao omitido. Isso preserva o alinhamento
  posicional e permite os flags de presenca.
- Mao esquerda e direita ficam em arrays separados, nunca no mesmo slot.
- `visibility` da pose e gravado (no pipeline original essa linha esta comentada).

Uso:
    python -m src.extract_landmarks
    python -m src.extract_landmarks --overwrite
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .landmarks_spec import COORDS, N_HAND, N_POSE

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}


def parse_filename(name: str, pattern: str) -> tuple[str | None, str | None]:
    """Extrai (sinal, articulador) do nome do arquivo.

    Substitui o `arq.split()[0]` do pipeline original, que depende do formato exato do
    nome e nao captura o articulador, isto e, a coluna da V-LIBRASIL, que define os folds por sinalizador.
    """
    m = re.match(pattern, name)
    if not m:
        return None, None
    d = m.groupdict()
    return d.get("sinal"), d.get("articulador")


def list_videos(videos_dir: Path) -> list[Path]:
    return sorted(p for p in videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def extract_one(video_path: Path, cfg: Config) -> dict[str, np.ndarray]:
    """Processa um video e devolve os arrays de landmarks.

    Returns:
        dict com `pose` (F,33,3), `pose_visibility` (F,33), `left_hand` (F,21,3),
        `right_hand` (F,21,3). NaN indica ausencia de deteccao.
    """
    import cv2
    import mediapipe as mp

    ext = cfg.extraction
    mp_holistic = mp.solutions.holistic

    pose_frames: list[np.ndarray] = []
    vis_frames: list[np.ndarray] = []
    lh_frames: list[np.ndarray] = []
    rh_frames: list[np.ndarray] = []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"nao foi possivel abrir o video: {video_path}")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=int(ext["model_complexity"]),
        min_detection_confidence=float(ext["min_detection_confidence"]),
        min_tracking_confidence=float(ext["min_tracking_confidence"]),
    ) as holistic:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = cv2.resize(frame, (int(ext["frame_width"]), int(ext["frame_height"])))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            frame_arrays = results_to_arrays(holistic.process(rgb))

            pose_frames.append(frame_arrays["pose"])
            vis_frames.append(frame_arrays["pose_visibility"])
            lh_frames.append(frame_arrays["left_hand"])
            rh_frames.append(frame_arrays["right_hand"])

    capture.release()

    if not pose_frames:
        raise RuntimeError(f"nenhum frame lido de {video_path}")

    return {
        "pose": np.stack(pose_frames).astype(np.float32),
        "pose_visibility": np.stack(vis_frames).astype(np.float32),
        "left_hand": np.stack(lh_frames).astype(np.float32),
        "right_hand": np.stack(rh_frames).astype(np.float32),
    }


def results_to_arrays(results) -> dict[str, np.ndarray]:
    """Converte a saida do Holistic para UM frame nos arrays do cache.

    Publica para que o uso em tempo real (src/checkpoint.py) monte os frames com a
    mesma funcao da extracao.

    Returns:
        dict com `pose` (33,3), `pose_visibility` (33,), `left_hand` (21,3),
        `right_hand` (21,3). NaN indica ausencia de deteccao.
    """
    return {
        "pose": _pose_to_array(results.pose_landmarks),
        "pose_visibility": _visibility_to_array(results.pose_landmarks),
        "left_hand": _hand_to_array(results.left_hand_landmarks),
        "right_hand": _hand_to_array(results.right_hand_landmarks),
    }


def _pose_to_array(landmarks) -> np.ndarray:
    out = np.full((N_POSE, COORDS), np.nan, np.float32)
    if landmarks is None:
        return out
    for i, lm in enumerate(landmarks.landmark):
        out[i] = (lm.x, lm.y, lm.z)
    return out


def _visibility_to_array(landmarks) -> np.ndarray:
    out = np.full((N_POSE,), np.nan, np.float32)
    if landmarks is None:
        return out
    for i, lm in enumerate(landmarks.landmark):
        out[i] = lm.visibility
    return out


def _hand_to_array(landmarks) -> np.ndarray:
    out = np.full((N_HAND, COORDS), np.nan, np.float32)
    if landmarks is None:
        return out
    for i, lm in enumerate(landmarks.landmark):
        out[i] = (lm.x, lm.y, lm.z)
    return out


def run(cfg: Config, overwrite: bool = False) -> pd.DataFrame:
    """Extrai landmarks de todos os videos e escreve o cache mais o metadata.csv."""
    videos = list_videos(cfg.videos_dir)
    if not videos:
        raise SystemExit(
            f"nenhum video encontrado em {cfg.videos_dir}. "
            "Baixe a V-Librasil ou rode `python -m src.make_synthetic` para testar o pipeline."
        )

    pattern = cfg.extraction["filename_regex"]
    valid_classes = set(cfg.classes)
    rows: list[dict] = []
    skipped: list[str] = []

    for path in videos:
        sinal, articulador = parse_filename(path.stem, pattern)
        if sinal is None:
            skipped.append(f"{path.name}: nome nao casa com o regex")
            continue
        if sinal not in valid_classes:
            skipped.append(f"{path.name}: sinal '{sinal}' fora da lista de classes")
            continue

        video_id = path.stem.replace(" ", "_")
        out_path = cfg.landmarks_dir / f"{video_id}.npz"

        if out_path.exists() and not overwrite:
            with np.load(out_path) as data:
                n_frames = int(data["pose"].shape[0])
        else:
            arrays = extract_one(path, cfg)
            np.savez_compressed(out_path, **arrays)
            n_frames = int(arrays["pose"].shape[0])
            print(f"[extract] {video_id}: {n_frames} frames")

        rows.append(
            {
                "video_id": video_id,
                "sinal": sinal,
                "articulador": str(articulador),
                "n_frames": n_frames,
                "npz": out_path.relative_to(cfg.cache_dir).as_posix(),
                "source_video": str(path.relative_to(cfg.videos_dir)),
            }
        )

    if skipped:
        print(f"\n[extract] {len(skipped)} arquivo(s) ignorado(s):")
        for s in skipped:
            print(f"  - {s}")

    df = pd.DataFrame(rows).sort_values(["sinal", "articulador"]).reset_index(drop=True)
    df.to_csv(cfg.metadata_csv, index=False)
    print(f"\n[extract] {len(df)} videos no cache. metadata: {cfg.metadata_csv}")
    print(df.groupby("sinal")["articulador"].count().to_string())
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Extrai landmarks dos videos (roda uma vez).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true", help="reprocessa videos já em cache")
    args = ap.parse_args()
    run(load_config(args.config), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
