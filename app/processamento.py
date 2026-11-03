"""Processamento da demo: MediaPipe, desenho dos landmarks e janela da webcam.

Separado da interface (app/demo.py) para ser testavel sem o Gradio. O modelo recebe os
landmarks pelo mesmo caminho da extracao (src/extract_landmarks.py): quadro
redimensionado para o tamanho de `extracao`, BGR -> RGB, Holistic com os mesmos
parametros e `results_to_arrays`. O desenho e feito sobre o quadro original, so para
exibicao: as coordenadas do MediaPipe sao fracoes da imagem e valem nos dois.
"""

from __future__ import annotations

import hashlib
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.extract_landmarks import VIDEO_EXTENSIONS, results_to_arrays

# Os 33 videos da V-LIBRASIL sao 16:9 (1920x1080 ou 1280x720) e a extracao os
# redimensiona para 1024x768. Um quadro de outro aspecto (webcam 4:3) e recortado para
# 16:9 antes, para o corpo sofrer a mesma deformacao que sofreu no treino.
ASPECTO_TREINO = 16 / 9

# Mediana da duracao dos videos da V-LIBRASIL (4,1 a 10,4 s), medida em 26/10/2026.
# O modelo ve o sinal inteiro, com a posicao neutra no inicio e no fim, reamostrado
# para T frames. Os videos tem 24, 30 ou 60 fps, entao a janela e medida em segundos.
JANELA_PADRAO_S = 5.5

# Fracao minima de quadros da janela com alguma mao detectada para haver predicao.
MIN_FRACAO_MAOS = 0.1

LARGURA_EXIBICAO = 960


def recortar_aspecto(frame: np.ndarray, aspecto: float = ASPECTO_TREINO) -> np.ndarray:
    """Recorte central do quadro para o aspecto dos videos de treino."""
    h, w = frame.shape[:2]
    if w / h > aspecto + 1e-3:
        nw = int(round(h * aspecto))
        x0 = (w - nw) // 2
        return frame[:, x0 : x0 + nw]
    if w / h < aspecto - 1e-3:
        nh = int(round(w / aspecto))
        y0 = (h - nh) // 2
        return frame[y0 : y0 + nh]
    return frame


def reduzir(frame: np.ndarray, largura: int = LARGURA_EXIBICAO) -> np.ndarray:
    """Reduz para exibicao, mantendo o aspecto. Nao amplia."""
    h, w = frame.shape[:2]
    if w <= largura:
        return frame
    return cv2.resize(frame, (largura, int(round(h * largura / w))), interpolation=cv2.INTER_AREA)


class ProcessadorFrames:
    """Holistic com os parametros da extracao. Um por video ou sessao de webcam, porque
    o rastreamento entre quadros (static_image_mode=False) guarda estado."""

    def __init__(self, extracao: dict, largura_exibicao: int = LARGURA_EXIBICAO):
        import mediapipe as mp

        self._mp = mp
        self.tamanho = (int(extracao["frame_width"]), int(extracao["frame_height"]))
        self.largura_exibicao = largura_exibicao
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=int(extracao["model_complexity"]),
            min_detection_confidence=float(extracao["min_detection_confidence"]),
            min_tracking_confidence=float(extracao["min_tracking_confidence"]),
        )

    def __enter__(self) -> "ProcessadorFrames":
        return self

    def __exit__(self, *exc) -> None:
        self.holistic.close()

    def processar(self, frame_bgr: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Quadro BGR (ja recortado) -> (arrays do quadro, quadro RGB com landmarks).

        A entrada do Holistic repete `extract_one`: resize, depois BGR -> RGB.
        """
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, self.tamanho), cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)

        desenho = cv2.cvtColor(reduzir(frame_bgr, self.largura_exibicao), cv2.COLOR_BGR2RGB)
        self._desenhar(desenho, results)
        return results_to_arrays(results), desenho

    def _desenhar(self, imagem: np.ndarray, results) -> None:
        du = self._mp.solutions.drawing_utils
        ds = self._mp.solutions.drawing_styles
        hol = self._mp.solutions.holistic
        du.draw_landmarks(
            imagem, results.pose_landmarks, hol.POSE_CONNECTIONS,
            landmark_drawing_spec=ds.get_default_pose_landmarks_style(),
        )
        for mao in (results.left_hand_landmarks, results.right_hand_landmarks):
            du.draw_landmarks(
                imagem, mao, hol.HAND_CONNECTIONS,
                landmark_drawing_spec=ds.get_default_hand_landmarks_style(),
                connection_drawing_spec=ds.get_default_hand_connections_style(),
            )


def empilhar(quadros: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lista de arrays por quadro -> pose (F,33,3), left_hand (F,21,3), right_hand (F,21,3)."""
    return tuple(np.stack([q[k] for q in quadros]) for k in ("pose", "left_hand", "right_hand"))


def fracao_com_mao(quadros: list[dict[str, np.ndarray]]) -> float:
    """Fracao dos quadros com pelo menos uma mao detectada."""
    if not quadros:
        return 0.0
    return float(np.mean([
        np.isfinite(q["left_hand"][0, 0]) or np.isfinite(q["right_hand"][0, 0]) for q in quadros
    ]))


class JanelaDeslizante:
    """Quadros dos ultimos `segundos`, pelo relogio, independente do fps da camera."""

    def __init__(self, segundos: float):
        self.segundos = float(segundos)
        self._itens: deque[tuple[float, dict[str, np.ndarray]]] = deque()

    def adicionar(self, t: float, quadro: dict[str, np.ndarray]) -> None:
        self._itens.append((t, quadro))
        while self._itens and t - self._itens[0][0] > self.segundos:
            self._itens.popleft()

    def __len__(self) -> int:
        return len(self._itens)

    def duracao(self) -> float:
        return self._itens[-1][0] - self._itens[0][0] if len(self._itens) > 1 else 0.0

    def quadros(self) -> list[dict[str, np.ndarray]]:
        return [q for _, q in self._itens]


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


@lru_cache(maxsize=4)
def _hashes_dos_videos(videos_dir: str, video_ids: frozenset[str]) -> dict[str, str]:
    """sha1 -> video_id dos videos de treino presentes em `videos_dir`."""
    pasta = Path(videos_dir)
    if not pasta.is_dir():
        return {}
    return {
        _sha1(p): p.stem.replace(" ", "_")
        for p in sorted(pasta.iterdir())
        if p.suffix.lower() in VIDEO_EXTENSIONS and p.stem.replace(" ", "_") in video_ids
    }


def video_do_treino(caminho: Path, videos_treino: list[str], videos_dir: Path) -> str | None:
    """video_id do treino que corresponde ao arquivo enviado, ou None.

    Compara o conteudo (sha1, pega arquivo renomeado) e o nome (pega arquivo
    recodificado). Um video do treino reencodado E renomeado passa despercebido.
    """
    ids = frozenset(videos_treino)
    por_hash = _hashes_dos_videos(str(videos_dir), ids).get(_sha1(caminho))
    if por_hash:
        return por_hash
    nome = caminho.stem.replace(" ", "_")
    return nome if nome in ids else None
