"""Arquivo do modelo treinado, autocontido, para uso fora dos experimentos (tempo real).

O checkpoint guarda, alem dos pesos, tudo o que o pre-processamento precisa para
transformar landmarks brutos do MediaPipe na entrada do modelo, na mesma ordem dos
experimentos:

    landmarks brutos (F frames) -> build_frame_matrix -> resample_to_length(T)
    -> colunas da ablacao -> padronizacao com media e desvio do treino -> modelo

Sem isso, o uso em tempo real teria de repetir a configuracao a mao, e qualquer
divergencia (outra ordem de colunas, outro T, padronizacao recalculada) muda a entrada
do modelo sem erro visivel.

O modelo salvo pelos experimentos e treinado com TODOS os videos da configuracao. O
desempenho esperado dele em um sinalizador novo e a acuracia LOSO do experimento,
gravada em `info["avaliacao_loso"]`. Nao ha teste independente desse modelo.

Uso:
    from src.checkpoint import load_checkpoint
    modelo = load_checkpoint("results/modelos/exp05_..._lstm_corrected.pth")
    # pose (F,33,3), left_hand (F,21,3), right_hand (F,21,3), NaN onde ausente,
    # um frame por vez com src.extract_landmarks.results_to_arrays
    sinal, prob = modelo.prever(pose, left_hand, right_hand)
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .landmarks_spec import FeatureSpec, build_frame_matrix, resample_to_length
from .models import build_model

FORMATO = 1


def _versao(pacote: str) -> str | None:
    try:
        return metadata.version(pacote)
    except metadata.PackageNotFoundError:
        return None


def block_columns(spec: FeatureSpec, blocks: list[str]) -> np.ndarray:
    """Indices das colunas dos blocos, na ordem dada. Mesma regra de
    `Samples.select_features`."""
    slices = spec.block_slices()
    return np.concatenate([np.arange(slices[b].start, slices[b].stop) for b in blocks])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    class_names: list[str],
    spec: FeatureSpec,
    blocks: list[str] | None,
    sequence_length: int,
    normalize: bool,
    mean: np.ndarray,
    std: np.ndarray,
    extracao: dict[str, Any],
    info: dict[str, Any],
) -> Path:
    """Grava o checkpoint. So tipos simples e tensores: carrega com weights_only=True.

    Args:
        spec: layout completo do vetor por frame, como sai de `build_dataset`.
        blocks: blocos selecionados (ablacao do exp06). None usa o vetor inteiro.
        mean, std: estatisticas de padronizacao do treino, shape (1, D) ou (D,).
        extracao: bloco `extraction` do YAML. O tempo real deve redimensionar o quadro
            para frame_width x frame_height: as coordenadas do MediaPipe sao fracoes da
            largura e da altura, entao o aspecto muda a escala relativa de x e y.
        info: origem do modelo (experimento, protocolo, seed, avaliacao LOSO...).
    """
    blocks = list(blocks) if blocks is not None else list(spec.block_slices())
    payload = {
        "formato": FORMATO,
        "modelo": {
            "nome": model_name,
            "input_size": int(len(block_columns(spec, blocks))),
            "num_classes": len(class_names),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        "classes": list(class_names),
        "features": {
            "pose_indices": [int(i) for i in spec.pose_indices],
            "use_pose": spec.use_pose,
            "use_hands": spec.use_hands,
            "use_presence_flags": spec.use_presence_flags,
            "blocks": blocks,
            "sequence_length": int(sequence_length),
            "normalize_anchor": bool(normalize),
        },
        "padronizacao": {
            "mean": torch.as_tensor(np.asarray(mean, np.float32).reshape(-1)),
            "std": torch.as_tensor(np.asarray(std, np.float32).reshape(-1)),
        },
        "extracao": dict(extracao),
        "versoes": {
            "torch": str(torch.__version__),  # TorchVersion nao passa no weights_only
            "numpy": np.__version__,
            "mediapipe": _versao("mediapipe"),
        },
        "info": info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


@dataclass
class ModeloTreinado:
    """Modelo carregado com o pre-processamento dos experimentos."""

    model: nn.Module
    class_names: list[str]
    spec: FeatureSpec
    colunas: np.ndarray
    sequence_length: int
    normalize: bool
    mean: np.ndarray
    std: np.ndarray
    extracao: dict[str, Any]
    info: dict[str, Any]
    device: torch.device

    def preparar(
        self, pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray
    ) -> np.ndarray:
        """Landmarks brutos de um trecho (F frames) -> entrada do modelo (T, D).

        Args:
            pose: (F, 33, 3), left_hand e right_hand: (F, 21, 3). NaN onde ausente.
        """
        matrix = build_frame_matrix(pose, left_hand, right_hand, self.spec, self.normalize)
        seq = resample_to_length(matrix, self.sequence_length)[:, self.colunas]
        return ((seq - self.mean) / self.std).astype(np.float32)

    @torch.no_grad()
    def probabilidades(
        self, pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray
    ) -> np.ndarray:
        """Probabilidade de cada classe, na ordem de `class_names`."""
        x = torch.from_numpy(self.preparar(pose, left_hand, right_hand))[None].to(self.device)
        return torch.softmax(self.model(x), dim=1)[0].cpu().numpy()

    def prever(
        self, pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray
    ) -> tuple[str, float]:
        """Classe mais provavel e a probabilidade dela."""
        p = self.probabilidades(pose, left_hand, right_hand)
        i = int(p.argmax())
        return self.class_names[i], float(p[i])


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> ModeloTreinado:
    """Carrega um checkpoint de `save_checkpoint`, em modo de avaliacao."""
    device = device or torch.device("cpu")
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ck.get("formato") != FORMATO:
        raise ValueError(f"formato de checkpoint {ck.get('formato')}, esperado {FORMATO}")

    f = ck["features"]
    spec = FeatureSpec(
        pose_indices=tuple(f["pose_indices"]),
        use_pose=f["use_pose"],
        use_hands=f["use_hands"],
        use_presence_flags=f["use_presence_flags"],
    )

    m = ck["modelo"]
    model = build_model(m["nome"], input_size=m["input_size"], num_classes=m["num_classes"])
    model.load_state_dict(m["state_dict"])
    model.to(device).eval()

    return ModeloTreinado(
        model=model,
        class_names=list(ck["classes"]),
        spec=spec,
        colunas=block_columns(spec, f["blocks"]),
        sequence_length=int(f["sequence_length"]),
        normalize=bool(f["normalize_anchor"]),
        mean=ck["padronizacao"]["mean"].numpy(),
        std=ck["padronizacao"]["std"].numpy(),
        extracao=ck["extracao"],
        info=ck["info"],
        device=device,
    )
