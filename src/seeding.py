"""Fixacao de seed. Corrige a falha S10 do pipeline original (execucao unica, seed unico)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fixa as seeds de random, numpy e torch.

    Com `deterministic=True`, forca kernels deterministicos do cuDNN. Isso deixa o
    treino um pouco mais lento, mas e o que permite reportar resultados reproduziveis.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
