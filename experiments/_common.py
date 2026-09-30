"""Utilitarios compartilhados pelos scripts de experimento."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite rodar os scripts diretamente (python experiments/exp01_....py)
# alem de como modulo (python -m experiments.exp01_...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="caminho do YAML de configuracao")
    ap.add_argument("--cpu", action="store_true", help="forca CPU mesmo com GPU disponivel")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="modo rapido: poucas epocas e poucas sequencias, so para validar o codigo",
    )
    return ap


def banner(title: str, subtitle: str = "") -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}")
    if subtitle:
        print(subtitle)
    print(line)
