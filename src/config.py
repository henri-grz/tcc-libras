"""Carregamento de configuracao e resolucao de caminhos."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    """Configuracao do projeto, com caminhos ja resolvidos em absoluto."""

    raw: dict[str, Any] = field(default_factory=dict)

    # --- atalhos de acesso ---------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return list(self.raw["classes"])

    @property
    def extraction(self) -> dict[str, Any]:
        return self.raw["extraction"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def baseline_original(self) -> dict[str, Any]:
        return self.raw["baseline_original"]

    @property
    def reproducao(self) -> dict[str, Any]:
        """Parametros de subamostragem da reproducao. Nao sao do trabalho original."""
        return self.raw.get("reproducao", {"max_sequences": 8000, "max_points": 200_000})

    # --- caminhos ------------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve um caminho de `paths` relativo a raiz do projeto."""
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def videos_dir(self) -> Path:
        return self.path("videos")

    @property
    def cache_dir(self) -> Path:
        return self.path("cache")

    @property
    def landmarks_dir(self) -> Path:
        return self.cache_dir / "landmarks"

    @property
    def metadata_csv(self) -> Path:
        return self.cache_dir / "metadata.csv"

    @property
    def results_dir(self) -> Path:
        return self.path("results")

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.landmarks_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    """Le o YAML de configuracao. Sem argumento, usa configs/default.yaml."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw=raw)
    cfg.ensure_dirs()
    return cfg
