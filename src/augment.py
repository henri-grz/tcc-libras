"""Aumento de dados para sequencias de landmarks (exp07).

Opera em lotes (B, T, D) no espaco dos landmarks ja normalizados pelos ombros, antes da
padronizacao. Aplicado so no treino, dentro de cada fold, depois da separacao: se fosse
feito antes, versoes de um mesmo video cairiam no treino e no teste, e o vazamento da
falha C2 voltaria. Parametros e registro previo em docs/06_protocolo_exp07_aumento.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import Config
from .landmarks_spec import COORDS, FeatureSpec


@dataclass(frozen=True)
class ParamsAumento:
    """Intensidades do aumento. Os valores reais vem do bloco `aumento` do YAML."""

    corte_temporal_max: float = 0.10
    rotacao_graus: float = 15.0
    escala: tuple[float, float] = (0.9, 1.1)
    ruido_std: float = 0.01
    # Altura / largura do quadro da extracao (768 / 1024). O MediaPipe devolve x como
    # fracao da largura e y como fracao da altura, entao as duas unidades diferem.
    aspecto: float = 1.0

    @classmethod
    def from_config(cls, cfg: Config) -> "ParamsAumento":
        a, ext = cfg.raw["aumento"], cfg.extraction
        return cls(
            corte_temporal_max=float(a["corte_temporal_max"]),
            rotacao_graus=float(a["rotacao_graus"]),
            escala=(float(a["escala"][0]), float(a["escala"][1])),
            ruido_std=float(a["ruido_std"]),
            aspecto=float(ext["frame_height"]) / float(ext["frame_width"]),
        )


def colunas_de_coordenadas(spec: FeatureSpec) -> int:
    """Quantas colunas do inicio do vetor sao coordenadas (pose e maos, sem as flags)."""
    n = max(s.stop for nome, s in spec.block_slices().items() if nome != "flags")
    assert n % COORDS == 0, "as coordenadas devem vir em trios x, y, z"
    return n


def corte_temporal(x: torch.Tensor, frac_max: float, gen: torch.Generator) -> torch.Tensor:
    """Remove ate `frac_max` do inicio e do fim e reamostra para o mesmo T (linear)."""
    b, t, d = x.shape
    if frac_max <= 0 or t < 2:
        return x
    ini = torch.rand(b, generator=gen) * frac_max
    fim = 1.0 - torch.rand(b, generator=gen) * frac_max
    pos = (ini[:, None] + (fim - ini)[:, None] * torch.linspace(0.0, 1.0, t)[None, :]) * (t - 1)
    i0 = pos.floor().long().clamp(0, t - 1)
    i1 = (i0 + 1).clamp(max=t - 1)
    w = (pos - i0.float()).unsqueeze(-1)
    a = torch.gather(x, 1, i0.unsqueeze(-1).expand(b, t, d))
    c = torch.gather(x, 1, i1.unsqueeze(-1).expand(b, t, d))
    return a * (1 - w) + c * w


def aumentar_lote(
    x: torch.Tensor, spec: FeatureSpec, p: ParamsAumento, gen: torch.Generator
) -> torch.Tensor:
    """Corte temporal, rotacao e escala no plano x-y e ruido. Devolve um tensor novo.

    `x` e `gen` ficam na CPU. A rotacao e feita com y convertido para a unidade de x
    (`p.aspecto`): girar as fracoes de largura e de altura diretamente tambem deformaria
    o corpo. Ponto ausente (todas as coordenadas zero) continua zero. As flags so mudam
    pela reamostragem do corte temporal.
    """
    x = corte_temporal(x, p.corte_temporal_max, gen)
    b, t, _ = x.shape
    n = colunas_de_coordenadas(spec)
    pts = x[:, :, :n].reshape(b, t, n // COORDS, COORDS).clone()
    presente = pts.ne(0).any(dim=-1, keepdim=True)

    ang = (torch.rand(b, generator=gen) * 2 - 1) * math.radians(p.rotacao_graus)
    lo, hi = p.escala
    sx = (lo + torch.rand(b, generator=gen) * (hi - lo))[:, None, None]
    sy = (lo + torch.rand(b, generator=gen) * (hi - lo))[:, None, None]
    cos, sin = torch.cos(ang)[:, None, None], torch.sin(ang)[:, None, None]
    px = pts[..., 0].clone()  # copia: a linha de y le o x antes de girar
    py = pts[..., 1] * p.aspecto  # y em unidades de x: o quadro nao e quadrado
    pts[..., 0] = (cos * px - sin * py) * sx
    pts[..., 1] = (sin * px + cos * py) / p.aspecto * sy

    ruido = torch.randn(pts.shape, generator=gen) * p.ruido_std
    pts = torch.where(presente, pts + ruido, torch.zeros_like(pts))

    out = x.clone()
    out[:, :, :n] = pts.reshape(b, t, n)
    return out
