"""Utilitarios compartilhados pelos scripts de experimento."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Permite rodar os scripts diretamente (python experiments/exp01_....py)
# alem de como modulo (python -m experiments.exp01_...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoint import save_checkpoint  # noqa: E402
from src.config import Config  # noqa: E402
from src.dataset import Samples, load_metadata, standardization_stats, standardize  # noqa: E402
from src.models import build_model  # noqa: E402
from src.seeding import set_seed  # noqa: E402
from src.splits import grouped_split  # noqa: E402
from src.train import train_model  # noqa: E402

# Metricas LOSO copiadas para dentro do checkpoint, como desempenho esperado do modelo.
METRICAS_LOSO = ("accuracy", "balanced_accuracy", "f1_macro")


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


def add_protocol_arg(ap: argparse.ArgumentParser) -> None:
    """Escolha do protocolo de treino do exp05 e do exp06.

    Sem a flag, vale `training.protocolo_loso` do YAML. Ver
    docs/04_revisao_protocolo_exp05.md.
    """
    ap.add_argument(
        "--protocolo",
        choices=("revisado", "original"),
        default=None,
        help="revisado: epocas fixas, sem validacao | original: early stopping na "
             "loss de validacao (padrao: training.protocolo_loso do YAML)",
    )


def stopping_kwargs(protocolo: str, tr_cfg: dict, epochs: int) -> dict:
    """Argumentos de parada de `train_model` para cada protocolo."""
    if protocolo == "revisado":
        return {"epochs": epochs, "monitor": "none"}
    return {
        "epochs": epochs,
        "monitor": "val_loss",
        "patience": int(tr_cfg["early_stopping_patience"]),
    }


def salvar_modelo_final(
    cfg: Config,
    samples: Samples,
    nome: str,
    model_name: str,
    seed: int,
    treino: dict,
    aggregate: dict,
    info: dict,
    smoke: bool,
    blocks: list[str] | None = None,
    transform_factory: Callable[[np.ndarray, np.ndarray], Callable] | None = None,
) -> Path:
    """Treina com TODOS os videos e grava o checkpoint em `<results>/modelos/`.

    Roda depois da avaliacao LOSO e nao altera nenhum resultado dela. E o modelo para
    uso em tempo real: os modelos dos folds deixam um sinalizador de fora e nao sao
    salvos. O desempenho esperado e a acuracia LOSO (`aggregate`), copiada para o
    arquivo. Em modo smoke o nome ganha o sufixo `_smoke`.

    Args:
        treino: argumentos de `train_model` (lote, taxa, device, parada). Com
            monitor="val_loss" (protocolo original), a validacao sai de um split
            agrupado por video com `training.val_size`.
        transform_factory: recebe media e desvio do treino e devolve o
            `batch_transform`; com ele, o treino recebe os videos sem padronizar (exp07).
    """
    set_seed(seed)
    if treino.get("monitor") == "val_loss":
        tr_idx, val_idx = grouped_split(
            samples.y, samples.groups, test_size=float(cfg.training["val_size"]), seed=seed
        )
    else:
        tr_idx, val_idx = np.arange(len(samples)), np.array([], dtype=np.int64)

    mean, std = standardization_stats(samples.X[tr_idx])
    X_tr, X_val = standardize(samples.X[tr_idx], samples.X[val_idx])
    model = build_model(model_name, input_size=samples.dim, num_classes=len(samples.class_names))
    if transform_factory is None:
        hist = train_model(model, X_tr, samples.y[tr_idx], X_val, samples.y[val_idx], **treino)
    else:
        hist = train_model(
            model, samples.X[tr_idx], samples.y[tr_idx], X_val, samples.y[val_idx],
            batch_transform=transform_factory(mean, std), **treino,
        )

    n_frames = load_metadata(cfg).set_index("video_id").loc[samples.groups, "n_frames"]
    path = cfg.results_dir / "modelos" / f"{nome}{'_smoke' if smoke else ''}.pth"
    save_checkpoint(
        path, model, model_name, samples.class_names, samples.spec, blocks,
        sequence_length=samples.seq_len,
        normalize=bool(cfg.features["normalize_anchor"]),
        mean=mean, std=std,
        extracao=cfg.extraction,
        info={
            **info,
            "smoke": smoke,
            "seed": seed,
            "parada": {k: treino[k] for k in ("epochs", "monitor", "patience") if k in treino},
            "epocas_treinadas": hist.stopped_epoch + 1,
            "batch_size": treino.get("batch_size"),
            "learning_rate": treino.get("learning_rate"),
            "videos_treino": samples.groups[tr_idx].tolist(),
            "videos_validacao": samples.groups[val_idx].tolist(),
            "train_acc_final": float(hist.train_acc[-1]),
            "frames_por_video": {
                "min": int(n_frames.min()),
                "mediana": float(n_frames.median()),
                "max": int(n_frames.max()),
            },
            "avaliacao_loso": {
                "n_runs": aggregate["n_runs"],
                **{k: {"mean": aggregate[k]["mean"], "std": aggregate[k]["std"]}
                   for k in METRICAS_LOSO if k in aggregate},
            },
        },
    )
    print(
        f"[modelo] {path} | {len(tr_idx)} videos de treino, "
        f"{hist.stopped_epoch + 1} epocas, acc_treino_final={hist.train_acc[-1] * 100:.2f}%"
    )
    return path
