"""Metricas e persistencia de resultados.

Correcao da falha S6: o baseline reportava, na coluna "Acuracia por classe", a formula
TP/(TP+FN), que e recall. Nos Quadros 4 e 5 do trabalho original as colunas "Acuracia" e
"Recall" sao identicas em todas as linhas por causa disso. Aqui cada metrica por classe
e reportada com o nome correto, e nao existe "acuracia por classe".
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Metricas globais e por classe, com nomes corretos."""
    labels = list(range(len(class_names)))

    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    return {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Acuracia balanceada e a metrica honesta com classes desbalanceadas.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "chance_level": float(1.0 / max(len(class_names), 1)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
        "per_class": [
            {
                "classe": class_names[i],
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
            }
            for i in labels
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
    }


def aggregate_runs(runs: list[dict[str, Any]], keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Media e desvio padrao de varias execucoes (folds ou seeds).

    Correcao da falha S10: o baseline reportava uma unica execucao com seed unico.
    """
    keys = keys or (
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    )
    out: dict[str, Any] = {"n_runs": len(runs)}
    for k in keys:
        vals = np.array([r[k] for r in runs if k in r], dtype=float)
        if vals.size == 0:
            continue
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "values": [float(v) for v in vals],
        }
    return out


def save_result(
    name: str, payload: dict[str, Any], results_dir: Path, echo: bool = True
) -> Path:
    """Grava o resultado em JSON. Nenhum experimento reporta so no stdout."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {"experimento": name, "timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
    path = results_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if echo:
        print(f"\n[resultado] {path}")
    return path


def plot_confusion_matrix(
    cm: list[list[int]], class_names: list[str], out_path: Path, title: str = ""
) -> Path | None:
    """Salva a matriz de confusao como PNG.

    Correcao da falha S11: o baseline plotava mas nunca apresentou no texto.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib ausente, matriz de confusao nao gerada")
        return None

    arr = np.array(cm, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = arr / arr.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(class_names), 1.0 + 0.6 * len(class_names)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    ax.set_xlabel("Predicao")
    ax.set_ylabel("Classe verdadeira")
    if title:
        ax.set_title(title, fontsize=10)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if arr[i, j] > 0:
                ax.text(
                    j, i, int(arr[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def print_summary(metrics: dict[str, Any], header: str = "") -> None:
    """Resumo legivel no terminal."""
    if header:
        print(f"\n=== {header} ===")
    print(f"amostras de teste     : {metrics['n_test']}")
    print(f"acuracia              : {metrics['accuracy'] * 100:.2f}%")
    print(f"acuracia balanceada   : {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"nivel de chance       : {metrics['chance_level'] * 100:.2f}%")
    print(f"f1 macro              : {metrics['f1_macro'] * 100:.2f}%")
    print(f"f1 ponderado          : {metrics['f1_weighted'] * 100:.2f}%")
    print("\npor classe (precision / recall / f1 / n):")
    for row in metrics["per_class"]:
        print(
            f"  {row['classe']:<10} "
            f"{row['precision'] * 100:6.2f} / {row['recall'] * 100:6.2f} / "
            f"{row['f1'] * 100:6.2f} / {row['support']:3d}"
        )
