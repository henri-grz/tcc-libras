"""Laco de treino com early stopping na validacao.

Correcao das falhas S3 e S4: o pipeline original nao tinha conjunto de validacao e
monitorava a loss de TREINO no early stopping. Aqui o monitor padrao e a loss de validacao e os
melhores pesos sao restaurados no fim.

O modo `monitor="train_loss"` existe apenas para reproduzir o comportamento original
no exp01.

O modo `monitor="none"` treina todas as epocas e mantem os pesos finais, sem
validacao. E o protocolo revisado do exp05 e do exp06, ver
docs/04_revisao_protocolo_exp05.md.

`batch_transform` aplica o aumento de dados do exp07 a cada lote de treino, e so ao
treino.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_epoch: int = -1

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
        }


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _eval_pass(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item() * yb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
    if total == 0:
        return float("nan"), float("nan")
    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    patience: int = 15,
    monitor: str = "val_loss",
    device: torch.device | None = None,
    verbose: bool = True,
    restore_best: bool = True,
    batch_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> History:
    """Treina o modelo e devolve o historico.

    Args:
        monitor: "val_loss" (early stopping na validacao), "train_loss" (comportamento
            do pipeline original) ou "none" (epocas fixas, pesos finais, protocolo
            revisado).
        patience: epocas sem melhora antes de parar. Ignorado com monitor="none".
        restore_best: com False, mantem os pesos da ultima epoca mesmo com early
            stopping, como o codigo de Prestes (Apendice B, p. 55).
        batch_transform: aplicada a cada lote de treino, ainda na CPU, antes do forward;
            nunca na validacao nem no teste (exp07). Com ela, `train_acc` mede o acerto
            nos lotes transformados.

    Returns:
        History. Os pesos do modelo ficam nos melhores encontrados pelo monitor, ou nos
        da ultima epoca com monitor="none" ou restore_best=False.
    """
    if monitor not in {"val_loss", "train_loss", "none"}:
        raise ValueError(f"monitor desconhecido: {monitor}")
    device = device or get_device()
    model = model.to(device)

    has_val = X_val is not None and y_val is not None and len(y_val) > 0
    if monitor == "val_loss" and not has_val:
        raise ValueError("monitor='val_loss' exige conjunto de validacao")

    train_loader = _loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _loader(X_val, y_val, batch_size, shuffle=False) if has_val else None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    hist = History()
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            if batch_transform is not None:
                xb = batch_transform(xb)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        tr_loss = running_loss / max(total, 1)
        tr_acc = correct / max(total, 1)
        hist.train_loss.append(tr_loss)
        hist.train_acc.append(tr_acc)

        if has_val:
            v_loss, v_acc = _eval_pass(model, val_loader, criterion, device)
            hist.val_loss.append(v_loss)
            hist.val_acc.append(v_acc)
        else:
            v_loss, v_acc = float("nan"), float("nan")

        if monitor == "none":
            hist.best_epoch = epoch
        else:
            score = v_loss if monitor == "val_loss" else tr_loss
            if score < best_score - 1e-6:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                hist.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            msg = f"[epoch {epoch + 1}/{epochs}] loss={tr_loss:.4f} acc={tr_acc:.3f}"
            if has_val:
                msg += f" | val_loss={v_loss:.4f} val_acc={v_acc:.3f}"
            print(msg)

        if epochs_without_improvement >= patience:
            hist.stopped_epoch = epoch
            if verbose:
                print(f"[early stopping] epoca {epoch + 1}, monitor={monitor}")
            break

    if monitor != "none" and restore_best:
        model.load_state_dict(best_state)
    if hist.stopped_epoch < 0:
        hist.stopped_epoch = epochs - 1
    return hist


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    """Devolve as classes preditas para X."""
    device = device or get_device()
    model = model.to(device).eval()
    preds: list[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)
