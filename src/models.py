"""Modelos. Inclui a replica fiel do modelo do pipeline original e as versoes corrigidas.

`LSTMOriginal` reproduz exatamente a classe do Apendice B, inclusive dois pontos do diagnostico:
`dropout=0.3` com `num_layers=1` (que o PyTorch ignora, falha S1) e a camada `lstm2`
bidirecional com duas camadas internas, que o Quadro 3 do trabalho nao menciona
(falha S2). Nao corrigir nada aqui: esta classe existe para reproduzir o pipeline original.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMOriginal(nn.Module):
    """Replica fiel da arquitetura de Prestes (2024), Apendice B, p. 52-53.

    Mantida exatamente como no original para permitir a reproducao. Ver `LSTMCorrected`
    para a versao com dropout efetivo.
    """

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        # dropout ignorado pelo PyTorch quando num_layers=1 (falha S1)
        self.lstm1 = nn.LSTM(input_size, 30, batch_first=True, dropout=0.3, num_layers=1)
        self.lstm2 = nn.LSTM(30, 64, batch_first=True, dropout=0.3, num_layers=2, bidirectional=True)
        self.lstm3 = nn.LSTM(64 * 2, 64, batch_first=True, dropout=0.3, num_layers=1)
        self.lstm4 = nn.LSTM(64, 64, batch_first=True, dropout=0.3, num_layers=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class LSTMCorrected(nn.Module):
    """LSTM empilhada com dropout efetivo e pooling explicito.

    Diferencas em relacao a `LSTMOriginal`:
    - dropout aplicado de fato, entre camadas do mesmo modulo LSTM e antes do head
    - profundidade declarada por parametro, sem camadas fantasma
    - pooling escolhivel: ultimo passo ou media temporal
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pooling: str = "last",
    ):
        super().__init__()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling deve ser 'last' ou 'mean'")
        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1) if self.pooling == "mean" else out[:, -1, :]
        return self.head(pooled)


class SinglePointMLP(nn.Module):
    """Classificador sobre um unico ponto 3D. Usado pelo exp03.

    Serve de referencia trivial e de teste da falha C1: se este modelo chegar perto da
    acuracia do pipeline original, a modelagem temporal da LSTM nao esta contribuindo,
    e o numero medido reflete classificacao estatica de nuvem de pontos com vazamento.
    """

    def __init__(self, input_size: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # aceita (B, T, D) achatando para o ultimo passo
            x = x[:, -1, :]
        return self.net(x)


def build_model(name: str, input_size: int, num_classes: int, **kwargs) -> nn.Module:
    """Fabrica de modelos por nome, para os scripts de experimento."""
    registry = {
        "lstm_original": lambda: LSTMOriginal(input_size, num_classes),
        "lstm_corrected": lambda: LSTMCorrected(input_size, num_classes, **kwargs),
        "bilstm_corrected": lambda: LSTMCorrected(
            input_size, num_classes, bidirectional=True, **kwargs
        ),
        "single_point_mlp": lambda: SinglePointMLP(input_size, num_classes, **kwargs),
    }
    if name not in registry:
        raise KeyError(f"modelo '{name}' desconhecido. disponiveis: {list(registry)}")
    return registry[name]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
