# src/models/gru.py
from __future__ import annotations

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    Encodes the past with a GRU and predicts the next `horizon` steps.
    Input:  x [B, L, F]
    Output: y [B, H, T]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        target_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.target_dim = target_dim

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Linear(hidden_size, horizon * target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, _ = self.encoder(x)      # [B, L, H]
        last = feats[:, -1]             # [B, H]
        y = self.head(last)             # [B, H*T]
        return y.view(x.size(0), self.horizon, self.target_dim)