from __future__ import annotations

import torch
from torch import nn

from snu_order.data.label import PAIRS


class PairwiseHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, pair_hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pairs = PAIRS
        self.scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim, pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, 1),
        )

    @staticmethod
    def _feature_for_pair(frame_tokens: torch.Tensor, i: int, j: int) -> torch.Tensor:
        xi = frame_tokens[:, i]
        xj = frame_tokens[:, j]
        return torch.cat([xi, xj, xi - xj, xi * xj], dim=-1)

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        if frame_tokens.ndim != 3 or frame_tokens.shape[1] != 4:
            raise ValueError(f"frame_tokens must have shape [B,4,H], got {tuple(frame_tokens.shape)}")
        if frame_tokens.shape[2] != self.hidden_dim:
            raise ValueError(f"frame token dim must be {self.hidden_dim}, got {frame_tokens.shape[2]}")
        logits = [self.scorer(self._feature_for_pair(frame_tokens, i, j)) for i, j in self.pairs]
        return torch.cat(logits, dim=1)

