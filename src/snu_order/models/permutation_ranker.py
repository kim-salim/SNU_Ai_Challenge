from __future__ import annotations

import torch
from torch import nn

from snu_order.order.permutation24 import PERMS


class PermutationRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        scorer_hidden_dims: tuple[int, int] = (1024, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.perms = PERMS
        first_hidden, second_hidden = scorer_hidden_dims
        self.scorer = nn.Sequential(
            nn.Linear(10 * hidden_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.GELU(),
            nn.Dropout(min(dropout, 0.1)),
            nn.Linear(second_hidden, 1),
        )

    @staticmethod
    def _feature_for_perm(frame_tokens: torch.Tensor, perm: tuple[int, int, int, int]) -> torch.Tensor:
        x0 = frame_tokens[:, perm[0]]
        x1 = frame_tokens[:, perm[1]]
        x2 = frame_tokens[:, perm[2]]
        x3 = frame_tokens[:, perm[3]]
        return torch.cat(
            [
                x0,
                x1,
                x2,
                x3,
                x1 - x0,
                x2 - x1,
                x3 - x2,
                x2 - x0,
                x3 - x1,
                x3 - x0,
            ],
            dim=-1,
        )

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        if frame_tokens.ndim != 3 or frame_tokens.shape[1] != 4:
            raise ValueError(f"frame_tokens must have shape [B,4,H], got {tuple(frame_tokens.shape)}")
        if frame_tokens.shape[2] != self.hidden_dim:
            raise ValueError(f"frame token dim must be {self.hidden_dim}, got {frame_tokens.shape[2]}")
        scores = [self.scorer(self._feature_for_perm(frame_tokens, perm)) for perm in self.perms]
        return torch.cat(scores, dim=1)

