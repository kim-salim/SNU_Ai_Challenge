from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FrameProjector(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        quality_dim: int = 9,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if quality_dim < 0:
            raise ValueError("quality_dim must be non-negative")
        input_dim = embedding_dim * 4 + 1 + quality_dim
        self.embedding_dim = embedding_dim
        self.quality_dim = quality_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        frame_emb: torch.Tensor,
        quality: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_emb.ndim != 2:
            raise ValueError(f"text_emb must have shape [B,D], got {tuple(text_emb.shape)}")
        if frame_emb.ndim != 3 or frame_emb.shape[1] != 4:
            raise ValueError(f"frame_emb must have shape [B,4,D], got {tuple(frame_emb.shape)}")
        if frame_emb.shape[0] != text_emb.shape[0] or frame_emb.shape[2] != text_emb.shape[1]:
            raise ValueError("text_emb and frame_emb batch/embedding dimensions must match")
        if quality is None:
            quality = frame_emb.new_zeros(frame_emb.shape[0], 4, self.quality_dim)
        if quality.ndim != 3 or quality.shape[:2] != frame_emb.shape[:2]:
            raise ValueError(f"quality must have shape [B,4,Q], got {tuple(quality.shape)}")
        if quality.shape[2] != self.quality_dim:
            raise ValueError(f"quality last dim must be {self.quality_dim}, got {quality.shape[2]}")

        text = text_emb.unsqueeze(1).expand(-1, 4, -1)
        cosine = F.cosine_similarity(frame_emb, text, dim=-1).unsqueeze(-1)
        feat = torch.cat([frame_emb, text, frame_emb - text, frame_emb * text, cosine, quality], dim=-1)
        return self.net(feat)

