from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PermPairLoss(nn.Module):
    def __init__(self, pair_aux_weight: float = 0.3, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.pair_aux_weight = float(pair_aux_weight)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        perm_scores: torch.Tensor,
        target_perm_idx: torch.Tensor,
        pair_logits: torch.Tensor | None = None,
        pairwise_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        perm_loss = self.ce(perm_scores, target_perm_idx.long())
        if pair_logits is not None and pairwise_labels is not None and self.pair_aux_weight > 0:
            pair_loss = self.bce(pair_logits, pairwise_labels.float())
            total = perm_loss + self.pair_aux_weight * pair_loss
        else:
            pair_loss = perm_scores.new_tensor(0.0)
            total = perm_loss
        metrics = {
            "loss": float(total.detach().cpu()),
            "perm_loss": float(perm_loss.detach().cpu()),
            "pair_loss": float(pair_loss.detach().cpu()),
        }
        return total, metrics


def compute_perm_pair_loss(
    perm_scores: torch.Tensor,
    target_perm_idx: torch.Tensor,
    pair_logits: torch.Tensor,
    pairwise_labels: torch.Tensor,
    pair_aux_weight: float = 0.3,
    label_smoothing: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    perm_loss = F.cross_entropy(perm_scores, target_perm_idx.long(), label_smoothing=label_smoothing)
    pair_loss = F.binary_cross_entropy_with_logits(pair_logits, pairwise_labels.float())
    total = perm_loss + pair_aux_weight * pair_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "perm_loss": float(perm_loss.detach().cpu()),
        "pair_loss": float(pair_loss.detach().cpu()),
    }

