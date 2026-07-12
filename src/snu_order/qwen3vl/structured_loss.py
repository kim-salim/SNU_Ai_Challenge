from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .permutations import PAIRS, PERMS, pairwise_labels_from_answer, perm_index_to_answer


def build_position_mask(device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros((len(PERMS), 4, 4), dtype=torch.bool, device=device)
    for class_idx in range(len(PERMS)):
        answer = perm_index_to_answer(class_idx)
        for frame_idx, position in enumerate(answer):
            mask[class_idx, frame_idx, position - 1] = True
    return mask


def build_pairwise_mask(device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros((len(PERMS), len(PAIRS)), dtype=torch.bool, device=device)
    for class_idx in range(len(PERMS)):
        labels = pairwise_labels_from_answer(perm_index_to_answer(class_idx))
        for pair_idx, label in enumerate(labels):
            mask[class_idx, pair_idx] = bool(label)
    return mask


def _masked_logsumexp(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 1 or mask.shape[0] != log_probs.shape[-1]:
        raise ValueError(f"mask shape {tuple(mask.shape)} is incompatible with log_probs {tuple(log_probs.shape)}")
    selected = log_probs.masked_fill(~mask.unsqueeze(0), torch.finfo(log_probs.dtype).min)
    return torch.logsumexp(selected, dim=-1)


@dataclass
class StructuredLossOutput:
    loss: torch.Tensor
    permutation_loss: torch.Tensor
    pairwise_marginal_loss: torch.Tensor
    position_marginal_loss: torch.Tensor

    def as_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach().cpu()),
            "permutation_loss": float(self.permutation_loss.detach().cpu()),
            "pairwise_marginal_loss": float(self.pairwise_marginal_loss.detach().cpu()),
            "position_marginal_loss": float(self.position_marginal_loss.detach().cpu()),
        }


class StructuredPermutationLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        permutation_weight: float = 1.0,
        pairwise_marginal_weight: float = 0.2,
        position_marginal_weight: float = 0.1,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.permutation_weight = float(permutation_weight)
        self.pairwise_marginal_weight = float(pairwise_marginal_weight)
        self.position_marginal_weight = float(position_marginal_weight)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("position_mask", build_position_mask(), persistent=False)
        self.register_buffer("pairwise_mask", build_pairwise_mask(), persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        target_perm_idx: torch.Tensor,
        answers: torch.Tensor,
    ) -> StructuredLossOutput:
        if logits.ndim != 2 or logits.shape[1] != len(PERMS):
            raise ValueError(f"logits must have shape [B,24], got {tuple(logits.shape)}")
        if answers.ndim != 2 or answers.shape[1] != 4:
            raise ValueError(f"answers must have shape [B,4], got {tuple(answers.shape)}")
        logits_f = logits.float()
        target = target_perm_idx.long().view(-1)
        answers_l = answers.long()
        if logits_f.shape[0] != target.shape[0] or logits_f.shape[0] != answers_l.shape[0]:
            raise ValueError("logits, target_perm_idx, and answers batch sizes must match")

        permutation_loss = F.cross_entropy(logits_f, target, label_smoothing=self.label_smoothing)
        log_probs = F.log_softmax(logits_f, dim=-1)

        position_terms = []
        for frame_idx in range(4):
            true_positions = answers_l[:, frame_idx] - 1
            for position in range(4):
                row_mask = true_positions == position
                if not bool(row_mask.any()):
                    continue
                class_mask = self.position_mask[:, frame_idx, position].to(log_probs.device)
                position_terms.append(-_masked_logsumexp(log_probs[row_mask], class_mask))
        position_loss = torch.cat(position_terms).mean() if position_terms else logits_f.new_tensor(0.0)

        pairwise_terms = []
        for pair_idx, (i, j) in enumerate(PAIRS):
            true_before = answers_l[:, i] < answers_l[:, j]
            for relation in (False, True):
                row_mask = true_before == relation
                if not bool(row_mask.any()):
                    continue
                class_mask = (self.pairwise_mask[:, pair_idx] == relation).to(log_probs.device)
                pairwise_terms.append(-_masked_logsumexp(log_probs[row_mask], class_mask))
        pairwise_loss = torch.cat(pairwise_terms).mean() if pairwise_terms else logits_f.new_tensor(0.0)

        total = (
            self.permutation_weight * permutation_loss
            + self.pairwise_marginal_weight * pairwise_loss
            + self.position_marginal_weight * position_loss
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Structured permutation loss produced NaN/Inf")
        return StructuredLossOutput(total, permutation_loss, pairwise_loss, position_loss)
