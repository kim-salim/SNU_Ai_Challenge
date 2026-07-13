from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .permutations import PAIRS, PERMS, answer_to_perm_index, pairwise_labels_from_answer, perm_index_to_answer


def class_position_table(*, device: torch.device | None = None) -> torch.Tensor:
    values = [[pos - 1 for pos in perm_index_to_answer(idx)] for idx in range(len(PERMS))]
    return torch.tensor(values, dtype=torch.long, device=device)


def pair_sign_table(*, device: torch.device | None = None) -> torch.Tensor:
    values: list[list[float]] = []
    positions = class_position_table(device=None)
    for class_idx in range(len(PERMS)):
        row = []
        for i, j in PAIRS:
            row.append(1.0 if int(positions[class_idx, i]) < int(positions[class_idx, j]) else -1.0)
        values.append(row)
    return torch.tensor(values, dtype=torch.float32, device=device)


def stage_targets_from_answer(answer: torch.Tensor) -> torch.Tensor:
    if answer.shape[-1] != 4:
        raise ValueError(f"answer must end with 4 values, got {tuple(answer.shape)}")
    targets = answer.long() - 1
    if bool(((targets < 0) | (targets > 3)).any()):
        raise ValueError("answer values must be in [1,4]")
    return targets


def pair_targets_from_answer(answer: torch.Tensor) -> torch.Tensor:
    if answer.ndim == 1:
        answer = answer.unsqueeze(0)
    if answer.shape[-1] != 4:
        raise ValueError(f"answer must have shape [B,4], got {tuple(answer.shape)}")
    rows = [pairwise_labels_from_answer(row.tolist()) for row in answer.detach().cpu()]
    return torch.tensor(rows, dtype=torch.float32, device=answer.device)


def stage_scores_from_logits(stage_logits: torch.Tensor) -> torch.Tensor:
    if stage_logits.ndim != 3 or tuple(stage_logits.shape[1:]) != (4, 4):
        raise ValueError(f"stage_logits must have shape [B,4,4], got {tuple(stage_logits.shape)}")
    log_probs = F.log_softmax(stage_logits, dim=-1)
    positions = class_position_table(device=stage_logits.device)
    gather_index = positions.unsqueeze(0).expand(stage_logits.shape[0], -1, -1).unsqueeze(-1)
    gathered = log_probs.unsqueeze(1).expand(-1, len(PERMS), -1, -1).gather(-1, gather_index).squeeze(-1)
    return gathered.mean(dim=-1)


def pair_scores_from_logits(pair_logits: torch.Tensor) -> torch.Tensor:
    if pair_logits.ndim != 2 or pair_logits.shape[1] != len(PAIRS):
        raise ValueError(f"pair_logits must have shape [B,6], got {tuple(pair_logits.shape)}")
    signs = pair_sign_table(device=pair_logits.device).to(pair_logits.dtype)
    return F.logsigmoid(pair_logits.unsqueeze(1) * signs.unsqueeze(0)).mean(dim=-1)


def structured_permutation_logits(
    stage_logits: torch.Tensor,
    pair_logits: torch.Tensor | None = None,
    *,
    stage_weight: float = 1.0,
    pair_weight: float = 0.3,
) -> torch.Tensor:
    final = float(stage_weight) * stage_scores_from_logits(stage_logits)
    if pair_logits is not None and float(pair_weight) != 0.0:
        final = final + float(pair_weight) * pair_scores_from_logits(pair_logits)
    return final


def canonical_remap_indices(shuffle_idx: torch.Tensor | list[int] | tuple[int, ...]) -> torch.Tensor:
    shuffle = torch.as_tensor(shuffle_idx, dtype=torch.long).view(-1)
    if shuffle.numel() != 4 or sorted(int(v) for v in shuffle.tolist()) != [0, 1, 2, 3]:
        raise ValueError(f"shuffle_idx must be a permutation of [0,1,2,3], got {shuffle.tolist()}")
    mapping = torch.empty((len(PERMS),), dtype=torch.long)
    for new_class_idx in range(len(PERMS)):
        new_answer = perm_index_to_answer(new_class_idx)
        canonical_answer = [0, 0, 0, 0]
        for new_slot, old_slot in enumerate(shuffle.tolist()):
            canonical_answer[int(old_slot)] = int(new_answer[new_slot])
        mapping[new_class_idx] = answer_to_perm_index(canonical_answer)
    return mapping


def remap_logits_to_canonical(logits: torch.Tensor, shuffle_idx: torch.Tensor | list[int] | tuple[int, ...]) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != len(PERMS):
        raise ValueError(f"logits must have shape [B,24], got {tuple(logits.shape)}")
    if torch.as_tensor(shuffle_idx).ndim == 1:
        mapping = canonical_remap_indices(shuffle_idx).to(logits.device)
        out = torch.empty_like(logits)
        out[:, mapping] = logits
        return out
    shuffle_batch = torch.as_tensor(shuffle_idx, dtype=torch.long, device=logits.device)
    if shuffle_batch.shape != (logits.shape[0], 4):
        raise ValueError(f"batched shuffle_idx must have shape [B,4], got {tuple(shuffle_batch.shape)}")
    rows = []
    for row, shuffle in zip(logits, shuffle_batch, strict=True):
        mapping = canonical_remap_indices(shuffle.detach().cpu()).to(logits.device)
        out_row = torch.empty_like(row)
        out_row[mapping] = row
        rows.append(out_row)
    return torch.stack(rows, dim=0)


def symmetric_kl_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    return 0.5 * (F.kl_div(log_p, q, reduction="batchmean") + F.kl_div(log_q, p, reduction="batchmean"))


@dataclass
class StagePairLossOutput:
    loss: torch.Tensor
    permutation_loss: torch.Tensor
    stage_loss: torch.Tensor
    pair_loss: torch.Tensor
    consistency_loss: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach().cpu()),
            "permutation_loss": float(self.permutation_loss.detach().cpu()),
            "stage_loss": float(self.stage_loss.detach().cpu()),
            "pair_loss": float(self.pair_loss.detach().cpu()),
            "consistency_loss": float(self.consistency_loss.detach().cpu()),
        }


class StagePairStructuredLoss(nn.Module):
    def __init__(
        self,
        *,
        permutation_weight: float = 1.0,
        stage_weight: float = 0.3,
        pair_weight: float = 0.2,
        consistency_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.permutation_weight = float(permutation_weight)
        self.stage_weight = float(stage_weight)
        self.pair_weight = float(pair_weight)
        self.consistency_weight = float(consistency_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_perm_idx: torch.Tensor,
        answer: torch.Tensor,
        *,
        consistency_logits: torch.Tensor | None = None,
    ) -> StagePairLossOutput:
        final_logits = outputs["final_logits"]
        stage_logits = outputs["stage_logits"]
        pair_logits = outputs.get("pair_logits")
        target_perm_idx = target_perm_idx.long().view(-1)
        answer = answer.long()
        perm_loss = F.cross_entropy(final_logits, target_perm_idx)
        stage_targets = stage_targets_from_answer(answer)
        stage_loss = F.cross_entropy(stage_logits.reshape(-1, 4), stage_targets.reshape(-1))
        if pair_logits is None:
            pair_loss = final_logits.new_zeros(())
        else:
            pair_loss = F.binary_cross_entropy_with_logits(pair_logits, pair_targets_from_answer(answer).to(pair_logits.dtype))
        if consistency_logits is None:
            consistency_loss = final_logits.new_zeros(())
        else:
            consistency_loss = symmetric_kl_loss(final_logits, consistency_logits)
        loss = (
            self.permutation_weight * perm_loss
            + self.stage_weight * stage_loss
            + self.pair_weight * pair_loss
            + self.consistency_weight * consistency_loss
        )
        return StagePairLossOutput(loss, perm_loss, stage_loss, pair_loss, consistency_loss)


def prediction_answers_from_logits(logits: torch.Tensor) -> list[list[int]]:
    return [perm_index_to_answer(int(idx)) for idx in logits.detach().float().argmax(dim=1).cpu().tolist()]
