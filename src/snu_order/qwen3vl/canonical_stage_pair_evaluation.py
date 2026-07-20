from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import torch

from .permutations import PERMS
from .stage_pair_scorer import structured_permutation_logits


@dataclass(frozen=True)
class StableRanking:
    scores: torch.Tensor
    order: torch.Tensor
    prediction: torch.Tensor
    gt_rank: torch.Tensor | None
    top1_margin: torch.Tensor


def canonical_cpu_float32_scores(
    stage_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    *,
    stage_weight: float = 1.0,
    pair_weight: float = 0.3,
) -> torch.Tensor:
    """The only Stage/Pair-to-24-way evaluation path used by Retention v3."""
    stage = stage_logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    pair = pair_logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return structured_permutation_logits(
        stage,
        pair,
        stage_weight=float(stage_weight),
        pair_weight=float(pair_weight),
    ).to(dtype=torch.float32).contiguous()


def stable_class_order(scores: torch.Tensor) -> torch.Tensor:
    values = scores.detach().to(device="cpu", dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != len(PERMS):
        raise ValueError(f"scores must have shape [N,{len(PERMS)}], got {tuple(values.shape)}")
    # Input columns are already in ascending class-index order. A stable descending
    # sort therefore resolves exact ties by the lower permutation class index.
    return torch.argsort(values, dim=1, descending=True, stable=True)


def stable_ranking(scores: torch.Tensor, targets: torch.Tensor | None = None) -> StableRanking:
    values = scores.detach().to(device="cpu", dtype=torch.float32).contiguous()
    order = stable_class_order(values)
    prediction = order[:, 0].long()
    top1_margin = values.gather(1, order[:, :1]).squeeze(1) - values.gather(1, order[:, 1:2]).squeeze(1)
    gt_rank: torch.Tensor | None = None
    if targets is not None:
        target_values = targets.detach().to(device="cpu", dtype=torch.long).view(-1)
        if target_values.shape[0] != values.shape[0]:
            raise ValueError("scores and targets must have the same row count")
        matches = order.eq(target_values[:, None])
        if not bool(matches.any(dim=1).all()):
            raise AssertionError("every target class must occur exactly once in the ranking")
        gt_rank = matches.to(torch.int64).argmax(dim=1).add(1)
    return StableRanking(values, order, prediction, gt_rank, top1_margin)


def semantic_prediction_sha256(
    ids: Sequence[str],
    ranking: StableRanking,
) -> str:
    if len(ids) != int(ranking.prediction.shape[0]):
        raise ValueError("ID and prediction row counts differ")
    payload = {
        "ids": [str(value) for value in ids],
        "prediction": ranking.prediction.tolist(),
        "gt_rank": None if ranking.gt_rank is None else ranking.gt_rank.tolist(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
