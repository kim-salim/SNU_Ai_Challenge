from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from snu_order.utils.config import get_by_path

from .champion_retention import CACHE_FORMAT, file_sha256, remap_teacher_logits_for_input_shuffle
from .permutations import PAIRS
from .stage_pair_scorer import pair_targets_from_answer, stage_targets_from_answer, structured_permutation_logits


PREREGISTERED_RESCUE_COVERAGE_FLOOR = 0.03


def apply_preregistered_component_coverage(
    weights: dict[str, float], coverage_policy: dict[str, Any]
) -> dict[str, float]:
    floor = float(coverage_policy.get("minimum_component_rescue_fraction", -1.0))
    if abs(floor - PREREGISTERED_RESCUE_COVERAGE_FLOOR) > 1e-12:
        raise RuntimeError("Retention v3 rescue coverage floor differs from preregistration")
    coverage = coverage_policy.get("rescue_coverage")
    eligible = coverage_policy.get("eligible_rescue_components")
    expected_keys = {"stage", "pair", "permutation"}
    if not isinstance(coverage, dict) or set(coverage) != expected_keys:
        raise RuntimeError("Retention v3 component rescue coverage is malformed")
    if not isinstance(eligible, dict) or set(eligible) != expected_keys:
        raise RuntimeError("Retention v3 component-cache eligibility is malformed")
    recomputed = {
        component: float(coverage[component]) >= PREREGISTERED_RESCUE_COVERAGE_FLOOR
        for component in expected_keys
    }
    if {key: bool(value) for key, value in eligible.items()} != recomputed:
        raise RuntimeError("Retention v3 component eligibility does not match preregistered coverage")
    effective = dict(weights)
    for component, is_eligible in recomputed.items():
        if not is_eligible:
            effective[f"rescue_{component}"] = 0.0
    return effective


class ComponentReferenceStore:
    """Strict ID-bound Stage/Pair cache used only for teacher masks or v1 protection."""

    def __init__(
        self,
        path: str | Path,
        *,
        prefix: str,
        expected_ids: Sequence[str],
        expected_split_sha256: str,
        expected_heads_sha256: str | None = None,
    ) -> None:
        if prefix not in {"teacher", "reference"}:
            raise ValueError(f"Unsupported component cache prefix: {prefix}")
        self.path = Path(path)
        self.prefix = prefix
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        required = {
            "cache_format", "ids", "stage_logits", "pair_logits", "target_perm_idx",
            "answer", "teacher_correct", "identity",
        }
        if not isinstance(payload, dict) or set(payload) != required or payload["cache_format"] != CACHE_FORMAT:
            raise RuntimeError(f"{prefix} component cache schema mismatch")
        ids = [str(value) for value in payload["ids"]]
        expected = [str(value) for value in expected_ids]
        if ids != expected or len(ids) != len(set(ids)):
            raise RuntimeError(f"{prefix} component cache ID ordering mismatch")
        identity = payload["identity"]
        if not isinstance(identity, dict) or identity.get("train_split_sha256") != expected_split_sha256:
            raise RuntimeError(f"{prefix} component cache split binding mismatch")
        observed_heads = str(identity.get("teacher_heads_sha256", ""))
        if expected_heads_sha256 and observed_heads != str(expected_heads_sha256):
            raise RuntimeError(
                f"{prefix} heads binding mismatch: expected={expected_heads_sha256} observed={observed_heads}"
            )
        n = len(ids)
        self.ids = ids
        self.stage_logits = payload["stage_logits"].float().contiguous()
        self.pair_logits = payload["pair_logits"].float().contiguous()
        self.answer = payload["answer"].long().contiguous()
        self.target_perm_idx = payload["target_perm_idx"].long().contiguous()
        if tuple(self.stage_logits.shape) != (n, 4, 4) or tuple(self.pair_logits.shape) != (n, len(PAIRS)):
            raise RuntimeError(f"{prefix} component cache logit shape mismatch")
        self.identity = identity
        self.sha256 = file_sha256(self.path)
        self._index = {sample_id: index for index, sample_id in enumerate(ids)}

    def sample(self, sample_id: str, shuffle_idx: Sequence[int] | None) -> dict[str, torch.Tensor]:
        index = self._index[str(sample_id)]
        stage, pair = remap_teacher_logits_for_input_shuffle(
            self.stage_logits[index], self.pair_logits[index], shuffle_idx
        )
        final = structured_permutation_logits(
            stage.unsqueeze(0), pair.unsqueeze(0), stage_weight=1.0, pair_weight=0.3
        )[0]
        return {
            f"{self.prefix}_stage_logits": stage,
            f"{self.prefix}_pair_logits": pair,
            f"{self.prefix}_final_logits": final,
        }


def component_safe_aux_scale(progress: float) -> float:
    value = min(1.0, max(0.0, float(progress)))
    if value < 0.10 or value >= 0.70:
        return 0.0
    if value < 0.15:
        return (value - 0.10) / 0.05
    if value < 0.40:
        return 1.0
    decay_progress = (value - 0.40) / 0.30
    return 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def _gt_margin_multiclass(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    target_values = targets.long()
    gt = values.gather(-1, target_values.unsqueeze(-1)).squeeze(-1)
    masked = values.scatter(-1, target_values.unsqueeze(-1), float("-inf"))
    return gt - masked.max(dim=-1).values


def _masked_component_mean(values: torch.Tensor, mask: torch.Tensor, *, per_item_cap: float) -> tuple[torch.Tensor, int]:
    weights = mask.to(device=values.device, dtype=values.dtype).clamp(max=float(per_item_cap))
    active = int(weights.gt(0).sum().item())
    if active == 0:
        return values.new_zeros(()), 0
    return (values * weights).sum() / weights.sum(), active


@dataclass(frozen=True)
class ComponentSafeLossOutput:
    loss: torch.Tensor
    uncapped_loss: torch.Tensor
    schedule_scale: float
    rescue_stage: torch.Tensor
    rescue_pair: torch.Tensor
    rescue_permutation: torch.Tensor
    protect_stage: torch.Tensor
    protect_pair: torch.Tensor
    protect_permutation: torch.Tensor
    active_counts: dict[str, int]


def component_safe_retention_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    cfg: dict[str, Any],
    *,
    base_loss: torch.Tensor,
    progress: float,
) -> ComponentSafeLossOutput:
    zero = outputs["final_logits"].new_zeros(())
    scale = component_safe_aux_scale(progress)
    names = (
        "rescue_stage", "rescue_pair", "rescue_permutation",
        "protect_stage", "protect_pair", "protect_permutation",
    )
    if not bool(get_by_path(cfg, "retention.enabled", False)) or scale == 0.0:
        return ComponentSafeLossOutput(zero, zero, scale, zero, zero, zero, zero, zero, zero, {name: 0 for name in names})
    if str(get_by_path(cfg, "retention.mode", "")) != "component_safe_v3":
        raise RuntimeError("component-safe loss requires retention.mode=component_safe_v3")
    if bool(get_by_path(cfg, "retention.soft_kl", False)):
        raise RuntimeError("Retention v3 forbids soft KL")
    required = (
        "teacher_stage_logits", "teacher_pair_logits", "teacher_final_logits",
        "reference_stage_logits", "reference_pair_logits", "reference_final_logits",
        "answer", "target_perm_idx",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise RuntimeError(f"component-safe batch is missing: {missing}")

    device = outputs["final_logits"].device
    stage_target = stage_targets_from_answer(batch["answer"]).to(device)
    pair_target = pair_targets_from_answer(batch["answer"]).to(device)
    perm_target = batch["target_perm_idx"].long().to(device)
    teacher_stage = batch["teacher_stage_logits"].float().to(device)
    teacher_pair = batch["teacher_pair_logits"].float().to(device)
    teacher_final = batch["teacher_final_logits"].float().to(device)
    reference_stage = batch["reference_stage_logits"].float().to(device)
    reference_pair = batch["reference_pair_logits"].float().to(device)
    reference_final = batch["reference_final_logits"].float().to(device)

    teacher_stage_correct = teacher_stage.argmax(dim=-1).eq(stage_target)
    teacher_pair_signed = teacher_pair * pair_target.mul(2).sub(1)
    teacher_pair_correct = teacher_pair_signed.gt(0)
    teacher_perm_margin = _gt_margin_multiclass(teacher_final, perm_target)
    teacher_perm_correct = teacher_final.argmax(dim=-1).eq(perm_target)
    teacher_perm_safe = teacher_perm_correct & teacher_perm_margin.ge(
        float(get_by_path(cfg, "retention.thresholds.teacher_permutation_margin_min"))
    )
    teacher_stage_weight = teacher_stage.softmax(dim=-1).gather(
        -1, stage_target.unsqueeze(-1)
    ).squeeze(-1).detach()
    teacher_pair_weight = torch.sigmoid(teacher_pair_signed).detach()
    teacher_perm_weight = teacher_final.softmax(dim=-1).gather(
        -1, perm_target.unsqueeze(-1)
    ).squeeze(-1).detach()

    reference_stage_margin = _gt_margin_multiclass(reference_stage, stage_target)
    reference_pair_margin = reference_pair * pair_target.mul(2).sub(1)
    reference_perm_margin = _gt_margin_multiclass(reference_final, perm_target)
    reference_stage_correct = reference_stage.argmax(dim=-1).eq(stage_target)
    reference_pair_correct = reference_pair_margin.gt(0)
    reference_perm_correct = reference_final.argmax(dim=-1).eq(perm_target)

    stage_low = float(get_by_path(cfg, "retention.thresholds.reference_stage_q30"))
    stage_high = float(get_by_path(cfg, "retention.thresholds.reference_stage_q50"))
    pair_low = float(get_by_path(cfg, "retention.thresholds.reference_pair_q30"))
    pair_high = float(get_by_path(cfg, "retention.thresholds.reference_pair_q50"))
    perm_low = float(get_by_path(cfg, "retention.thresholds.reference_permutation_q30"))
    perm_high = float(get_by_path(cfg, "retention.thresholds.reference_permutation_q50"))

    rescue_stage_mask = teacher_stage_correct & (~reference_stage_correct | reference_stage_margin.lt(stage_low))
    rescue_pair_mask = teacher_pair_correct & (~reference_pair_correct | reference_pair_margin.lt(pair_low))
    rescue_perm_mask = teacher_perm_safe & (~reference_perm_correct | reference_perm_margin.lt(perm_low))
    protect_stage_base = reference_stage_correct & reference_stage_margin.gt(stage_high)
    protect_pair_base = reference_pair_correct & reference_pair_margin.gt(pair_high)
    protect_perm_base = reference_perm_correct & reference_perm_margin.gt(perm_high)

    student_stage = outputs["stage_logits"].float()
    student_pair = outputs["pair_logits"].float()
    student_final = outputs["final_logits"].float()
    stage_pgt = student_stage.softmax(dim=-1).gather(-1, stage_target.unsqueeze(-1)).squeeze(-1)
    pair_pgt = torch.sigmoid(student_pair * pair_target.mul(2).sub(1))
    perm_pgt = student_final.softmax(dim=-1).gather(-1, perm_target.unsqueeze(-1)).squeeze(-1)
    protect_tau = float(get_by_path(cfg, "retention.protect_probability_threshold", 0.5))
    protect_stage_mask = protect_stage_base & stage_pgt.lt(protect_tau)
    protect_pair_mask = protect_pair_base & pair_pgt.lt(protect_tau)
    protect_perm_mask = protect_perm_base & perm_pgt.lt(protect_tau)

    gamma = float(get_by_path(cfg, "retention.focal_gamma", 1.0))
    stage_ce = F.cross_entropy(student_stage.reshape(-1, 4), stage_target.reshape(-1), reduction="none").reshape_as(stage_target)
    pair_bce = F.binary_cross_entropy_with_logits(student_pair, pair_target.to(student_pair.dtype), reduction="none")
    perm_ce = F.cross_entropy(student_final, perm_target, reduction="none")
    stage_focal = (1.0 - stage_pgt).detach().pow(gamma)
    pair_focal = (1.0 - pair_pgt).detach().pow(gamma)
    perm_focal = (1.0 - perm_pgt).detach().pow(gamma)
    item_cap = float(get_by_path(cfg, "retention.per_sample_aux_weight_cap", 1.0))

    rescue_stage, n_rs = _masked_component_mean(
        stage_ce * stage_focal,
        rescue_stage_mask.to(teacher_stage_weight.dtype) * teacher_stage_weight,
        per_item_cap=item_cap,
    )
    rescue_pair, n_rp = _masked_component_mean(
        pair_bce * pair_focal,
        rescue_pair_mask.to(teacher_pair_weight.dtype) * teacher_pair_weight,
        per_item_cap=item_cap,
    )
    rescue_perm, n_rpi = _masked_component_mean(
        perm_ce * perm_focal,
        rescue_perm_mask.to(teacher_perm_weight.dtype) * teacher_perm_weight,
        per_item_cap=item_cap,
    )
    protect_stage, n_ps = _masked_component_mean(stage_ce, protect_stage_mask, per_item_cap=item_cap)
    protect_pair, n_pp = _masked_component_mean(pair_bce, protect_pair_mask, per_item_cap=item_cap)
    protect_perm, n_ppi = _masked_component_mean(perm_ce, protect_perm_mask, per_item_cap=item_cap)

    uncapped = float(scale) * (
        float(get_by_path(cfg, "retention.weights.rescue_stage", 0.10)) * rescue_stage
        + float(get_by_path(cfg, "retention.weights.rescue_pair", 0.05)) * rescue_pair
        + float(get_by_path(cfg, "retention.weights.rescue_permutation", 0.10)) * rescue_perm
        + float(get_by_path(cfg, "retention.weights.protect_stage", 0.05)) * protect_stage
        + float(get_by_path(cfg, "retention.weights.protect_pair", 0.025)) * protect_pair
        + float(get_by_path(cfg, "retention.weights.protect_permutation", 0.05)) * protect_perm
    )
    max_ratio = float(get_by_path(cfg, "retention.max_aux_to_base_loss_ratio", 0.20))
    max_aux = base_loss.detach().abs() * max_ratio
    cap_factor = torch.minimum(uncapped.new_ones(()), max_aux / uncapped.detach().clamp_min(1e-12))
    loss = uncapped * cap_factor
    counts = dict(zip(names, (n_rs, n_rp, n_rpi, n_ps, n_pp, n_ppi), strict=True))
    return ComponentSafeLossOutput(
        loss, uncapped, scale,
        rescue_stage, rescue_pair, rescue_perm,
        protect_stage, protect_pair, protect_perm,
        counts,
    )


def component_safe_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": str(get_by_path(cfg, "retention.mode")),
        "teacher_cache": str(get_by_path(cfg, "retention.teacher_cache")),
        "reference_cache": str(get_by_path(cfg, "retention.reference_cache")),
        "soft_kl": bool(get_by_path(cfg, "retention.soft_kl", False)),
        "target_policy": "ground_truth_only_teacher_masks_and_weights",
        "teacher_confidence_weight": "detached_ground_truth_probability",
        "weights": dict(get_by_path(cfg, "retention.weights")),
        "coverage_policy": dict(get_by_path(cfg, "retention.coverage_policy", {})),
        "thresholds": dict(get_by_path(cfg, "retention.thresholds")),
        "max_aux_to_base_loss_ratio": float(get_by_path(cfg, "retention.max_aux_to_base_loss_ratio")),
        "schedule": {"start": 0.10, "ramp_end": 0.15, "plateau_end": 0.40, "decay_end": 0.70},
    }
