from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from snu_order.qwen3vl.canonical_stage_pair_evaluation import canonical_cpu_float32_scores, stable_ranking
from snu_order.qwen3vl.champion_retention import CACHE_FORMAT, file_sha256
from snu_order.qwen3vl.permutations import PAIRS
from snu_order.qwen3vl.stage_pair_scorer import pair_targets_from_answer, stage_targets_from_answer
from snu_order.utils.io import write_json


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("cache_format") != CACHE_FORMAT:
        raise RuntimeError(f"Unsupported component cache: {path}")
    return payload


def _gt_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    values = logits.float()
    target_values = targets.long()
    gt = values.gather(-1, target_values.unsqueeze(-1)).squeeze(-1)
    masked = values.scatter(-1, target_values.unsqueeze(-1), float("-inf"))
    return gt - masked.max(dim=-1).values


def _quantile(values: torch.Tensor, q: float) -> float:
    if not values.numel():
        raise RuntimeError("Cannot derive a threshold from an empty component set")
    return float(torch.quantile(values.float(), torch.tensor(float(q))).item())


def analyze(teacher_path: Path, reference_path: Path) -> dict[str, Any]:
    teacher = _load(teacher_path)
    reference = _load(reference_path)
    if teacher["ids"] != reference["ids"]:
        raise RuntimeError("Teacher and v1 reference cache IDs differ")
    if not torch.equal(teacher["answer"].long(), reference["answer"].long()):
        raise RuntimeError("Teacher and v1 reference cache Answers differ")
    if not torch.equal(teacher["target_perm_idx"].long(), reference["target_perm_idx"].long()):
        raise RuntimeError("Teacher and v1 reference target mapping differs")

    answer = teacher["answer"].long()
    target = teacher["target_perm_idx"].long()
    stage_target = stage_targets_from_answer(answer)
    pair_target = pair_targets_from_answer(answer)
    pair_sign = pair_target.mul(2).sub(1)
    teacher_stage = teacher["stage_logits"].float()
    reference_stage = reference["stage_logits"].float()
    teacher_pair = teacher["pair_logits"].float()
    reference_pair = reference["pair_logits"].float()
    teacher_final = canonical_cpu_float32_scores(teacher_stage, teacher_pair)
    reference_final = canonical_cpu_float32_scores(reference_stage, reference_pair)
    teacher_rank = stable_ranking(teacher_final, target)
    reference_rank = stable_ranking(reference_final, target)

    teacher_stage_correct = teacher_stage.argmax(-1).eq(stage_target)
    reference_stage_correct = reference_stage.argmax(-1).eq(stage_target)
    teacher_pair_margin = teacher_pair * pair_sign
    reference_pair_margin = reference_pair * pair_sign
    teacher_pair_correct = teacher_pair_margin.gt(0)
    reference_pair_correct = reference_pair_margin.gt(0)
    teacher_perm_correct = teacher_rank.prediction.eq(target)
    reference_perm_correct = reference_rank.prediction.eq(target)
    teacher_stage_margin = _gt_margin(teacher_stage, stage_target)
    reference_stage_margin = _gt_margin(reference_stage, stage_target)
    teacher_perm_margin = _gt_margin(teacher_final, target)
    reference_perm_margin = _gt_margin(reference_final, target)

    thresholds = {
        "teacher_permutation_margin_min": _quantile(teacher_perm_margin[teacher_perm_correct], 0.30),
        "reference_stage_q30": _quantile(reference_stage_margin[reference_stage_correct], 0.30),
        "reference_stage_q50": _quantile(reference_stage_margin[reference_stage_correct], 0.50),
        "reference_pair_q30": _quantile(reference_pair_margin[reference_pair_correct], 0.30),
        "reference_pair_q50": _quantile(reference_pair_margin[reference_pair_correct], 0.50),
        "reference_permutation_q30": _quantile(reference_perm_margin[reference_perm_correct], 0.30),
        "reference_permutation_q50": _quantile(reference_perm_margin[reference_perm_correct], 0.50),
    }
    rescue_stage = teacher_stage_correct & (
        ~reference_stage_correct | reference_stage_margin.lt(thresholds["reference_stage_q30"])
    )
    rescue_pair = teacher_pair_correct & (
        ~reference_pair_correct | reference_pair_margin.lt(thresholds["reference_pair_q30"])
    )
    rescue_perm = teacher_perm_correct & teacher_perm_margin.ge(thresholds["teacher_permutation_margin_min"]) & (
        ~reference_perm_correct | reference_perm_margin.lt(thresholds["reference_permutation_q30"])
    )
    pair_gap = torch.stack(
        [answer[:, left].sub(answer[:, right]).abs() for left, right in PAIRS], dim=1
    )
    sample_multi = rescue_stage.sum(1) + rescue_pair.sum(1) + rescue_perm.long()
    coverage_floor = 0.03
    rescue_coverage = {
        "stage": float(rescue_stage.sum().item()) / float(answer.shape[0] * 4),
        "pair": float(rescue_pair.sum().item()) / float(answer.shape[0] * len(PAIRS)),
        "permutation": float(rescue_perm.sum().item()) / float(answer.shape[0]),
    }
    eligible_rescue_components = {
        name: coverage >= coverage_floor for name, coverage in rescue_coverage.items()
    }
    status = (
        "COMPONENT_CACHE_AUDIT_PASS"
        if any(eligible_rescue_components.values())
        else "COMPONENT_CACHE_AUDIT_NO_ELIGIBLE_RESCUE"
    )
    return {
        "status": status,
        "sample_count": len(teacher["ids"]),
        "teacher_cache": str(teacher_path.resolve()),
        "teacher_cache_sha256": file_sha256(teacher_path),
        "reference_cache": str(reference_path.resolve()),
        "reference_cache_sha256": file_sha256(reference_path),
        "teacher_identity": teacher["identity"],
        "reference_identity": reference["identity"],
        "thresholds": thresholds,
        "teacher_oof": bool(teacher["identity"].get("teacher_is_oof", False)),
        "coverage_policy": {
            "minimum_component_rescue_fraction": coverage_floor,
            "rescue_coverage": rescue_coverage,
            "eligible_rescue_components": eligible_rescue_components,
            "policy": "disable_only_the_rescue_weight_for_an_ineligible_component",
        },
        "stage": {
            "teacher_correct": int(teacher_stage_correct.sum()),
            "reference_correct": int(reference_stage_correct.sum()),
            "teacher_correct_reference_wrong": int((teacher_stage_correct & ~reference_stage_correct).sum()),
            "reference_correct_teacher_wrong": int((reference_stage_correct & ~teacher_stage_correct).sum()),
            "rescue_components": int(rescue_stage.sum()),
            "rescue_by_target_position": {
                str(position + 1): int(rescue_stage[stage_target.eq(position)].sum()) for position in range(4)
            },
        },
        "pair": {
            "teacher_correct": int(teacher_pair_correct.sum()),
            "reference_correct": int(reference_pair_correct.sum()),
            "teacher_correct_reference_wrong": int((teacher_pair_correct & ~reference_pair_correct).sum()),
            "reference_correct_teacher_wrong": int((reference_pair_correct & ~teacher_pair_correct).sum()),
            "rescue_components": int(rescue_pair.sum()),
            "rescue_by_temporal_gap": {
                str(gap): int(rescue_pair[pair_gap.eq(gap)].sum()) for gap in (1, 2, 3)
            },
        },
        "permutation": {
            "teacher_correct": int(teacher_perm_correct.sum()),
            "reference_correct": int(reference_perm_correct.sum()),
            "teacher_correct_reference_wrong": int((teacher_perm_correct & ~reference_perm_correct).sum()),
            "reference_correct_teacher_wrong": int((reference_perm_correct & ~teacher_perm_correct).sum()),
            "rescue_samples": int(rescue_perm.sum()),
        },
        "multi_component_rescue_sample_count": int(sample_multi.gt(1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--reference-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(Path(args.teacher_cache), Path(args.reference_cache))
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
