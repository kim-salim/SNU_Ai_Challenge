from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_json

from .calibration_stage_pair import load_raw_stage_pair_logits
from .canonical_stage_pair_evaluation import canonical_cpu_float32_scores, stable_ranking
from .metrics_stage_pair import compute_stage_pair_metrics


def _mcnemar_exact(fixed: int, broken: int) -> float:
    discordant = int(fixed) + int(broken)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(fixed, broken) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _fold(sample_id: str) -> int:
    return int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:8], 16) % 5


def _bootstrap_delta(base_correct: torch.Tensor, candidate_correct: torch.Tensor, repeats: int = 10_000) -> list[int]:
    rng = random.Random(20260720)
    delta = candidate_correct.to(torch.int64) - base_correct.to(torch.int64)
    n = int(delta.numel())
    values = []
    for _ in range(int(repeats)):
        values.append(sum(int(delta[rng.randrange(n)]) for _ in range(n)))
    values.sort()
    return [values[int(0.025 * repeats)], values[min(repeats - 1, int(0.975 * repeats))]]


def compare(
    control_path: str | Path,
    candidate_path: str | Path,
    *,
    control_calibration_path: str | Path | None = None,
    candidate_calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    control = load_raw_stage_pair_logits(control_path)
    candidate = load_raw_stage_pair_logits(candidate_path)
    if control["ids"] != candidate["ids"]:
        raise RuntimeError("C0 and K0 ID ordering differs")
    for key in ("target_perm_idx", "answer"):
        if not torch.equal(control[key].long(), candidate[key].long()):
            raise RuntimeError(f"C0 and K0 {key} differs")
    target = control["target_perm_idx"].long().cpu()
    control_scores = canonical_cpu_float32_scores(control["stage_logits"], control["pair_logits"])
    candidate_scores = canonical_cpu_float32_scores(candidate["stage_logits"], candidate["pair_logits"])
    control_rank = stable_ranking(control_scores, target)
    candidate_rank = stable_ranking(candidate_scores, target)
    control_correct = control_rank.prediction.eq(target)
    candidate_correct = candidate_rank.prediction.eq(target)
    fixed_mask = ~control_correct & candidate_correct
    broken_mask = control_correct & ~candidate_correct
    fixed = int(fixed_mask.sum())
    broken = int(broken_mask.sum())
    control_metrics = compute_stage_pair_metrics(
        control_scores, target,
        stage_logits=control["stage_logits"], pair_logits=control["pair_logits"], answer=control["answer"],
    )
    candidate_metrics = compute_stage_pair_metrics(
        candidate_scores, target,
        stage_logits=candidate["stage_logits"], pair_logits=candidate["pair_logits"], answer=candidate["answer"],
    )
    folds = []
    fold_deltas = []
    for fold_index in range(5):
        mask = torch.tensor([_fold(sample_id) == fold_index for sample_id in control["ids"]])
        base_count = int(control_correct[mask].sum())
        candidate_count = int(candidate_correct[mask].sum())
        delta = candidate_count - base_count
        fold_deltas.append(delta)
        folds.append({"fold": fold_index, "count": int(mask.sum()), "control_correct": base_count, "candidate_correct": candidate_count, "delta": delta})
    correct_delta = int(candidate_metrics["correct_count"]) - int(control_metrics["correct_count"])
    top3_sample_delta = round(
        (float(candidate_metrics["top3_accuracy"]) - float(control_metrics["top3_accuracy"])) * len(control["ids"])
    )
    stage3_delta = (
        float(candidate_metrics["stage_accuracy_by_position"]["3"])
        - float(control_metrics["stage_accuracy_by_position"]["3"])
    )
    ratios = {
        "stage_component_std": float(candidate_metrics["stage_component_score_std"]) / max(float(control_metrics["stage_component_score_std"]), 1e-12),
        "pair_component_std": float(candidate_metrics["pair_component_score_std"]) / max(float(control_metrics["pair_component_score_std"]), 1e-12),
        "top1_margin": float(candidate_metrics["mean_top1_top2_margin"]) / max(float(control_metrics["mean_top1_top2_margin"]), 1e-12),
    }
    screening_checks = {
        "raw_correct_delta_ge_5": correct_delta >= 5,
        "mrr_delta_ge_minus_0p0005": float(candidate_metrics["MRR"]) - float(control_metrics["MRR"]) >= -0.0005,
        "top3_sample_delta_nonnegative": top3_sample_delta >= 0,
        "fixed_minus_broken_ge_5": fixed - broken >= 5,
        "stage_only_not_worse": int(candidate_metrics["stage_only_correct_count"]) >= int(control_metrics["stage_only_correct_count"]),
        "stage3_not_worse": stage3_delta >= 0.0,
        "high_margin_wrong_not_worse": int(candidate_metrics["high_margin_wrong_count"]) <= int(control_metrics["high_margin_wrong_count"]),
        "stage_std_ratio_ge_0p95": ratios["stage_component_std"] >= 0.95,
        "pair_std_ratio_ge_0p95": ratios["pair_component_std"] >= 0.95,
        "margin_ratio_ge_0p95": ratios["top1_margin"] >= 0.95,
        "four_folds_non_degraded": sum(value >= 0 for value in fold_deltas) >= 4,
        "three_folds_improved": sum(value > 0 for value in fold_deltas) >= 3,
    }
    calibration_comparison = None
    if (control_calibration_path is None) != (candidate_calibration_path is None):
        raise RuntimeError("C0 and K0 calibration artifacts must be supplied together")
    if control_calibration_path is not None and candidate_calibration_path is not None:
        control_calibration = json.loads(Path(control_calibration_path).read_text(encoding="utf-8"))
        candidate_calibration = json.loads(Path(candidate_calibration_path).read_text(encoding="utf-8"))
        control_oof = control_calibration["oof_metrics"]
        candidate_oof = candidate_calibration["oof_metrics"]
        oof_correct_delta = int(candidate_oof["correct_count"]) - int(control_oof["correct_count"])
        calibration_comparison = {
            "control_oof": control_oof,
            "candidate_oof": candidate_oof,
            "oof_correct_delta": oof_correct_delta,
            "control_parameters": {
                key: control_calibration[key]
                for key in ("pair_weight", "stage_temperature", "pair_temperature")
            },
            "candidate_parameters": {
                key: candidate_calibration[key]
                for key in ("pair_weight", "stage_temperature", "pair_temperature")
            },
        }
        screening_checks["oof_calibrated_correct_delta_ge_5"] = oof_correct_delta >= 5
    status = "K0_SCREENING_GATE_PASS" if all(screening_checks.values()) else "K0_SCREENING_GATE_FAIL"
    return {
        "status": status,
        "comparison": "K0_component_safe_minus_C0_exact_contract",
        "control": control_metrics,
        "candidate": candidate_metrics,
        "delta": {
            "correct": correct_delta,
            "MRR": float(candidate_metrics["MRR"]) - float(control_metrics["MRR"]),
            "top3_samples": top3_sample_delta,
            "adjacent_swap": int(candidate_metrics["adjacent_swap_error_count"]) - int(control_metrics["adjacent_swap_error_count"]),
            "swap_3_4": int(candidate_metrics["adjacent_swap_by_stage"]["3-4"]) - int(control_metrics["adjacent_swap_by_stage"]["3-4"]),
            "high_margin_wrong": int(candidate_metrics["high_margin_wrong_count"]) - int(control_metrics["high_margin_wrong_count"]),
            "stage_only_correct": int(candidate_metrics["stage_only_correct_count"]) - int(control_metrics["stage_only_correct_count"]),
            "stage3_accuracy": stage3_delta,
        },
        "paired": {
            "fixed": fixed,
            "broken": broken,
            "net": fixed - broken,
            "mcnemar_exact_p": _mcnemar_exact(fixed, broken),
            "bootstrap_95_ci_correct_delta": _bootstrap_delta(control_correct, candidate_correct),
        },
        "score_separation_ratios": ratios,
        "calibration_comparison": calibration_comparison,
        "folds": folds,
        "fold_assignment_method": "sha256(sample_id) modulo 5; fixed before candidate evaluation",
        "screening_checks": screening_checks,
        "next_action": (
            "RUN_ONE_TRUE_RETENTION_OR_STAGE34_FOLLOWUP"
            if status == "K0_SCREENING_GATE_PASS"
            else "RETAIN_C0_AND_DO_NOT_RUN_AUTOMATIC_FOLLOWUP"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-raw", required=True)
    parser.add_argument("--candidate-raw", required=True)
    parser.add_argument("--control-calibration", default=None)
    parser.add_argument("--candidate-calibration", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(
        args.control_raw,
        args.candidate_raw,
        control_calibration_path=args.control_calibration,
        candidate_calibration_path=args.candidate_calibration,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
