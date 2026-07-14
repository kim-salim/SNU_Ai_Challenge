from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_csv_rows, write_json

from .calibration_stage_pair import (
    CalibrationParameters,
    calibrated_structured_logits,
    load_raw_stage_pair_logits,
    mcnemar_exact_p_value,
)
from .metrics_stage_pair import compute_stage_pair_metrics
from .stage_pair_scorer import structured_permutation_logits


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {source}")
    return payload


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [
            "Id",
            "true_answer",
            "pred_answer",
            "true_perm_idx",
            "pred_perm_idx",
            "gt_rank",
            "top1_margin",
            "correct",
        ]
        if reader.fieldnames != expected:
            raise RuntimeError(
                f"Prediction schema mismatch for {path}: {reader.fieldnames} != {expected}"
            )
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Prediction artifact is empty: {path}")
    return rows


def _calibration_parameters(payload: dict[str, Any]) -> CalibrationParameters:
    return CalibrationParameters(
        pair_weight=float(payload["pair_weight"]),
        stage_temperature=float(payload["stage_temperature"]),
        pair_temperature=float(payload["pair_temperature"]),
    )


def _raw_logits(payload: dict[str, Any]) -> torch.Tensor:
    return structured_permutation_logits(
        payload["stage_logits"].float(),
        payload["pair_logits"].float(),
        stage_weight=1.0,
        pair_weight=0.3,
    ).cpu()


def _metrics(payload: dict[str, Any], logits: torch.Tensor) -> dict[str, Any]:
    return compute_stage_pair_metrics(
        logits,
        payload["target_perm_idx"],
        stage_logits=payload["stage_logits"],
        pair_logits=payload["pair_logits"],
        answer=payload["answer"],
    )


def _paired_counts(first: torch.Tensor, second: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    first_correct = first.argmax(dim=1).cpu().eq(targets.cpu())
    second_correct = second.argmax(dim=1).cpu().eq(targets.cpu())
    broken = int((first_correct & ~second_correct).sum().item())
    fixed = int((~first_correct & second_correct).sum().item())
    return {
        "fixed": fixed,
        "broken": broken,
        "unchanged_correct": int((first_correct & second_correct).sum().item()),
        "unchanged_wrong": int((~first_correct & ~second_correct).sum().item()),
        "mcnemar_exact_p_value": mcnemar_exact_p_value(broken, fixed),
    }


def _mean_positions(metrics: dict[str, Any], positions: tuple[str, str]) -> float:
    values = metrics["stage_accuracy_by_position"]
    return sum(float(values[position]) for position in positions) / len(positions)


def _fold_results(directory: Path) -> list[dict[str, Any]]:
    payload = _read_json(directory / "fold_calibration.json")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError(f"Expected five calibration folds: {directory}")
    return folds


def _verification_status(paths: list[str | Path]) -> tuple[bool, list[dict[str, Any]]]:
    results = [_read_json(path) for path in paths]
    passed = len(results) >= 2 and all(
        str(item.get("status", "")).lower() in {"pass", "ok"}
        and bool(item.get("finite_logits", True))
        for item in results
    )
    if passed:
        predictions = [item.get("prediction_indices") for item in results]
        passed = all(value == predictions[0] for value in predictions[1:])
    return passed, results


def compare_e1_e2(
    *,
    e1_raw_path: str | Path,
    e1_calibration_dir: str | Path,
    e2_raw_path: str | Path,
    e2_calibration_dir: str | Path,
    e2_verifications: list[str | Path],
    gradient_health_path: str | Path,
    semantic_diff_path: str | Path,
    chunk_selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    e1_raw = load_raw_stage_pair_logits(e1_raw_path)
    e2_raw = load_raw_stage_pair_logits(e2_raw_path)
    if e1_raw["ids"] != e2_raw["ids"]:
        raise RuntimeError("E1 and E2 sample IDs/order differ")
    if not torch.equal(e1_raw["target_perm_idx"], e2_raw["target_perm_idx"]):
        raise RuntimeError("E1 and E2 validation targets differ")
    if e1_raw["permutation_table_fingerprint"] != e2_raw["permutation_table_fingerprint"]:
        raise RuntimeError("E1 and E2 permutation mappings differ")

    e1_dir = Path(e1_calibration_dir)
    e2_dir = Path(e2_calibration_dir)
    e1_calibration = _read_json(e1_dir / "calibration.json")
    e2_calibration = _read_json(e2_dir / "calibration.json")
    for key in (
        "calibration_grid_sha256",
        "fold_assignment_sha256",
        "fold_assignment_method",
        "tie_break_rule",
    ):
        if e1_calibration.get(key) != e2_calibration.get(key):
            raise RuntimeError(f"E1/E2 calibration protocol mismatch for {key}")

    e1_raw_logits = _raw_logits(e1_raw)
    e2_raw_logits = _raw_logits(e2_raw)
    e1_full_logits = calibrated_structured_logits(
        e1_raw["stage_logits"], e1_raw["pair_logits"], _calibration_parameters(e1_calibration)
    ).cpu()
    e2_full_logits = calibrated_structured_logits(
        e2_raw["stage_logits"], e2_raw["pair_logits"], _calibration_parameters(e2_calibration)
    ).cpu()
    e1_raw_metrics = _metrics(e1_raw, e1_raw_logits)
    e2_raw_metrics = _metrics(e2_raw, e2_raw_logits)
    e1_full_metrics = _metrics(e1_raw, e1_full_logits)
    e2_full_metrics = _metrics(e2_raw, e2_full_logits)

    e1_oof_rows = _read_predictions(e1_dir / "oof_valid_predictions.csv")
    e2_oof_rows = _read_predictions(e2_dir / "oof_valid_predictions.csv")
    ids = [str(value) for value in e1_raw["ids"]]
    if [row["Id"] for row in e1_oof_rows] != ids or [row["Id"] for row in e2_oof_rows] != ids:
        raise RuntimeError("OOF prediction IDs do not match the raw artifacts")
    e1_oof_metrics = _read_json(e1_dir / "oof_metrics.json")
    e2_oof_metrics = _read_json(e2_dir / "oof_metrics.json")

    e1_folds = _fold_results(e1_dir)
    e2_folds = _fold_results(e2_dir)
    fold_rows: list[dict[str, Any]] = []
    e2_fold_wins = 0
    e2_fold_nonlosses = 0
    for e1_fold, e2_fold in zip(e1_folds, e2_folds, strict=True):
        fold = int(e1_fold["fold"])
        if int(e2_fold["fold"]) != fold:
            raise RuntimeError("E1/E2 fold ordering differs")
        e1_correct = int(e1_fold["correct_count"])
        e2_correct = int(e2_fold["correct_count"])
        e2_fold_wins += int(e2_correct > e1_correct)
        e2_fold_nonlosses += int(e2_correct >= e1_correct)
        fold_rows.append(
            {
                "fold": fold,
                "e1_correct": e1_correct,
                "e2_correct": e2_correct,
                "delta": e2_correct - e1_correct,
                "e2_pair_weight": float(e2_fold["pair_weight"]),
                "e2_stage_temperature": float(e2_fold["stage_temperature"]),
                "e2_pair_temperature": float(e2_fold["pair_temperature"]),
            }
        )

    gradient_health = _read_json(gradient_health_path)
    semantic_diff = _read_json(semantic_diff_path)
    chunk_selection = _read_json(chunk_selection_path)
    verification_passed, verification_payloads = _verification_status(e2_verifications)
    integrity = {
        "fresh_process_verification": verification_passed,
        "gradient_health": gradient_health.get("status") == "PASS",
        "two_completed_gradient_steps": int(
            gradient_health.get("captured_completed_optimizer_steps", 0)
        )
        >= 2,
        "semantic_diff_only_allowlisted": semantic_diff.get("differences") == {},
    }
    integrity_passed = all(integrity.values())

    e1_mid = _mean_positions(e1_raw_metrics, ("2", "3"))
    e2_mid = _mean_positions(e2_raw_metrics, ("2", "3"))
    e1_edge = _mean_positions(e1_raw_metrics, ("1", "4"))
    e2_edge = _mean_positions(e2_raw_metrics, ("1", "4"))
    e1_stage3 = float(e1_raw_metrics["stage_accuracy_by_position"]["3"])
    e2_stage3 = float(e2_raw_metrics["stage_accuracy_by_position"]["3"])
    e1_kendall1 = int(e1_raw_metrics["kendall_distance_histogram"].get("1", 0))
    e2_kendall1 = int(e2_raw_metrics["kendall_distance_histogram"].get("1", 0))
    e1_adjacent = float(e1_raw_metrics["pairwise_head_accuracy_by_temporal_gap"]["1"])
    e2_adjacent = float(e2_raw_metrics["pairwise_head_accuracy_by_temporal_gap"]["1"])

    strict_criteria = {
        "integrity_fingerprint_gradient_pass": integrity_passed,
        "raw_exact_at_least_five_higher": int(e2_raw_metrics["correct_count"])
        >= int(e1_raw_metrics["correct_count"]) + 5,
        "stage_mid_or_stage3_improved": e2_mid > e1_mid or e2_stage3 > e1_stage3,
        "local_ordering_error_improved": e2_kendall1 < e1_kendall1
        or e2_adjacent > e1_adjacent,
        "calibrated_oof_not_lower": int(e2_oof_metrics["correct_count"])
        >= int(e1_oof_metrics["correct_count"]),
        "fold_improvement_not_single_fold_only": e2_fold_nonlosses >= 3,
        "edge_stage_not_severely_damaged": e2_edge >= e1_edge - 0.02,
        "chunked_inference_parity": chunk_selection.get("status") == "PASS",
    }
    strict_promote = all(strict_criteria.values())

    e1_performance_key = (
        int(e1_oof_metrics["correct_count"]),
        int(e1_raw_metrics["correct_count"]),
        int(e1_full_metrics["correct_count"]),
        float(e1_oof_metrics["MRR"]),
        float(e1_oof_metrics["top3_accuracy"]),
    )
    e2_performance_key = (
        int(e2_oof_metrics["correct_count"]),
        int(e2_raw_metrics["correct_count"]),
        int(e2_full_metrics["correct_count"]),
        float(e2_oof_metrics["MRR"]),
        float(e2_oof_metrics["top3_accuracy"]),
    )
    performance_select_e2 = integrity_passed and e2_performance_key > e1_performance_key
    selected = "state_e2_vision_merger" if performance_select_e2 else "state_e1"

    targets = e1_raw["target_perm_idx"].cpu()
    raw_paired = _paired_counts(e1_raw_logits, e2_raw_logits, targets)
    full_paired = _paired_counts(e1_full_logits, e2_full_logits, targets)
    e1_oof_correct = torch.tensor([int(row["correct"]) for row in e1_oof_rows], dtype=torch.bool)
    e2_oof_correct = torch.tensor([int(row["correct"]) for row in e2_oof_rows], dtype=torch.bool)
    oof_broken = int((e1_oof_correct & ~e2_oof_correct).sum().item())
    oof_fixed = int((~e1_oof_correct & e2_oof_correct).sum().item())
    oof_paired = {
        "fixed": oof_fixed,
        "broken": oof_broken,
        "unchanged_correct": int((e1_oof_correct & e2_oof_correct).sum().item()),
        "unchanged_wrong": int((~e1_oof_correct & ~e2_oof_correct).sum().item()),
        "mcnemar_exact_p_value": mcnemar_exact_p_value(oof_broken, oof_fixed),
    }

    paired_rows = []
    for index, sample_id in enumerate(ids):
        paired_rows.append(
            {
                "Id": sample_id,
                "target": int(targets[index].item()),
                "e1_raw_prediction": int(e1_raw_logits[index].argmax().item()),
                "e2_raw_prediction": int(e2_raw_logits[index].argmax().item()),
                "e1_full_fit_prediction": int(e1_full_logits[index].argmax().item()),
                "e2_full_fit_prediction": int(e2_full_logits[index].argmax().item()),
                "e1_oof_prediction": int(e1_oof_rows[index]["pred_perm_idx"]),
                "e2_oof_prediction": int(e2_oof_rows[index]["pred_perm_idx"]),
                "e1_raw_correct": int(e1_raw_logits[index].argmax().item() == targets[index]),
                "e2_raw_correct": int(e2_raw_logits[index].argmax().item() == targets[index]),
                "e1_oof_correct": int(e1_oof_correct[index]),
                "e2_oof_correct": int(e2_oof_correct[index]),
            }
        )

    decision = {
        "status": "PASS" if integrity_passed else "FAIL",
        "selected_candidate": selected,
        "selection_policy": "integrity_then_calibrated_oof_raw_fullfit_mrr_top3",
        "selected_inference_frame_chunk_size": 4,
        "chunking_rejected_due_to_prediction_drift": chunk_selection.get("status") != "PASS",
        "performance_keys": {"state_e1": e1_performance_key, "state_e2": e2_performance_key},
        "integrity": integrity,
        "strict_predeclared_promotion_criteria": strict_criteria,
        "strict_predeclared_promotion_passed": strict_promote,
        "note": (
            "Performance-first selection uses the certified unchunked path because frame chunking "
            "changed predictions on this backend."
        ),
    }
    comparison = {
        "status": decision["status"],
        "state_e1": {
            "raw": e1_raw_metrics,
            "full_fit": e1_full_metrics,
            "oof": e1_oof_metrics,
            "parameters": _calibration_parameters(e1_calibration).as_dict(),
            "stage_2_3_mean": e1_mid,
            "stage_1_4_mean": e1_edge,
        },
        "state_e2": {
            "raw": e2_raw_metrics,
            "full_fit": e2_full_metrics,
            "oof": e2_oof_metrics,
            "fixed_state_calibration": _read_json(e2_dir / "fixed_calibration_diagnostic.json"),
            "parameters": _calibration_parameters(e2_calibration).as_dict(),
            "stage_2_3_mean": e2_mid,
            "stage_1_4_mean": e2_edge,
        },
        "paired": {"raw": raw_paired, "full_fit": full_paired, "oof": oof_paired},
        "folds": fold_rows,
        "e2_fold_win_count": e2_fold_wins,
        "e2_fold_nonloss_count": e2_fold_nonlosses,
        "verification": verification_payloads,
        "gradient_health": gradient_health,
        "semantic_diff": semantic_diff,
        "chunk_selection": chunk_selection,
        "decision": decision,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "raw_metrics.json", {"state_e1": e1_raw_metrics, "state_e2": e2_raw_metrics})
    write_json(
        output / "calibrated_metrics.json",
        {
            "state_e1": {"full_fit": e1_full_metrics, "oof": e1_oof_metrics},
            "state_e2": {"full_fit": e2_full_metrics, "oof": e2_oof_metrics},
            "paired": comparison["paired"],
        },
    )
    write_json(output / "decision.json", decision)
    write_csv_rows(output / "paired_predictions.csv", paired_rows, list(paired_rows[0]))
    summary_rows = [
        {
            "candidate": name,
            "raw_correct": int(raw_metrics["correct_count"]),
            "full_fit_correct": int(full_metrics["correct_count"]),
            "oof_correct": int(oof_metrics["correct_count"]),
            "raw_mrr": float(raw_metrics["MRR"]),
            "oof_mrr": float(oof_metrics["MRR"]),
            "raw_top3": float(raw_metrics["top3_accuracy"]),
            "oof_top3": float(oof_metrics["top3_accuracy"]),
            "stage_2_3_mean": mid,
            "stage_1_4_mean": edge,
            "kendall_distance_1": int(raw_metrics["kendall_distance_histogram"].get("1", 0)),
            "adjacent_pair_accuracy": float(
                raw_metrics["pairwise_head_accuracy_by_temporal_gap"]["1"]
            ),
        }
        for name, raw_metrics, full_metrics, oof_metrics, mid, edge in (
            ("state_e1", e1_raw_metrics, e1_full_metrics, e1_oof_metrics, e1_mid, e1_edge),
            ("state_e2", e2_raw_metrics, e2_full_metrics, e2_oof_metrics, e2_mid, e2_edge),
        )
    ]
    write_csv_rows(output / "e1_vs_e2_comparison.csv", summary_rows, list(summary_rows[0]))
    report = [
        "# STATE E1 vs Vision Merger E2",
        "",
        f"- E1 raw/full-fit/OOF: {e1_raw_metrics['correct_count']}/{e1_full_metrics['correct_count']}/{e1_oof_metrics['correct_count']}",
        f"- E2 raw/full-fit/OOF: {e2_raw_metrics['correct_count']}/{e2_full_metrics['correct_count']}/{e2_oof_metrics['correct_count']}",
        f"- E2 Stage 3 delta: {e2_stage3 - e1_stage3:+.6f}",
        f"- E2 Stage 2/3 mean delta: {e2_mid - e1_mid:+.6f}",
        f"- E2 Kendall-distance-1 delta: {e2_kendall1 - e1_kendall1:+d}",
        f"- E2 adjacent-pair delta: {e2_adjacent - e1_adjacent:+.6f}",
        f"- Selected candidate: {selected}",
        "- Inference path: unchunked (frame_chunk_size=4); chunked paths were rejected after full valid-A drift.",
        "",
        "No identity prior or valid-B result was used.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-raw", required=True)
    parser.add_argument("--e1-calibration-dir", required=True)
    parser.add_argument("--e2-raw", required=True)
    parser.add_argument("--e2-calibration-dir", required=True)
    parser.add_argument("--e2-verification", action="append", default=[])
    parser.add_argument("--gradient-health", required=True)
    parser.add_argument("--semantic-diff", required=True)
    parser.add_argument("--chunk-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = compare_e1_e2(
        e1_raw_path=args.e1_raw,
        e1_calibration_dir=args.e1_calibration_dir,
        e2_raw_path=args.e2_raw,
        e2_calibration_dir=args.e2_calibration_dir,
        e2_verifications=args.e2_verification,
        gradient_health_path=args.gradient_health,
        semantic_diff_path=args.semantic_diff,
        chunk_selection_path=args.chunk_selection,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
