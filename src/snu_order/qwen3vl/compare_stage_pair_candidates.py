from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_csv_rows, write_json

from .calibration_stage_pair import load_raw_stage_pair_logits, mcnemar_exact_p_value
from .stage_pair_scorer import structured_permutation_logits


REFERENCE_COUNTS = {
    "state_raw": 720,
    "state_full_fit": 731,
    "state_oof": 724,
    "four_token_raw": 722,
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Id", "true_perm_idx", "pred_perm_idx", "correct"}
    if not rows or set(rows[0]) != {
        "Id",
        "true_answer",
        "pred_answer",
        "true_perm_idx",
        "pred_perm_idx",
        "gt_rank",
        "top1_margin",
        "correct",
    }:
        raise RuntimeError(f"OOF prediction schema mismatch: {path}")
    if not required.issubset(rows[0]):
        raise RuntimeError(f"OOF prediction fields are missing: {path}")
    return rows


def _verification_passes(paths: list[Path], *, minimum_count: int) -> bool:
    if len(paths) < minimum_count:
        return False
    for path in paths:
        payload = _read_json(path)
        if str(payload.get("status", "")).lower() not in {"pass", "ok"}:
            return False
    return True


def _correct_count(payload: dict[str, Any]) -> int:
    return int(payload["correct_count"])


def _raw_fused_scores(payload: dict[str, Any]) -> torch.Tensor:
    existing = payload.get("raw_fused_scores")
    if torch.is_tensor(existing):
        return existing.detach().float().cpu()
    return structured_permutation_logits(
        payload["stage_logits"].detach().float().cpu(),
        payload["pair_logits"].detach().float().cpu(),
        stage_weight=1.0,
        pair_weight=0.3,
    )


def compare_candidates(
    *,
    state_raw_path: str | Path,
    state_calibration_dir: str | Path,
    four_token_raw_path: str | Path,
    four_token_calibration_dir: str | Path,
    state_verifications: list[str | Path],
    four_token_verifications: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    state_raw = load_raw_stage_pair_logits(state_raw_path)
    four_raw = load_raw_stage_pair_logits(four_token_raw_path)
    if state_raw["ids"] != four_raw["ids"]:
        raise RuntimeError("STATE and 4-token raw artifacts have different sample IDs or order")
    if not torch.equal(state_raw["target_perm_idx"], four_raw["target_perm_idx"]):
        raise RuntimeError("STATE and 4-token raw artifacts have different validation targets")

    state_dir = Path(state_calibration_dir)
    four_dir = Path(four_token_calibration_dir)
    state_calibration = _read_json(state_dir / "calibration.json")
    four_calibration = _read_json(four_dir / "calibration.json")
    if state_calibration.get("calibration_grid_sha256") != four_calibration.get("calibration_grid_sha256"):
        raise RuntimeError("STATE and 4-token calibration grids differ")
    if state_calibration.get("fold_assignment_sha256") != four_calibration.get("fold_assignment_sha256"):
        raise RuntimeError("STATE and 4-token fold assignments differ")
    if state_calibration.get("tie_break_rule") != four_calibration.get("tie_break_rule"):
        raise RuntimeError("STATE and 4-token calibration tie-break rules differ")

    state_full = _read_json(state_dir / "calibrated_metrics.json")
    state_oof = _read_json(state_dir / "oof_metrics.json")
    state_comparison = _read_json(state_dir / "raw_vs_calibrated_comparison.json")
    state_folds = _read_json(state_dir / "fold_calibration.json")["folds"]
    four_full = _read_json(four_dir / "calibrated_metrics.json")
    four_oof = _read_json(four_dir / "oof_metrics.json")
    four_comparison = _read_json(four_dir / "raw_vs_calibrated_comparison.json")
    four_fixed = _read_json(four_dir / "fixed_calibration_diagnostic.json")
    four_folds = _read_json(four_dir / "fold_calibration.json")["folds"]

    observed_counts = {
        "state_raw": _correct_count(state_calibration["raw_metrics"]),
        "state_full_fit": _correct_count(state_full),
        "state_oof": _correct_count(state_oof),
        "four_token_raw": _correct_count(four_calibration["raw_metrics"]),
    }
    if observed_counts != REFERENCE_COUNTS:
        raise RuntimeError(
            f"Reference calibration counts were not reproduced: expected={REFERENCE_COUNTS}, "
            f"observed={observed_counts}"
        )

    state_rows = _read_prediction_rows(state_dir / "oof_valid_predictions.csv")
    four_rows = _read_prediction_rows(four_dir / "oof_valid_predictions.csv")
    if [row["Id"] for row in state_rows] != state_raw["ids"]:
        raise RuntimeError("STATE OOF prediction IDs do not match the raw artifact")
    if [row["Id"] for row in four_rows] != state_raw["ids"]:
        raise RuntimeError("4-token OOF prediction IDs do not match the raw artifact")

    state_raw_predictions = _raw_fused_scores(state_raw).argmax(dim=1).tolist()
    four_raw_predictions = _raw_fused_scores(four_raw).argmax(dim=1).tolist()
    paired_rows: list[dict[str, Any]] = []
    state_oof_correct: list[bool] = []
    four_oof_correct: list[bool] = []
    for index, (state_row, four_row) in enumerate(zip(state_rows, four_rows, strict=True)):
        target = int(state_raw["target_perm_idx"][index].item())
        state_correct = bool(int(state_row["correct"]))
        four_correct = bool(int(four_row["correct"]))
        state_oof_correct.append(state_correct)
        four_oof_correct.append(four_correct)
        paired_rows.append(
            {
                "Id": state_row["Id"],
                "target": target,
                "state_raw_prediction": int(state_raw_predictions[index]),
                "state_calibrated_oof_prediction": int(state_row["pred_perm_idx"]),
                "four_token_raw_prediction": int(four_raw_predictions[index]),
                "four_token_calibrated_oof_prediction": int(four_row["pred_perm_idx"]),
                "state_oof_correct": int(state_correct),
                "four_token_oof_correct": int(four_correct),
            }
        )

    fixed = sum(
        int(not state_correct and four_correct)
        for state_correct, four_correct in zip(state_oof_correct, four_oof_correct, strict=True)
    )
    broken = sum(
        int(state_correct and not four_correct)
        for state_correct, four_correct in zip(state_oof_correct, four_oof_correct, strict=True)
    )
    fold_rows = []
    fold_wins = 0
    for state_fold, four_fold in zip(state_folds, four_folds, strict=True):
        if int(state_fold["fold"]) != int(four_fold["fold"]):
            raise RuntimeError("STATE and 4-token fold result ordering differs")
        state_correct = int(state_fold["correct_count"])
        four_correct = int(four_fold["correct_count"])
        fold_wins += int(four_correct > state_correct)
        fold_rows.append(
            {
                "fold": int(state_fold["fold"]),
                "state_correct": state_correct,
                "four_token_correct": four_correct,
                "delta": four_correct - state_correct,
                "four_token_pair_weight": float(four_fold["pair_weight"]),
                "four_token_stage_temperature": float(four_fold["stage_temperature"]),
                "four_token_pair_temperature": float(four_fold["pair_temperature"]),
            }
        )

    state_verified = _verification_passes([Path(path) for path in state_verifications], minimum_count=2)
    four_verified = _verification_passes(
        [Path(path) for path in four_token_verifications], minimum_count=2
    )
    criteria = {
        "four_token_raw_722_reproduced": observed_counts["four_token_raw"] == 722,
        "state_reference_counts_reproduced": all(
            observed_counts[key] == REFERENCE_COUNTS[key]
            for key in ("state_raw", "state_full_fit", "state_oof")
        ),
        "four_token_oof_at_least_five_higher": (
            _correct_count(four_oof) >= _correct_count(state_oof) + 5
        ),
        "four_token_wins_at_least_four_folds": fold_wins >= 4,
        "four_token_full_fit_not_lower": _correct_count(four_full) >= _correct_count(state_full),
        "four_token_top3_not_worse_by_half_point": (
            float(four_oof["top3_accuracy"]) >= float(state_oof["top3_accuracy"]) - 0.005
        ),
        "four_token_has_calibration_plateau": bool(
            four_comparison.get("plateau", {}).get("has_plateau", False)
        ),
        "fresh_process_fingerprint_verification_passed": state_verified and four_verified,
    }
    select_four = all(criteria.values())
    decision = {
        "status": "PASS",
        "selected_candidate": "four_token_epoch3" if select_four else "state_e1",
        "reason": (
            "4-token satisfied every predeclared promotion criterion"
            if select_four
            else "4-token did not satisfy every predeclared promotion criterion; retain STATE E1"
        ),
        "criteria": criteria,
        "observed_counts": observed_counts,
        "four_token_full_fit_correct": _correct_count(four_full),
        "four_token_oof_correct": _correct_count(four_oof),
        "state_full_fit_correct": _correct_count(state_full),
        "state_oof_correct": _correct_count(state_oof),
        "four_token_fold_win_count": fold_wins,
        "paired_state_fixed_by_four_token": fixed,
        "paired_state_broken_by_four_token": broken,
        "paired_mcnemar_exact_p_value": mcnemar_exact_p_value(broken, fixed),
    }
    comparison = {
        "status": "PASS",
        "state": {
            "raw": state_calibration["raw_metrics"],
            "full_fit": state_full,
            "oof": state_oof,
            "selected_parameters": {
                key: state_calibration[key]
                for key in ("pair_weight", "stage_temperature", "pair_temperature")
            },
            "plateau": state_comparison.get("plateau"),
        },
        "four_token": {
            "raw": four_calibration["raw_metrics"],
            "fixed_state_calibration": four_fixed,
            "full_fit": four_full,
            "oof": four_oof,
            "selected_parameters": {
                key: four_calibration[key]
                for key in ("pair_weight", "stage_temperature", "pair_temperature")
            },
            "plateau": four_comparison.get("plateau"),
        },
        "folds": fold_rows,
        "decision": decision,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output / "paired_predictions.csv", paired_rows, list(paired_rows[0]))
    metric_rows = [
        {
            "candidate": name,
            "raw_correct": _correct_count(calibration["raw_metrics"]),
            "full_fit_correct": _correct_count(full),
            "oof_correct": _correct_count(oof),
            "raw_mrr": calibration["raw_metrics"]["MRR"],
            "full_fit_mrr": full["MRR"],
            "oof_mrr": oof["MRR"],
            "raw_top3": calibration["raw_metrics"]["top3_accuracy"],
            "full_fit_top3": full["top3_accuracy"],
            "oof_top3": oof["top3_accuracy"],
        }
        for name, calibration, full, oof in (
            ("state_e1", state_calibration, state_full, state_oof),
            ("four_token_epoch3", four_calibration, four_full, four_oof),
        )
    ]
    write_csv_rows(output / "comparison_state_vs_4token.csv", metric_rows, list(metric_rows[0]))
    write_json(output / "calibration_comparison.json", comparison)
    write_json(output / "decision.json", decision)
    report = [
        "# STATE E1 vs 4-token Epoch 3",
        "",
        f"- STATE raw/full-fit/OOF: {observed_counts['state_raw']}/{observed_counts['state_full_fit']}/{observed_counts['state_oof']}",
        f"- 4-token raw/full-fit/OOF: {observed_counts['four_token_raw']}/{_correct_count(four_full)}/{_correct_count(four_oof)}",
        f"- 4-token fold wins: {fold_wins}/5",
        f"- Selected candidate: {decision['selected_candidate']}",
        "",
        "The decision applies the predeclared OOF-first promotion rules without identity bias or valid-B.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-raw", required=True)
    parser.add_argument("--state-calibration-dir", required=True)
    parser.add_argument("--four-token-raw", required=True)
    parser.add_argument("--four-token-calibration-dir", required=True)
    parser.add_argument("--state-verification", action="append", default=[])
    parser.add_argument("--four-token-verification", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = compare_candidates(
        state_raw_path=args.state_raw,
        state_calibration_dir=args.state_calibration_dir,
        four_token_raw_path=args.four_token_raw,
        four_token_calibration_dir=args.four_token_calibration_dir,
        state_verifications=args.state_verification,
        four_token_verifications=args.four_token_verification,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
