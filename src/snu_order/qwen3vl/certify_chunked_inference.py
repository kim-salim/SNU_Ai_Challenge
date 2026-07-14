from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_json

from .calibration_stage_pair import calibrated_structured_logits, load_calibration, load_raw_stage_pair_logits


BF16_ABSOLUTE_TOLERANCE = 0.125
PERCENTILE_SAMPLE_LIMIT = 1_000_000
COMPONENT_KEYS = (
    "stage_logits",
    "pair_logits",
    "stage_component_scores",
    "pair_component_scores",
    "raw_fused_scores",
)


def _diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    difference = (left.detach().float().cpu() - right.detach().float().cpu()).abs().flatten()
    if not difference.numel():
        return {
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "total_elements": 0,
            "percentile_sampled_elements": 0,
        }
    total = int(difference.numel())
    if total > PERCENTILE_SAMPLE_LIMIT:
        stride = (total + PERCENTILE_SAMPLE_LIMIT - 1) // PERCENTILE_SAMPLE_LIMIT
        percentile_values = difference[::stride][:PERCENTILE_SAMPLE_LIMIT]
    else:
        percentile_values = difference
    return {
        "p50": float(torch.quantile(percentile_values, 0.50).item()),
        "p95": float(torch.quantile(percentile_values, 0.95).item()),
        "p99": float(torch.quantile(percentile_values, 0.99).item()),
        "max": float(difference.max().item()),
        "total_elements": total,
        "percentile_sampled_elements": int(percentile_values.numel()),
    }


def _load_features(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {"ids", "frame_hidden", "frame_chunk_size"}:
        raise RuntimeError(f"Frame feature artifact schema mismatch: {path}")
    if int(payload["frame_hidden"].shape[0]) != len(payload["ids"]):
        raise RuntimeError(f"Frame feature artifact row count mismatch: {path}")
    return payload


def certify_chunked_parity(
    *,
    legacy_raw: str | Path,
    legacy_repeat_raw: str | Path,
    chunked_raw: str | Path,
    legacy_features: str | Path,
    chunked_features: str | Path,
    calibration_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    legacy = load_raw_stage_pair_logits(legacy_raw)
    repeat = load_raw_stage_pair_logits(legacy_repeat_raw)
    chunked = load_raw_stage_pair_logits(chunked_raw)
    if legacy["ids"] != repeat["ids"] or legacy["ids"] != chunked["ids"]:
        raise RuntimeError("Parity runs have different sample IDs or order")
    if not torch.equal(legacy["target_perm_idx"], chunked["target_perm_idx"]):
        raise RuntimeError("Parity runs have different targets")
    legacy_feature_payload = _load_features(legacy_features)
    chunked_feature_payload = _load_features(chunked_features)
    if legacy_feature_payload["ids"] != legacy["ids"] or chunked_feature_payload["ids"] != legacy["ids"]:
        raise RuntimeError("Frame feature IDs do not match raw logit IDs")

    legacy_repeat_stats = {key: _diff_stats(legacy[key], repeat[key]) for key in COMPONENT_KEYS}
    chunk_stats = {key: _diff_stats(legacy[key], chunked[key]) for key in COMPONENT_KEYS}
    feature_stats = _diff_stats(
        legacy_feature_payload["frame_hidden"], chunked_feature_payload["frame_hidden"]
    )
    legacy_repeat_max = max(value["max"] for value in legacy_repeat_stats.values())
    threshold = max(2 * legacy_repeat_max + 1e-5, BF16_ABSOLUTE_TOLERANCE)
    finite = all(
        bool(torch.isfinite(payload[key]).all())
        for payload in (legacy, repeat, chunked)
        for key in COMPONENT_KEYS
    ) and bool(torch.isfinite(chunked_feature_payload["frame_hidden"]).all())
    raw_prediction_mismatches = int(
        legacy["raw_fused_scores"].argmax(dim=1).ne(chunked["raw_fused_scores"].argmax(dim=1)).sum().item()
    )
    calibration = load_calibration(calibration_path)
    legacy_calibrated = calibrated_structured_logits(
        legacy["stage_logits"], legacy["pair_logits"], calibration
    )
    chunked_calibrated = calibrated_structured_logits(
        chunked["stage_logits"], chunked["pair_logits"], calibration
    )
    calibrated_prediction_mismatches = int(
        legacy_calibrated.argmax(dim=1).ne(chunked_calibrated.argmax(dim=1)).sum().item()
    )
    true_rank_mismatches = int(
        legacy["true_class_rank"].ne(chunked["true_class_rank"]).sum().item()
    )
    max_chunk_drift = max(
        max(value["max"] for value in chunk_stats.values()),
        feature_stats["max"],
    )
    full_parity = (
        finite
        and raw_prediction_mismatches == 0
        and calibrated_prediction_mismatches == 0
        and true_rank_mismatches == 0
        and max_chunk_drift <= threshold
    )
    report = {
        "status": "PASS" if full_parity else "FAIL",
        "sample_count": len(legacy["ids"]),
        "predeclared_bf16_absolute_tolerance": BF16_ABSOLUTE_TOLERANCE,
        "effective_absolute_tolerance": threshold,
        "all_finite": finite,
        "legacy_repeat_max_abs_diff": legacy_repeat_max,
        "legacy_repeat_component_diff": legacy_repeat_stats,
        "chunk_vs_legacy_component_diff": chunk_stats,
        "pooled_feature_diff": feature_stats,
        "chunk_vs_legacy_max_abs_diff": max_chunk_drift,
        "raw_prediction_mismatch_count": raw_prediction_mismatches,
        "calibrated_prediction_mismatch_count": calibrated_prediction_mismatches,
        "true_rank_mismatch_count": true_rank_mismatches,
        "legacy_frame_chunk_size": legacy_feature_payload["frame_chunk_size"],
        "chunked_frame_chunk_size": chunked_feature_payload["frame_chunk_size"],
    }
    write_json(output_path, report)
    if not full_parity:
        raise RuntimeError(f"Chunked inference parity failed: {json.dumps(report, sort_keys=True)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-raw", required=True)
    parser.add_argument("--legacy-repeat-raw", required=True)
    parser.add_argument("--chunked-raw", required=True)
    parser.add_argument("--legacy-features", required=True)
    parser.add_argument("--chunked-features", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = certify_chunked_parity(
        legacy_raw=args.legacy_raw,
        legacy_repeat_raw=args.legacy_repeat_raw,
        chunked_raw=args.chunked_raw,
        legacy_features=args.legacy_features,
        chunked_features=args.chunked_features,
        calibration_path=args.calibration,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
