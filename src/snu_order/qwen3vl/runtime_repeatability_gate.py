from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_json

from .calibration_stage_pair import load_raw_stage_pair_logits
from .canonical_stage_pair_evaluation import (
    canonical_cpu_float32_scores,
    semantic_prediction_sha256,
    stable_ranking,
)


def audit_repeats(paths: list[str | Path]) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError("Exactly three fresh-process raw artifacts are required")
    payloads = [load_raw_stage_pair_logits(path) for path in paths]
    reference = payloads[0]
    reference_scores = canonical_cpu_float32_scores(reference["stage_logits"], reference["pair_logits"])
    reference_rank = stable_ranking(reference_scores, reference["target_perm_idx"])
    if reference_rank.gt_rank is None:
        raise AssertionError("GT ranks were not computed")
    runs = []
    pairwise = []
    for index, payload in enumerate(payloads, start=1):
        if payload["ids"] != reference["ids"]:
            raise RuntimeError(f"Fresh run {index} ID ordering mismatch")
        if not torch.equal(payload["target_perm_idx"].long(), reference["target_perm_idx"].long()):
            raise RuntimeError(f"Fresh run {index} target mapping mismatch")
        scores = canonical_cpu_float32_scores(payload["stage_logits"], payload["pair_logits"])
        ranking = stable_ranking(scores, payload["target_perm_idx"])
        if ranking.gt_rank is None:
            raise AssertionError("GT ranks were not computed")
        runs.append(
            {
                "run": index,
                "raw_path": str(Path(paths[index - 1]).resolve()),
                "correct_count": int(ranking.prediction.eq(payload["target_perm_idx"].long()).sum()),
                "semantic_prediction_sha256": semantic_prediction_sha256(payload["ids"], ranking),
            }
        )
    for left in range(3):
        for right in range(left + 1, 3):
            left_payload = payloads[left]
            right_payload = payloads[right]
            left_scores = canonical_cpu_float32_scores(left_payload["stage_logits"], left_payload["pair_logits"])
            right_scores = canonical_cpu_float32_scores(right_payload["stage_logits"], right_payload["pair_logits"])
            left_rank = stable_ranking(left_scores, left_payload["target_perm_idx"])
            right_rank = stable_ranking(right_scores, right_payload["target_perm_idx"])
            assert left_rank.gt_rank is not None and right_rank.gt_rank is not None
            pairwise.append(
                {
                    "left_run": left + 1,
                    "right_run": right + 1,
                    "prediction_mismatch": int(left_rank.prediction.ne(right_rank.prediction).sum()),
                    "gt_rank_mismatch": int(left_rank.gt_rank.ne(right_rank.gt_rank).sum()),
                    "stage_logits_max_abs_diff": float(
                        (left_payload["stage_logits"].float() - right_payload["stage_logits"].float()).abs().max()
                    ),
                    "pair_logits_max_abs_diff": float(
                        (left_payload["pair_logits"].float() - right_payload["pair_logits"].float()).abs().max()
                    ),
                    "candidate_scores_max_abs_diff": float((left_scores - right_scores).abs().max()),
                }
            )
    passed = all(
        row["prediction_mismatch"] == 0 and row["gt_rank_mismatch"] == 0
        for row in pairwise
    ) and len({row["semantic_prediction_sha256"] for row in runs}) == 1
    return {
        "status": "P0_FRESH_PROCESS_REPEATABILITY_PASS" if passed else "P0_FRESH_PROCESS_REPEATABILITY_FAIL",
        "sample_count": len(reference["ids"]),
        "runs": runs,
        "pairwise": pairwise,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_repeats(args.raw)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "P0_FRESH_PROCESS_REPEATABILITY_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
