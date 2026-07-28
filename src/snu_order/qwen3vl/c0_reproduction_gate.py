from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from snu_order.utils.io import write_json

from .calibration_stage_pair import load_raw_stage_pair_logits
from .canonical_stage_pair_evaluation import canonical_cpu_float32_scores, semantic_prediction_sha256, stable_ranking


def compare(reference_path: str | Path, control_path: str | Path) -> dict:
    reference = load_raw_stage_pair_logits(reference_path)
    control = load_raw_stage_pair_logits(control_path)
    if reference["ids"] != control["ids"]:
        raise RuntimeError("C0 valid ID ordering differs from the v1 reference")
    if not torch.equal(reference["target_perm_idx"].long(), control["target_perm_idx"].long()):
        raise RuntimeError("C0 target permutation mapping differs from the v1 reference")
    reference_scores = canonical_cpu_float32_scores(reference["stage_logits"], reference["pair_logits"])
    control_scores = canonical_cpu_float32_scores(control["stage_logits"], control["pair_logits"])
    reference_rank = stable_ranking(reference_scores, reference["target_perm_idx"])
    control_rank = stable_ranking(control_scores, control["target_perm_idx"])
    assert reference_rank.gt_rank is not None and control_rank.gt_rank is not None
    prediction_mismatch = int(reference_rank.prediction.ne(control_rank.prediction).sum())
    rank_mismatch = int(reference_rank.gt_rank.ne(control_rank.gt_rank).sum())
    reference_correct = int(reference_rank.prediction.eq(reference["target_perm_idx"]).sum())
    control_correct = int(control_rank.prediction.eq(control["target_perm_idx"]).sum())
    status = (
        "C0_V1_EXACT_CONTRACT_REPRODUCTION_PASS"
        if prediction_mismatch == 0 and rank_mismatch == 0
        else "C0_V1_EXACT_CONTRACT_REPRODUCTION_FAIL"
    )
    return {
        "status": status,
        "sample_count": len(reference["ids"]),
        "reference_correct": reference_correct,
        "control_correct": control_correct,
        "correct_delta": control_correct - reference_correct,
        "prediction_mismatch": prediction_mismatch,
        "gt_rank_mismatch": rank_mismatch,
        "stage_logits_max_abs_diff": float((reference["stage_logits"].float() - control["stage_logits"].float()).abs().max()),
        "pair_logits_max_abs_diff": float((reference["pair_logits"].float() - control["pair_logits"].float()).abs().max()),
        "reference_prediction_sha256": semantic_prediction_sha256(reference["ids"], reference_rank),
        "control_prediction_sha256": semantic_prediction_sha256(control["ids"], control_rank),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-raw", required=True)
    parser.add_argument("--control-raw", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.reference_raw, args.control_raw)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "C0_V1_EXACT_CONTRACT_REPRODUCTION_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
