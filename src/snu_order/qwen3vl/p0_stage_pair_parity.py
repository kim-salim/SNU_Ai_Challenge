from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_json

from .canonical_stage_pair_evaluation import (
    canonical_cpu_float32_scores,
    semantic_prediction_sha256,
    stable_ranking,
)
from .metrics_stage_pair import compute_stage_pair_metrics


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def audit_raw_stage_pair_artifact(
    raw_logits_path: str | Path,
    *,
    training_metrics_path: str | Path | None = None,
    repeat_count: int = 3,
) -> dict[str, Any]:
    payload = torch.load(raw_logits_path, map_location="cpu", weights_only=False)
    required = {"ids", "stage_logits", "pair_logits", "target_perm_idx", "answer"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError("P0 requires a Stage/Pair raw-logit artifact")
    ids = [str(value) for value in payload["ids"]]
    targets = payload["target_perm_idx"].long().cpu()
    answers = payload["answer"].long().cpu()
    repeated: list[dict[str, Any]] = []
    reference_scores: torch.Tensor | None = None
    reference_prediction: torch.Tensor | None = None
    reference_rank: torch.Tensor | None = None
    for index in range(int(repeat_count)):
        scores = canonical_cpu_float32_scores(payload["stage_logits"], payload["pair_logits"])
        ranking = stable_ranking(scores, targets)
        assert ranking.gt_rank is not None
        if reference_scores is None:
            reference_scores = scores
            reference_prediction = ranking.prediction
            reference_rank = ranking.gt_rank
        repeated.append(
            {
                "repeat": index + 1,
                "score_bitwise_equal": bool(torch.equal(scores, reference_scores)),
                "prediction_mismatch": int(ranking.prediction.ne(reference_prediction).sum().item()),
                "gt_rank_mismatch": int(ranking.gt_rank.ne(reference_rank).sum().item()),
                "semantic_prediction_sha256": semantic_prediction_sha256(ids, ranking),
            }
        )
    assert reference_scores is not None and reference_prediction is not None and reference_rank is not None
    metrics = compute_stage_pair_metrics(
        reference_scores,
        targets,
        stage_logits=payload["stage_logits"],
        pair_logits=payload["pair_logits"],
        answer=answers,
    )
    stored: dict[str, Any] = {}
    if "raw_fused_scores" in payload:
        old_scores = payload["raw_fused_scores"].float().cpu()
        stored["score_max_abs_diff"] = float((old_scores - reference_scores).abs().max().item())
    if "raw_prediction" in payload:
        stored["prediction_mismatch"] = int(payload["raw_prediction"].long().cpu().ne(reference_prediction).sum().item())
    if "true_class_rank" in payload:
        old_rank = payload["true_class_rank"].long().cpu()
        rank_mismatch = old_rank.ne(reference_rank)
        stored["gt_rank_mismatch"] = int(rank_mismatch.sum().item())
        tie_explained = 0
        for row_index in rank_mismatch.nonzero(as_tuple=False).flatten().tolist():
            target_score = reference_scores[row_index, int(targets[row_index])]
            tied_classes = reference_scores[row_index].eq(target_score).sum()
            tie_explained += int(int(tied_classes.item()) > 1)
        stored["gt_rank_mismatch_exact_tie_count"] = tie_explained
        stored["gt_rank_mismatch_all_exact_ties"] = tie_explained == stored["gt_rank_mismatch"]
    training_metrics = _load_json(training_metrics_path)
    training_correct = None if training_metrics is None else training_metrics.get("correct_count")
    training_parity = training_correct is None or int(training_correct) == int(metrics["correct_count"])
    repeated_pass = all(
        row["score_bitwise_equal"]
        and row["prediction_mismatch"] == 0
        and row["gt_rank_mismatch"] == 0
        for row in repeated
    )
    stored_prediction_pass = stored.get("prediction_mismatch", 0) == 0
    stored_rank_pass = stored.get("gt_rank_mismatch", 0) == 0 or stored.get(
        "gt_rank_mismatch_all_exact_ties", False
    )
    status = (
        "P0_CANONICAL_SCORER_PARITY_PASS"
        if repeated_pass and stored_prediction_pass and stored_rank_pass and training_parity
        else "P0_CANONICAL_SCORER_PARITY_FAIL"
    )
    return {
        "status": status,
        "scorer_mode": "canonical_cpu_float32",
        "sample_count": len(ids),
        "correct_count": int(metrics["correct_count"]),
        "exact_match": float(metrics["exact_match"]),
        "MRR": float(metrics["MRR"]),
        "top3_accuracy": float(metrics["top3_accuracy"]),
        "stored_artifact_comparison": stored,
        "training_metrics_correct_count": training_correct,
        "training_correct_parity": training_parity,
        "repeats": repeated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-logits", required=True)
    parser.add_argument("--training-metrics")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-count", type=int, default=3)
    args = parser.parse_args()
    result = audit_raw_stage_pair_artifact(
        args.raw_logits,
        training_metrics_path=args.training_metrics,
        repeat_count=args.repeat_count,
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "P0_CANONICAL_SCORER_PARITY_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
