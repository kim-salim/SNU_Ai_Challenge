from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_csv_rows, write_json

from .metrics24 import compute_metrics_from_logits, rows_from_logits
from .permutations import PAIRS, pairwise_labels_from_answer, perm_index_to_answer
from .stage_pair_scorer import pair_targets_from_answer, stage_targets_from_answer


def _position_stage_accuracy(stage_logits: torch.Tensor, answer: torch.Tensor) -> tuple[float, dict[str, float]]:
    pred = stage_logits.detach().float().argmax(dim=-1).cpu()
    targets = stage_targets_from_answer(answer.detach().cpu())
    correct = pred.eq(targets)
    by_pos = {}
    for pos in range(4):
        mask = targets.eq(pos)
        by_pos[str(pos + 1)] = float(correct[mask].float().mean().item()) if bool(mask.any()) else 0.0
    return float(correct.float().mean().item()) if correct.numel() else 0.0, by_pos


def _pair_head_metrics(pair_logits: torch.Tensor | None, answer: torch.Tensor) -> dict[str, Any]:
    if pair_logits is None:
        return {
            "pairwise_head_accuracy": 0.0,
            "pairwise_head_accuracy_by_pair": {},
            "pairwise_head_accuracy_by_temporal_gap": {},
        }
    pred = pair_logits.detach().float().gt(0).cpu()
    targets = pair_targets_from_answer(answer.detach().cpu()).bool()
    correct = pred.eq(targets)
    by_pair = {
        f"{left + 1}-{right + 1}": float(correct[:, pair_index].float().mean().item())
        for pair_index, (left, right) in enumerate(PAIRS)
    }
    answers = answer.detach().long().cpu()
    by_gap: dict[str, float] = {}
    for gap in (1, 2, 3):
        masks = []
        values = []
        for pair_index, (left, right) in enumerate(PAIRS):
            mask = answers[:, left].sub(answers[:, right]).abs().eq(gap)
            masks.append(mask)
            values.append(correct[:, pair_index])
        gap_mask = torch.stack(masks, dim=1)
        gap_correct = torch.stack(values, dim=1)
        by_gap[str(gap)] = (
            float(gap_correct[gap_mask].float().mean().item()) if bool(gap_mask.any()) else 0.0
        )
    return {
        "pairwise_head_accuracy": float(correct.float().mean().item()) if pred.numel() else 0.0,
        "pairwise_head_accuracy_by_pair": by_pair,
        "pairwise_head_accuracy_by_temporal_gap": by_gap,
    }


def _margin_metrics(final_logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    scores = final_logits.detach().float().cpu()
    target_values = targets.detach().long().view(-1).cpu()
    if not scores.shape[0]:
        return {
            "mean_correct_margin": 0.0,
            "mean_wrong_margin": 0.0,
            "high_margin_wrong_threshold": 1.0,
            "high_margin_wrong_count": 0,
        }
    top2 = torch.topk(scores, k=2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    correct = scores.argmax(dim=1).eq(target_values)
    return {
        "mean_correct_margin": float(margins[correct].mean().item()) if bool(correct.any()) else 0.0,
        "mean_wrong_margin": float(margins[~correct].mean().item()) if bool((~correct).any()) else 0.0,
        "median_wrong_margin": float(statistics.median(margins[~correct].tolist())) if bool((~correct).any()) else 0.0,
        "high_margin_wrong_threshold": 1.0,
        "high_margin_wrong_count": int(((~correct) & margins.ge(1.0)).sum().item()),
    }


def compute_stage_pair_metrics(
    final_logits: torch.Tensor,
    target_perm_idx: torch.Tensor,
    *,
    stage_logits: torch.Tensor,
    pair_logits: torch.Tensor | None,
    answer: torch.Tensor,
    latencies: list[float] | None = None,
    shuffle_consistency_rate: float | None = None,
) -> dict[str, Any]:
    metrics = compute_metrics_from_logits(final_logits, target_perm_idx, latencies=latencies)
    stage_acc, stage_by_position = _position_stage_accuracy(stage_logits, answer)
    metrics["stage_accuracy"] = stage_acc
    metrics["stage_accuracy_by_position"] = stage_by_position
    metrics.update(_pair_head_metrics(pair_logits, answer))
    metrics.update(_margin_metrics(final_logits, target_perm_idx))
    kendall = metrics.get("kendall_distance_histogram", {})
    metrics["kendall_distance_le_2_error_count"] = sum(
        int(kendall.get(str(distance), 0)) for distance in (1, 2)
    )
    metrics["kendall_distance_ge_4_error_count"] = sum(
        int(kendall.get(str(distance), 0)) for distance in (4, 5, 6)
    )
    if shuffle_consistency_rate is not None:
        metrics["shuffle_consistency_rate"] = float(shuffle_consistency_rate)
    metrics["complete_reverse_error_count"] = metrics.get("exact_reverse_error_count", 0)
    metrics["complete_reverse_error_rate"] = metrics.get("exact_reverse_error_rate", 0.0)
    return metrics


def write_stage_pair_artifacts(
    output_dir: str | Path,
    ids: list[str],
    final_logits: torch.Tensor,
    targets: torch.Tensor,
    metrics: dict[str, Any],
    *,
    stage_logits: torch.Tensor | None = None,
    pair_logits: torch.Tensor | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = rows_from_logits(ids, final_logits, targets)
    fieldnames = ["Id", "true_answer", "pred_answer", "true_perm_idx", "pred_perm_idx", "gt_rank", "top1_margin", "correct"]
    write_csv_rows(out / "valid_predictions.csv", rows, fieldnames)
    write_csv_rows(out / "wrong_cases.csv", [row for row in rows if not int(row["correct"])], fieldnames)
    scores = final_logits.detach().float().cpu()
    stage_scores = None if stage_logits is None else stage_logits.detach().float().cpu()
    pair_scores = None if pair_logits is None else pair_logits.detach().float().cpu()
    with (out / "raw_scores.jsonl").open("w", encoding="utf-8") as f:
        for idx, (sample_id, score_row, row) in enumerate(zip(ids, scores.tolist(), rows, strict=True)):
            payload: dict[str, Any] = {
                "Id": sample_id,
                "logits": [float(v) for v in score_row],
                "true_perm_idx": row["true_perm_idx"],
                "pred_perm_idx": row["pred_perm_idx"],
                "gt_rank": row["gt_rank"],
            }
            if stage_scores is not None:
                payload["stage_logits"] = stage_scores[idx].tolist()
            if pair_scores is not None:
                payload["pair_logits"] = pair_scores[idx].tolist()
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    write_json(out / "metrics.json", metrics)
    write_json(out / "class_histogram.json", {
        "predicted": metrics.get("predicted_class_histogram", {}),
        "true": metrics.get("true_class_histogram", {}),
    })


def compare_with_baseline(new_predictions_csv: str | Path, baseline_predictions_csv: str | Path) -> dict[str, Any] | None:
    baseline_path = Path(baseline_predictions_csv)
    if not baseline_path.exists():
        return None
    new_rows = _read_prediction_correctness(new_predictions_csv)
    old_rows = _read_prediction_correctness(baseline_path)
    common = sorted(set(new_rows) & set(old_rows))
    old_correct_new_correct = 0
    old_correct_new_wrong = 0
    old_wrong_new_correct = 0
    both_wrong = 0
    for sample_id in common:
        old = old_rows[sample_id]
        new = new_rows[sample_id]
        old_correct_new_correct += int(old and new)
        old_correct_new_wrong += int(old and not new)
        old_wrong_new_correct += int((not old) and new)
        both_wrong += int((not old) and (not new))
    b = old_correct_new_wrong
    c = old_wrong_new_correct
    mcnemar = ((abs(b - c) - 1) ** 2 / max(b + c, 1)) if (b + c) else 0.0
    old_acc = sum(int(old_rows[s]) for s in common) / max(len(common), 1)
    new_acc = sum(int(new_rows[s]) for s in common) / max(len(common), 1)
    return {
        "common_count": len(common),
        "old_correct_new_correct": old_correct_new_correct,
        "old_correct_new_wrong": old_correct_new_wrong,
        "old_wrong_new_correct": old_wrong_new_correct,
        "both_wrong": both_wrong,
        "old_exact_match": old_acc,
        "new_exact_match": new_acc,
        "exact_match_delta": new_acc - old_acc,
        "mcnemar_statistic": mcnemar,
    }


def _read_prediction_correctness(path: str | Path) -> dict[str, bool]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        id_col = "Id" if "Id" in reader.fieldnames else "id"
        correct_col = "correct" if "correct" in reader.fieldnames else "exact_match"
        result = {}
        for row in reader:
            result[str(row[id_col])] = str(row[correct_col]).strip().lower() in {"1", "true", "yes"}
        return result
