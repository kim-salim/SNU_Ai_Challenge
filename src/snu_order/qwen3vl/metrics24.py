from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_csv_rows, write_json

from .permutations import PAIRS, PERMS, pairwise_labels_from_answer, perm_index_to_answer
from .canonical_stage_pair_evaluation import stable_ranking


def kendall_distance(answer_a: list[int], answer_b: list[int]) -> int:
    distance = 0
    for i, j in PAIRS:
        if (answer_a[i] < answer_a[j]) != (answer_b[i] < answer_b[j]):
            distance += 1
    return distance


def is_exact_reverse(pred_idx: int, true_idx: int) -> bool:
    return tuple(PERMS[int(pred_idx)]) == tuple(reversed(PERMS[int(true_idx)]))


def _rank_of_gt(scores: torch.Tensor, target_idx: int) -> int:
    ranking = stable_ranking(scores.view(1, -1), torch.tensor([int(target_idx)]))
    assert ranking.gt_rank is not None
    return int(ranking.gt_rank[0].item())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percentile
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return float(ordered[lower] * (1 - frac) + ordered[upper] * frac)


def compute_metrics_from_logits(
    logits: torch.Tensor,
    target_perm_idx: torch.Tensor,
    *,
    latencies: list[float] | None = None,
) -> dict[str, Any]:
    if logits.ndim != 2 or logits.shape[1] != len(PERMS):
        raise ValueError(f"logits must have shape [N,24], got {tuple(logits.shape)}")
    targets = target_perm_idx.long().view(-1).cpu()
    scores = logits.detach().float().cpu()
    if scores.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have same first dimension")
    n = int(scores.shape[0])
    ranking = stable_ranking(scores, targets)
    pred_idx = ranking.prediction
    correct = pred_idx.eq(targets)
    ranks = [] if ranking.gt_rank is None else ranking.gt_rank.tolist()
    margins = ranking.top1_margin.tolist()

    pred_answers = [perm_index_to_answer(int(idx)) for idx in pred_idx.tolist()]
    true_answers = [perm_index_to_answer(int(idx)) for idx in targets.tolist()]
    pairwise_total = 0
    pairwise_correct = 0
    position_total = 0
    position_correct = 0
    reverse_count = 0
    adjacent_count = 0
    kendall_hist: Counter[str] = Counter()
    for pred_i, true_i, pred_answer, true_answer in zip(
        pred_idx.tolist(), targets.tolist(), pred_answers, true_answers, strict=True
    ):
        pred_pair = pairwise_labels_from_answer(pred_answer)
        true_pair = pairwise_labels_from_answer(true_answer)
        pairwise_correct += sum(int(a == b) for a, b in zip(pred_pair, true_pair, strict=True))
        pairwise_total += len(true_pair)
        position_correct += sum(int(a == b) for a, b in zip(pred_answer, true_answer, strict=True))
        position_total += 4
        reverse_count += int(is_exact_reverse(int(pred_i), int(true_i)))
        kd = kendall_distance(pred_answer, true_answer)
        adjacent_count += int(kd == 1)
        kendall_hist[str(kd)] += 1

    topk = {}
    sorted_indices = ranking.order
    for k in (1, 2, 3, 5):
        if n:
            topk[f"top{k}_accuracy"] = float((sorted_indices[:, :k] == targets[:, None]).any(dim=1).float().mean())
        else:
            topk[f"top{k}_accuracy"] = 0.0

    lat = latencies or []
    metrics: dict[str, Any] = {
        "sample_count": n,
        "correct_count": int(correct.sum().item()),
        "exact_match": float(correct.float().mean().item()) if n else 0.0,
        "pairwise_accuracy": pairwise_correct / max(pairwise_total, 1),
        "frame_position_accuracy": position_correct / max(position_total, 1),
        **topk,
        "mean_gt_rank": statistics.mean(ranks) if ranks else 0.0,
        "median_gt_rank": statistics.median(ranks) if ranks else 0.0,
        "MRR": statistics.mean([1.0 / rank for rank in ranks]) if ranks else 0.0,
        "mean_top1_top2_margin": statistics.mean(margins) if margins else 0.0,
        "exact_reverse_error_count": reverse_count,
        "exact_reverse_error_rate": reverse_count / max(n, 1),
        "adjacent_swap_error_count": adjacent_count,
        "adjacent_swap_error_rate": adjacent_count / max(n, 1),
        "kendall_distance_histogram": dict(sorted(kendall_hist.items(), key=lambda item: int(item[0]))),
        "predicted_class_histogram": dict(sorted(Counter(str(v) for v in pred_idx.tolist()).items(), key=lambda item: int(item[0]))),
        "true_class_histogram": dict(sorted(Counter(str(v) for v in targets.tolist()).items(), key=lambda item: int(item[0]))),
        "parse_failure_count": 0,
        "invalid_permutation_count": 0,
        "mean_latency_sec": statistics.mean(lat) if lat else 0.0,
        "p50_latency_sec": statistics.median(lat) if lat else 0.0,
        "p95_latency_sec": _percentile(lat, 0.95) if lat else 0.0,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        "peak_reserved_vram_gb": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
    }
    return metrics


def rows_from_logits(ids: list[str], logits: torch.Tensor, targets: torch.Tensor) -> list[dict[str, Any]]:
    scores = logits.detach().float().cpu()
    target_values = targets.long().view(-1).cpu()
    ranking = stable_ranking(scores, target_values)
    pred_idx = ranking.prediction
    rows: list[dict[str, Any]] = []
    for sample_id, score_row, true_idx, pred in zip(ids, scores, target_values, pred_idx, strict=True):
        assert ranking.gt_rank is not None
        row_index = len(rows)
        gt_rank = int(ranking.gt_rank[row_index].item())
        true_answer = perm_index_to_answer(int(true_idx))
        pred_answer = perm_index_to_answer(int(pred))
        rows.append(
            {
                "Id": sample_id,
                "true_answer": json.dumps(true_answer),
                "pred_answer": json.dumps(pred_answer),
                "true_perm_idx": int(true_idx),
                "pred_perm_idx": int(pred),
                "gt_rank": gt_rank,
                "top1_margin": float(ranking.top1_margin[row_index]),
                "correct": int(int(true_idx) == int(pred)),
            }
        )
    return rows


def write_eval_artifacts(
    output_dir: str | Path,
    ids: list[str],
    logits: torch.Tensor,
    targets: torch.Tensor,
    metrics: dict[str, Any],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = rows_from_logits(ids, logits, targets)
    fieldnames = ["Id", "true_answer", "pred_answer", "true_perm_idx", "pred_perm_idx", "gt_rank", "top1_margin", "correct"]
    write_csv_rows(out / "valid_predictions.csv", rows, fieldnames)
    write_csv_rows(out / "wrong_cases.csv", [row for row in rows if not int(row["correct"])], fieldnames)
    with (out / "raw_scores.jsonl").open("w", encoding="utf-8") as f:
        for sample_id, score_row, row in zip(ids, logits.detach().float().cpu().tolist(), rows, strict=True):
            payload = {
                "Id": sample_id,
                "logits": [float(v) for v in score_row],
                "true_perm_idx": row["true_perm_idx"],
                "pred_perm_idx": row["pred_perm_idx"],
                "gt_rank": row["gt_rank"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    write_json(out / "metrics.json", metrics)
    write_json(out / "class_histogram.json", {
        "predicted": metrics.get("predicted_class_histogram", {}),
        "true": metrics.get("true_class_histogram", {}),
    })
