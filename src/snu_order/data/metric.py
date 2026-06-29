from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from snu_order.data.label import answer_to_pairwise_labels
from snu_order.order.answer_convert import validate_answer


def exact_match_accuracy(
    pred_answers: Sequence[Sequence[int]],
    true_answers: Sequence[Sequence[int]],
) -> float:
    if len(pred_answers) != len(true_answers):
        raise ValueError(
            f"pred_answers and true_answers must have same length, got "
            f"{len(pred_answers)} and {len(true_answers)}"
        )
    if len(true_answers) == 0:
        return 0.0
    correct = 0
    for pred, true in zip(pred_answers, true_answers, strict=True):
        correct += int(validate_answer(pred) == validate_answer(true))
    return correct / len(true_answers)


def pairwise_accuracy(
    pred_answers: Sequence[Sequence[int]],
    true_answers: Sequence[Sequence[int]],
) -> float:
    if len(pred_answers) != len(true_answers):
        raise ValueError(
            f"pred_answers and true_answers must have same length, got "
            f"{len(pred_answers)} and {len(true_answers)}"
        )
    correct = 0
    total = 0
    for pred, true in zip(pred_answers, true_answers, strict=True):
        pred_pair = answer_to_pairwise_labels(pred)
        true_pair = answer_to_pairwise_labels(true)
        correct += sum(int(p == t) for p, t in zip(pred_pair, true_pair, strict=True))
        total += len(true_pair)
    return correct / max(total, 1)


def top1_top2_margin(scores: Sequence[Sequence[float]] | np.ndarray) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 24:
        raise ValueError(f"scores must have shape [N, 24], got {arr.shape}")
    if arr.shape[0] == 0:
        return 0.0
    sorted_scores = np.sort(arr, axis=1)
    return float(np.mean(sorted_scores[:, -1] - sorted_scores[:, -2]))
