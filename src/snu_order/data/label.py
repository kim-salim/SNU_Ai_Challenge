from __future__ import annotations

from collections.abc import Sequence

from snu_order.order.answer_convert import answer_to_perm, validate_answer
from snu_order.order.permutation24 import perm_to_index

PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def answer_to_perm_index(answer: Sequence[int]) -> int:
    return perm_to_index(answer_to_perm(answer))


def answer_to_pairwise_labels(answer: Sequence[int]) -> list[float]:
    values = validate_answer(answer)
    return [1.0 if values[i] < values[j] else 0.0 for i, j in PAIRS]

