from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from snu_order.order.answer_convert import perm_to_answer
from snu_order.order.permutation24 import index_to_perm


def scores_to_perm_indices(scores: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(scores)
    if arr.ndim != 2 or arr.shape[1] != 24:
        raise ValueError(f"scores must have shape [N, 24], got {arr.shape}")
    return np.argmax(arr, axis=1).astype(np.int64)


def scores_to_answers(scores: Sequence[Sequence[float]] | np.ndarray) -> list[list[int]]:
    return [perm_to_answer(index_to_perm(int(idx))) for idx in scores_to_perm_indices(scores)]

