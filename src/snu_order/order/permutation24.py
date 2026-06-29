from __future__ import annotations

import itertools
from collections.abc import Sequence

from snu_order.order.answer_convert import validate_perm

PERMS: tuple[tuple[int, int, int, int], ...] = tuple(itertools.permutations(range(4)))
PERM_TO_INDEX = {perm: idx for idx, perm in enumerate(PERMS)}
INDEX_TO_PERM = {idx: perm for idx, perm in enumerate(PERMS)}


def get_all_perms() -> tuple[tuple[int, int, int, int], ...]:
    return PERMS


def perm_to_index(perm: Sequence[int]) -> int:
    checked = validate_perm(perm)
    return PERM_TO_INDEX[checked]


def index_to_perm(index: int) -> tuple[int, int, int, int]:
    idx = int(index)
    if idx < 0 or idx >= len(PERMS):
        raise ValueError(f"permutation index must be in [0, 23], got {index}")
    return INDEX_TO_PERM[idx]

