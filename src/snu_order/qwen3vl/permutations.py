from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from snu_order.order.answer_convert import answer_to_perm, perm_to_answer, validate_answer, validate_perm
from snu_order.order.permutation24 import PERMS, index_to_perm, perm_to_index

PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
FRAME_LABELS: tuple[str, str, str, str] = ("Frame A", "Frame B", "Frame C", "Frame D")


def answer_to_perm_index(answer: Sequence[int]) -> int:
    return perm_to_index(answer_to_perm(validate_answer(answer)))


def perm_index_to_answer(index: int) -> list[int]:
    return perm_to_answer(index_to_perm(int(index)))


def perm_index_to_order(index: int) -> tuple[int, int, int, int]:
    return index_to_perm(int(index))


def order_to_perm_index(order: Sequence[int]) -> int:
    return perm_to_index(validate_perm(order))


def remap_answer_for_input_shuffle(answer: Sequence[int], shuffle_idx: Sequence[int]) -> list[int]:
    checked_answer = validate_answer(answer)
    checked_shuffle = validate_perm(shuffle_idx)
    return validate_answer([checked_answer[i] for i in checked_shuffle])


def apply_frame_permutation(
    frames: Sequence[Any],
    answer: Sequence[int],
    shuffle_idx: Sequence[int],
) -> tuple[list[Any], list[int]]:
    checked_shuffle = validate_perm(shuffle_idx)
    if len(frames) != 4:
        raise ValueError(f"frames must have length 4, got {len(frames)}")
    new_frames = [frames[i] for i in checked_shuffle]
    new_answer = remap_answer_for_input_shuffle(answer, checked_shuffle)
    return new_frames, new_answer


def uniform_shuffle_for_sample(seed: int, sample_index: int, epoch: int = 0) -> tuple[int, int, int, int]:
    rng = random.Random(int(seed) + int(sample_index) * 1009 + int(epoch) * 1_000_003)
    return PERMS[rng.randrange(len(PERMS))]


def position_labels_from_answer(answer: Sequence[int]) -> list[int]:
    checked = validate_answer(answer)
    return [value - 1 for value in checked]


def pairwise_labels_from_answer(answer: Sequence[int]) -> list[int]:
    checked = validate_answer(answer)
    return [int(checked[i] < checked[j]) for i, j in PAIRS]


def answers_from_perm_indices(indices: Sequence[int]) -> list[list[int]]:
    return [perm_index_to_answer(int(index)) for index in indices]


def validate_perm_index(index: int) -> int:
    value = int(index)
    if value < 0 or value >= len(PERMS):
        raise ValueError(f"permutation index must be in [0, 23], got {index}")
    return value
