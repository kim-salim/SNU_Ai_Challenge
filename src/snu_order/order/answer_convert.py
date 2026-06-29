from __future__ import annotations

from collections.abc import Sequence


def _as_int_list(values: Sequence[int], *, name: str) -> list[int]:
    try:
        converted = [int(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc
    return converted


def validate_answer(answer: Sequence[int]) -> list[int]:
    values = _as_int_list(answer, name="answer")
    if len(values) != 4:
        raise ValueError(f"answer must have length 4, got {len(values)}")
    if sorted(values) != [1, 2, 3, 4]:
        raise ValueError(f"answer must be a permutation of [1, 2, 3, 4], got {values}")
    return values


def validate_perm(perm: Sequence[int]) -> tuple[int, int, int, int]:
    values = _as_int_list(perm, name="perm")
    if len(values) != 4:
        raise ValueError(f"perm must have length 4, got {len(values)}")
    if sorted(values) != [0, 1, 2, 3]:
        raise ValueError(f"perm must be a permutation of [0, 1, 2, 3], got {values}")
    return tuple(values)  # type: ignore[return-value]


def answer_to_perm(answer: Sequence[int]) -> tuple[int, int, int, int]:
    """Convert official answer format to input-frame-index temporal order."""
    values = validate_answer(answer)
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    return tuple(frame_idx for frame_idx, _ in pairs)  # type: ignore[return-value]


def perm_to_answer(perm: Sequence[int]) -> list[int]:
    """Convert input-frame-index temporal order to official answer format."""
    values = validate_perm(perm)
    answer: list[int | None] = [None] * 4
    for original_pos, frame_idx in enumerate(values, start=1):
        answer[frame_idx] = original_pos
    return [int(v) for v in answer]

