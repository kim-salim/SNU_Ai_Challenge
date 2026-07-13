from __future__ import annotations

import hashlib
import itertools
import random
from collections.abc import Sequence
from typing import Any


DEFAULT_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWX"
FRAME_LABELS = ("F1", "F2", "F3", "F4")


def validate_answer(answer: Sequence[int]) -> list[int]:
    values = [int(v) for v in answer]
    if len(values) != 4:
        raise ValueError(f"answer must have length 4, got {len(values)}")
    if set(values) != {1, 2, 3, 4}:
        raise ValueError(f"answer values must be exactly {{1,2,3,4}}, got {values}")
    return values


def order_to_answer(order: Sequence[int]) -> list[int]:
    order_tuple = tuple(int(v) for v in order)
    if len(order_tuple) != 4 or set(order_tuple) != {0, 1, 2, 3}:
        raise ValueError(f"order must be a permutation of 0..3, got {order_tuple}")
    answer = [0, 0, 0, 0]
    for chronological_position, frame_index in enumerate(order_tuple, start=1):
        answer[frame_index] = chronological_position
    return validate_answer(answer)


def answer_to_order(answer: Sequence[int]) -> tuple[int, int, int, int]:
    values = validate_answer(answer)
    return tuple(sorted(range(4), key=lambda idx: values[idx]))  # type: ignore[return-value]


def build_24_candidates(option_labels: str = DEFAULT_LABELS) -> list[dict[str, Any]]:
    if len(option_labels) < 24:
        raise ValueError(f"option_labels must contain at least 24 labels, got {option_labels!r}")
    candidates: list[dict[str, Any]] = []
    for index, order in enumerate(itertools.permutations(range(4))):
        labels = [FRAME_LABELS[idx] for idx in order]
        candidates.append(
            {
                "index": index,
                "label": option_labels[index],
                "order": tuple(order),
                "text": " ".join(labels),
            }
        )
    return candidates


def relabel_candidates(
    candidates: Sequence[dict[str, Any]],
    option_labels: str = DEFAULT_LABELS,
) -> list[dict[str, Any]]:
    if len(option_labels) < len(candidates):
        raise ValueError("option_labels is shorter than candidates")
    relabeled: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        updated = dict(candidate)
        updated["index"] = index
        updated["label"] = option_labels[index]
        relabeled.append(updated)
    return relabeled


def deterministic_shuffle_candidates(
    candidates: Sequence[dict[str, Any]],
    sample_id: str,
    option_labels: str = DEFAULT_LABELS,
) -> list[dict[str, Any]]:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    shuffled = [dict(candidate) for candidate in candidates]
    rng.shuffle(shuffled)
    return relabel_candidates(shuffled, option_labels=option_labels)


def _candidate_by_label(label: str, candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    normalized = str(label).strip().upper()
    for candidate in candidates:
        if str(candidate["label"]).upper() == normalized:
            return candidate
    available = [str(candidate["label"]) for candidate in candidates]
    raise ValueError(f"Unknown candidate label {label!r}. Available labels: {available}")


def candidate_label_to_order(
    label: str,
    candidates: Sequence[dict[str, Any]],
) -> tuple[int, int, int, int]:
    order = _candidate_by_label(label, candidates)["order"]
    return tuple(int(v) for v in order)  # type: ignore[return-value]


def candidate_label_to_answer(label: str, candidates: Sequence[dict[str, Any]]) -> list[int]:
    return order_to_answer(candidate_label_to_order(label, candidates))
