from __future__ import annotations

from collections.abc import Sequence


def assert_disjoint_ids(left: Sequence[str], right: Sequence[str]) -> None:
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping Id values")

