from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .checkpoint import checkpoint_sha256, load_checkpoint_metrics


def is_valid_b_path(path: str | Path) -> bool:
    text = str(path).lower()
    return "valid_b" in text or "lockbox" in text


def guard_not_lockbox_path(path: str | Path, *, purpose: str) -> None:
    if is_valid_b_path(path):
        raise ValueError(f"valid_b/lockbox path cannot be used for {purpose}: {path}")


def required_correct_count(sample_count: int, required_accuracy: float = 0.30) -> int:
    return int(math.ceil(float(required_accuracy) * int(sample_count)))


def check_lockbox_unlock(unlock_valid_b: bool) -> None:
    if not bool(unlock_valid_b):
        raise PermissionError("valid_b lockbox evaluation requires --unlock-valid-b")


def check_valid_a_threshold(
    checkpoint_path: str | Path,
    *,
    min_valid_a_accuracy: float,
    force: bool = False,
) -> float:
    metrics = load_checkpoint_metrics(checkpoint_path)
    valid_a = metrics.get("valid_a_exact_match", metrics.get("exact_match", metrics.get("best_exact_match")))
    if valid_a is None:
        raise ValueError(f"checkpoint does not contain valid_a exact_match metric: {checkpoint_path}")
    valid_a_f = float(valid_a)
    if valid_a_f < float(min_valid_a_accuracy) and not bool(force):
        raise PermissionError(
            f"valid_a exact_match {valid_a_f:.4f} is below lockbox threshold {min_valid_a_accuracy:.4f}"
        )
    return valid_a_f


def build_lockbox_gate(
    *,
    sample_count: int,
    correct_count: int,
    exact_match: float,
    required_accuracy: float,
    valid_a_best_accuracy: float,
    checkpoint_path: str | Path,
    evaluated_at: str,
) -> dict[str, Any]:
    required = required_correct_count(sample_count, required_accuracy)
    status = "SUBMISSION_CANDIDATE_PASS" if int(correct_count) >= required else "SUBMISSION_CANDIDATE_REJECT"
    return {
        "sample_count": int(sample_count),
        "correct_count": int(correct_count),
        "exact_match": float(exact_match),
        "required_accuracy": float(required_accuracy),
        "required_correct_count": int(required),
        "valid_a_best_accuracy": float(valid_a_best_accuracy),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "status": status,
        "evaluated_at": evaluated_at,
    }
