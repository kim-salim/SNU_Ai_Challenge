from __future__ import annotations

from pathlib import Path

from snu_order.qwen3vl.lockbox import (
    check_lockbox_unlock,
    check_valid_a_threshold,
    guard_not_lockbox_path,
    required_correct_count,
)


def test_lockbox_requires_unlock_flag():
    try:
        check_lockbox_unlock(False)
    except PermissionError:
        return
    raise AssertionError("missing unlock flag should raise")


def test_valid_a_threshold_blocks_lockbox(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "metrics.json").write_text('{"valid_a_exact_match": 0.32}', encoding="utf-8")
    try:
        check_valid_a_threshold(ckpt, min_valid_a_accuracy=0.33)
    except PermissionError:
        return
    raise AssertionError("valid_a below threshold should block")


def test_correct_count_threshold_uses_ceil():
    assert required_correct_count(1430, 0.30) == 429
    assert required_correct_count(10, 0.31) == 4


def test_trainer_rejects_valid_b_path():
    try:
        guard_not_lockbox_path("data/splits/ab_v1/valid_b_v1.csv", purpose="training")
    except ValueError:
        return
    raise AssertionError("valid_b path should be rejected")
