from __future__ import annotations

import json

import pytest
import torch

from snu_order.qwen3vl.calibration_stage_pair import (
    CalibrationParameters,
    _comparison,
    load_calibration,
    permutation_table_fingerprint,
    run_calibration,
    search_calibration_grid,
)
from snu_order.qwen3vl.permutations import answer_to_perm_index
from snu_order.qwen3vl.stage_pair_scorer import pair_targets_from_answer


def _payload():
    answer = torch.tensor([[2, 1, 3, 4]], dtype=torch.long)
    pair_targets = pair_targets_from_answer(answer)
    pair_logits = torch.where(pair_targets > 0, 8.0, -8.0)
    return {
        "format_version": 1,
        "ids": ["sample"],
        "stage_logits": torch.zeros((1, 4, 4), dtype=torch.float32),
        "pair_logits": pair_logits,
        "target_perm_idx": torch.tensor([answer_to_perm_index(answer[0].tolist())]),
        "answer": answer,
        "permutation_table_fingerprint": permutation_table_fingerprint(),
    }


def test_calibration_grid_recovers_known_pair_weight_and_tie_breaks_deterministically():
    selected, rows = search_calibration_grid(
        _payload(),
        pair_weights=[0.0, 0.3, 0.5],
        stage_temperatures=[0.8, 1.0],
        pair_temperatures=[0.8, 1.0],
    )
    assert selected == CalibrationParameters(0.3, 1.0, 1.0)
    selected_again, _ = search_calibration_grid(
        _payload(),
        pair_weights=[0.5, 0.3, 0.0],
        stage_temperatures=[1.0, 0.8],
        pair_temperatures=[1.0, 0.8],
    )
    assert selected_again == selected
    assert len(rows) == 12


def test_valid_b_tuning_is_rejected():
    with pytest.raises(RuntimeError, match="restricted to valid-A"):
        run_calibration(_payload(), "/tmp/unused-calibration", tune_split="valid_b")


def test_raw_fixed_broken_counts_are_exact():
    raw = torch.tensor([[4.0, 1.0], [4.0, 1.0], [1.0, 4.0], [1.0, 4.0]])
    calibrated = torch.tensor([[4.0, 1.0], [1.0, 4.0], [4.0, 1.0], [1.0, 4.0]])
    targets = torch.tensor([0, 1, 0, 1])
    assert _comparison(raw, calibrated, targets) == {
        "broken": 0,
        "fixed": 2,
        "unchanged_correct": 2,
        "unchanged_wrong": 0,
    }


def test_calibration_serialization_round_trip(tmp_path):
    result = run_calibration(
        _payload(),
        tmp_path,
        tune_split="valid_a",
        pair_weights=[0.0, 0.3],
        stage_temperatures=[1.0],
        pair_temperatures=[1.0],
    )
    loaded = load_calibration(tmp_path / "calibration.json")
    serialized = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert loaded.as_dict() == {
        key: serialized[key] for key in ("pair_weight", "stage_temperature", "pair_temperature")
    }
    assert result["calibration"] == serialized
    assert loaded == CalibrationParameters(0.3, 1.0, 1.0)
    for filename in (
        "calibration_grid.csv",
        "calibration.json",
        "calibrated_metrics.json",
        "calibrated_valid_predictions.csv",
        "calibrated_wrong_cases.csv",
        "raw_vs_calibrated_comparison.json",
    ):
        assert (tmp_path / filename).is_file()
