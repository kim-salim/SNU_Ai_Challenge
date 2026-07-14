from __future__ import annotations

import json

import torch

from snu_order.qwen3vl.calibration_stage_pair import run_calibration, save_raw_stage_pair_logits
from snu_order.qwen3vl.compare_e1_e2 import compare_e1_e2
from snu_order.qwen3vl.permutations import answer_to_perm_index
from snu_order.qwen3vl.stage_pair_scorer import pair_targets_from_answer


def _raw_payload(path, count: int = 50):
    answer = torch.tensor([[2, 1, 3, 4]], dtype=torch.long).repeat(count, 1)
    pair_targets = pair_targets_from_answer(answer)
    pair_logits = torch.where(pair_targets > 0, 8.0, -8.0)
    stage_logits = torch.zeros((count, 4, 4), dtype=torch.float32)
    targets = torch.tensor(
        [answer_to_perm_index(row.tolist()) for row in answer], dtype=torch.long
    )
    save_raw_stage_pair_logits(
        path,
        ids=[f"sample-{index}" for index in range(count)],
        stage_logits=stage_logits,
        pair_logits=pair_logits,
        target_perm_idx=targets,
        answer=answer,
    )


def test_e1_e2_comparison_is_deterministic_and_retains_e1_on_tie(tmp_path):
    e1_raw = tmp_path / "e1_raw.pt"
    e2_raw = tmp_path / "e2_raw.pt"
    _raw_payload(e1_raw)
    _raw_payload(e2_raw)
    e1_payload = torch.load(e1_raw, map_location="cpu", weights_only=False)
    e2_payload = torch.load(e2_raw, map_location="cpu", weights_only=False)
    e1_calibration = tmp_path / "e1_calibration"
    e2_calibration = tmp_path / "e2_calibration"
    grid = {
        "pair_weights": [0.0, 0.3],
        "stage_temperatures": [1.0],
        "pair_temperatures": [1.0],
    }
    run_calibration(e1_payload, e1_calibration, tune_split="valid_a", **grid)
    run_calibration(e2_payload, e2_calibration, tune_split="valid_a", **grid)

    verifications = []
    for index in (1, 2):
        path = tmp_path / f"verify-{index}.json"
        path.write_text(
            json.dumps(
                {"status": "ok", "finite_logits": True, "prediction_indices": [1, 2]}
            ),
            encoding="utf-8",
        )
        verifications.append(path)
    gradient = tmp_path / "gradient.json"
    gradient.write_text(
        json.dumps({"status": "PASS", "captured_completed_optimizer_steps": 2}),
        encoding="utf-8",
    )
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({"differences": {}}), encoding="utf-8")
    chunk = tmp_path / "chunk.json"
    chunk.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")

    result = compare_e1_e2(
        e1_raw_path=e1_raw,
        e1_calibration_dir=e1_calibration,
        e2_raw_path=e2_raw,
        e2_calibration_dir=e2_calibration,
        e2_verifications=verifications,
        gradient_health_path=gradient,
        semantic_diff_path=semantic,
        chunk_selection_path=chunk,
        output_dir=tmp_path / "comparison",
    )
    assert result["decision"]["status"] == "PASS"
    assert result["decision"]["selected_candidate"] == "state_e1"
    assert result["decision"]["selected_inference_frame_chunk_size"] == 4
    assert not result["decision"]["strict_predeclared_promotion_passed"]
    for filename in (
        "raw_metrics.json",
        "calibrated_metrics.json",
        "e1_vs_e2_comparison.csv",
        "paired_predictions.csv",
        "decision.json",
        "report.md",
    ):
        assert (tmp_path / "comparison" / filename).is_file()
