from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from snu_order.pipeline.make_submission import save_submission
from snu_order.qwen3vl.calibration_stage_pair import (
    checkpoint_calibration_bindings,
    permutation_table_fingerprint,
)
from snu_order.qwen3vl.finalize_hardening import finalize_hardening


def test_final_hardening_requires_bound_calibration_and_exact_819_schema(tmp_path, monkeypatch):
    output = tmp_path / "final"
    output.mkdir()
    reference = tmp_path / "sample_submission.csv"
    ids = [f"sample-{index:04d}" for index in range(819)]
    reference.write_text(
        "Id,Answer\n"
        + "".join(f'{sample_id},"[1,2,3,4]"\n' for sample_id in ids),
        encoding="utf-8",
    )
    submission = output / "submission.csv"
    save_submission(ids, [[1, 2, 3, 4]] * len(ids), submission, reference=reference)

    valid_split = tmp_path / "valid_a.csv"
    valid_split.write_text("Id,Answer\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(f"data:\n  valid_split: {valid_split}\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint_files = {
        "checkpoint_manifest.json": b"manifest",
        "adapter/adapter_model.safetensors": b"adapter",
        "heads.pt": b"heads",
        "prompt_fingerprint.json": json.dumps(
            {
                "anchor_text": "STATE:",
                "anchor_token_ids": [23852, 25],
                "pooling_mode": "anchor_span_mean",
            }
        ).encode(),
        "processor/tokenizer_config.json": b"processor",
        "permutations.json": b"permutations",
    }
    for relative, value in checkpoint_files.items():
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    bindings = checkpoint_calibration_bindings(
        checkpoint, {"data": {"valid_split": str(valid_split)}}
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "format_version": 2,
                "tune_split": "valid_a",
                "permutation_table_fingerprint": permutation_table_fingerprint(),
                "pair_weight": 0.3,
                "stage_temperature": 1.0,
                "pair_temperature": 1.0,
                "artifact_bindings": bindings,
            }
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"selected_candidate": "state_e1"}), encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "frame_chunk_size": 4,
                "peak_vram_bytes": 1,
                "peak_reserved_vram_bytes": 2,
                "end_to_end_wall_clock_sec": 3.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "fake-gpu")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=32 * 1024**3),
    )

    result = finalize_hardening(
        output_dir=output,
        submission_path=submission,
        sample_submission_path=reference,
        decision_path=decision,
        selected_candidate="state_e1",
        checkpoint_path=checkpoint,
        config_path=config,
        calibration_path=calibration,
        inference_profile_path=profile,
    )
    assert result["validator"]["status"] == "PASS"
    assert result["manifest"]["row_count"] == 819
    assert result["manifest"]["schema"] == ["Id", "Answer"]
    assert (output / "checksums.sha256").is_file()
    assert submission.read_bytes().splitlines()[0] == b"Id,Answer"
