import json

import pytest
import torch

from snu_order.qwen3vl.calibration_stage_pair import (
    permutation_table_fingerprint,
    save_raw_stage_pair_logits,
)
from snu_order.qwen3vl.certify_chunked_inference import certify_chunked_parity


def _artifacts(tmp_path, *, prediction_flip=False):
    torch.manual_seed(8)
    ids = ["a", "b", "c"]
    stage = torch.randn(3, 4, 4)
    pair = torch.randn(3, 6)
    target = torch.tensor([0, 1, 2])
    answer = torch.tensor([[1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4]])
    paths = {}
    for name in ("legacy", "repeat", "chunk"):
        stage_value = stage.clone()
        path = tmp_path / f"{name}.pt"
        save_raw_stage_pair_logits(
            path,
            ids=ids,
            stage_logits=stage_value,
            pair_logits=pair,
            target_perm_idx=target,
            answer=answer,
        )
        paths[name] = path
    if prediction_flip:
        payload = torch.load(paths["chunk"], map_location="cpu", weights_only=False)
        old_prediction = int(payload["raw_prediction"][0])
        new_prediction = (old_prediction + 1) % 24
        payload["raw_fused_scores"][0, new_prediction] = payload["raw_fused_scores"][0].max() + 100
        payload["raw_prediction"][0] = new_prediction
        torch.save(payload, paths["chunk"])
    feature = torch.randn(3, 4, 8)
    for name, chunk_size in (("legacy_feature", None), ("chunk_feature", 1)):
        path = tmp_path / f"{name}.pt"
        torch.save({"ids": ids, "frame_hidden": feature, "frame_chunk_size": chunk_size}, path)
        paths[name] = path
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "tune_split": "valid_a",
                "pair_weight": 1.0,
                "stage_temperature": 1.5,
                "pair_temperature": 0.8,
                "permutation_table_fingerprint": permutation_table_fingerprint(),
            }
        ),
        encoding="utf-8",
    )
    paths["calibration"] = calibration
    return paths


def test_chunk_certification_accepts_exact_full_parity(tmp_path):
    paths = _artifacts(tmp_path)
    report = certify_chunked_parity(
        legacy_raw=paths["legacy"],
        legacy_repeat_raw=paths["repeat"],
        chunked_raw=paths["chunk"],
        legacy_features=paths["legacy_feature"],
        chunked_features=paths["chunk_feature"],
        calibration_path=paths["calibration"],
        output_path=tmp_path / "report.json",
    )
    assert report["status"] == "PASS"
    assert report["raw_prediction_mismatch_count"] == 0


def test_chunk_certification_rejects_prediction_mismatch(tmp_path):
    paths = _artifacts(tmp_path, prediction_flip=True)
    with pytest.raises(RuntimeError, match="parity failed"):
        certify_chunked_parity(
            legacy_raw=paths["legacy"],
            legacy_repeat_raw=paths["repeat"],
            chunked_raw=paths["chunk"],
            legacy_features=paths["legacy_feature"],
            chunked_features=paths["chunk_feature"],
            calibration_path=paths["calibration"],
            output_path=tmp_path / "report.json",
        )
