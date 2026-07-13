from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from snu_order.qwen3vl.checkpoint import file_sha256
from snu_order.qwen3vl.modeling_stage_pair import Qwen3VLStagePairModel
from snu_order.qwen3vl.stage_pair_checkpoint import (
    load_stage_pair_checkpoint,
    save_stage_pair_checkpoint,
    verify_stage_pair_checkpoint_files,
)


class FakeProcessor:
    def __init__(self, fail=False):
        self.fail = fail

    def save_pretrained(self, path):
        if self.fail:
            raise RuntimeError("processor save failed")
        output = Path(path)
        output.mkdir(parents=True)
        (output / "processor_config.json").write_text("{}", encoding="utf-8")


class FailingBackbone(nn.Module):
    def save_pretrained(self, path, **kwargs):
        raise RuntimeError("adapter save failed")


def _cfg(lora=False):
    return {
        "experiment": {"id": "fixture"},
        "backbone": {
            "base_model_path": "Qwen/Qwen3.5-9B",
            "revision": "fixed",
            "hidden_size": 8,
        },
        "prompt": {"enable_thinking": None},
        "pooling": {"mode": "last_non_padding"},
        "lora": {"enabled": lora},
        "vision_merger_lora": {"enabled": False},
        "model": {
            "model_dim": 16,
            "set_layers": 1,
            "set_heads": 4,
            "set_ffn_dim": 32,
            "use_set_encoder": True,
            "use_pairwise": True,
        },
        "score": {"stage_weight": 1.0, "pair_weight": 0.3},
        "checkpoint": {"format_version": 2},
    }


def _model(set_layers=1):
    return Qwen3VLStagePairModel(
        None,
        hidden_size=8,
        model_dim=16,
        set_layers=set_layers,
        set_heads=4,
        set_ffn_dim=32,
    )


def _refresh_manifest(root: Path):
    manifest_path = root / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        {
            "relative_path": str(path.relative_to(root)),
            "byte_size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checkpoint_manifest.json"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_v2_mock_checkpoint_round_trip(monkeypatch, tmp_path):
    fingerprint = {"input_ids_sha256": "stable"}
    monkeypatch.setattr(
        "snu_order.qwen3vl.stage_pair_checkpoint.build_prompt_fingerprint",
        lambda cfg, processor: fingerprint,
    )
    original = _model()
    with torch.no_grad():
        for parameter in original.stage_head.parameters():
            parameter.add_(1.0)
    checkpoint = tmp_path / "valid"
    save_stage_pair_checkpoint(
        checkpoint,
        original,
        _cfg(False),
        {"exact_match": 0.5},
        processor=FakeProcessor(),
        minimal=True,
        prompt_fingerprint=fingerprint,
    )
    fresh = _model()
    load_stage_pair_checkpoint(
        checkpoint,
        fresh,
        strict=True,
        cfg=_cfg(False),
        processor=FakeProcessor(),
    )
    for left, right in zip(original.stage_head.parameters(), fresh.stage_head.parameters(), strict=True):
        assert torch.allclose(left, right)


def test_adapter_and_processor_save_failures_propagate(tmp_path):
    model = _model()
    model.backbone = FailingBackbone()
    with pytest.raises(RuntimeError, match="adapter save failed"):
        save_stage_pair_checkpoint(tmp_path / "adapter-fail", model, {}, {}, minimal=True)
    with pytest.raises(RuntimeError, match="processor save failed"):
        save_stage_pair_checkpoint(
            tmp_path / "processor-fail",
            _model(),
            {},
            {},
            processor=FakeProcessor(fail=True),
            minimal=True,
        )


def test_corrupt_checksum_and_missing_heads_fail(monkeypatch, tmp_path):
    fingerprint = {"input_ids_sha256": "stable"}
    monkeypatch.setattr(
        "snu_order.qwen3vl.stage_pair_checkpoint.build_prompt_fingerprint",
        lambda cfg, processor: fingerprint,
    )
    checkpoint = tmp_path / "checksum"
    save_stage_pair_checkpoint(
        checkpoint,
        _model(),
        _cfg(False),
        {},
        processor=FakeProcessor(),
        minimal=True,
        prompt_fingerprint=fingerprint,
    )
    (checkpoint / "metrics.json").write_text("corrupt", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch|size mismatch"):
        verify_stage_pair_checkpoint_files(checkpoint, runtime_cfg=_cfg(False), processor=FakeProcessor())

    checkpoint = tmp_path / "missing-heads"
    save_stage_pair_checkpoint(
        checkpoint,
        _model(),
        _cfg(False),
        {},
        processor=FakeProcessor(),
        minimal=True,
        prompt_fingerprint=fingerprint,
    )
    (checkpoint / "heads.pt").unlink()
    with pytest.raises(RuntimeError, match="Manifest file is missing"):
        verify_stage_pair_checkpoint_files(checkpoint, runtime_cfg=_cfg(False), processor=FakeProcessor())


def test_required_artifact_without_checksum_entry_fails(monkeypatch, tmp_path):
    fingerprint = {"input_ids_sha256": "stable"}
    monkeypatch.setattr(
        "snu_order.qwen3vl.stage_pair_checkpoint.build_prompt_fingerprint",
        lambda cfg, processor: fingerprint,
    )
    checkpoint = tmp_path / "missing-checksum-entry"
    save_stage_pair_checkpoint(
        checkpoint,
        _model(),
        _cfg(False),
        {},
        processor=FakeProcessor(),
        minimal=True,
        prompt_fingerprint=fingerprint,
    )
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["relative_path"] != "heads.pt"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lacks a checksum entry"):
        verify_stage_pair_checkpoint_files(checkpoint, runtime_cfg=_cfg(False), processor=FakeProcessor())


def test_missing_adapter_and_strict_state_mismatch_fail(monkeypatch, tmp_path):
    fingerprint = {"input_ids_sha256": "stable"}
    monkeypatch.setattr(
        "snu_order.qwen3vl.stage_pair_checkpoint.build_prompt_fingerprint",
        lambda cfg, processor: fingerprint,
    )
    checkpoint = tmp_path / "missing-adapter"
    save_stage_pair_checkpoint(
        checkpoint,
        _model(),
        _cfg(False),
        {},
        processor=FakeProcessor(),
        minimal=True,
        prompt_fingerprint=fingerprint,
    )
    saved_cfg = _cfg(True)
    (checkpoint / "config.json").write_text(json.dumps(saved_cfg), encoding="utf-8")
    (checkpoint / "lora_target_manifest.json").write_text("[]", encoding="utf-8")
    _refresh_manifest(checkpoint)
    monkeypatch.setattr("snu_order.qwen3vl.stage_pair_checkpoint.validate_lora_manifest", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Missing adapter config"):
        verify_stage_pair_checkpoint_files(checkpoint, runtime_cfg=saved_cfg, processor=FakeProcessor())

    checkpoint = tmp_path / "strict-mismatch"
    save_stage_pair_checkpoint(
        checkpoint,
        _model(set_layers=1),
        _cfg(False),
        {},
        processor=FakeProcessor(),
        minimal=True,
        prompt_fingerprint=fingerprint,
    )
    with pytest.raises(RuntimeError, match="state_dict|Missing key|Unexpected key"):
        load_stage_pair_checkpoint(
            checkpoint,
            _model(set_layers=2),
            strict=True,
            cfg=_cfg(False),
            processor=FakeProcessor(),
        )


def test_prompt_fingerprint_mismatch_fails(monkeypatch, tmp_path):
    saved_fingerprint = {"input_ids_sha256": "saved"}
    monkeypatch.setattr(
        "snu_order.qwen3vl.stage_pair_checkpoint.build_prompt_fingerprint",
        lambda cfg, processor: saved_fingerprint,
    )
    checkpoint = tmp_path / "fingerprint"
    save_stage_pair_checkpoint(
        checkpoint,
        _model(),
        _cfg(False),
        {},
        processor=FakeProcessor(),
        minimal=True,
        prompt_fingerprint=saved_fingerprint,
    )
    monkeypatch.setattr(
        "snu_order.qwen3vl.stage_pair_checkpoint.build_prompt_fingerprint",
        lambda cfg, processor: {"input_ids_sha256": "runtime"},
    )
    with pytest.raises(RuntimeError, match="Prompt fingerprint mismatch"):
        verify_stage_pair_checkpoint_files(checkpoint, runtime_cfg=_cfg(False), processor=FakeProcessor())
