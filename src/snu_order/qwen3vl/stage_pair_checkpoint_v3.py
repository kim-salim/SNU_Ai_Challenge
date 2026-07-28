from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.config import get_by_path
from snu_order.utils.io import write_json

from .checkpoint import file_sha256, unwrap_model
from .lora_targets import finalize_peft_lora_manifest, lora_target_summary, validate_lora_manifest
from .permutations import PERMS
from .qwen35_27b_port import (
    ARCHITECTURE_ID,
    EXPECTED_HIDDEN_SIZE,
    SUPPORTED_ARCHITECTURE_IDS,
    quantization_contract,
    validate_qwen35_27b_architecture,
)
from .stage_pair_prompt import (
    StagePairPromptSpec,
    assert_prompt_fingerprint_match,
    build_prompt_fingerprint,
)


FORMAT_VERSION = 3


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _port_runtime_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = (
        "architecture",
        "backbone.base_model_path",
        "backbone.revision",
        "backbone.model_type",
        "backbone.hidden_size",
        "backbone.torch_dtype",
        "backbone.device_map",
        "quantization",
        "prompt",
        "pooling.mode",
        "data.image_policy",
        "data.text_policy",
        "data.split_contract",
        "lora",
        "vision_merger_lora.enabled",
        "model",
        "score",
        "retention",
    )
    return {path: get_by_path(cfg, path, None) for path in paths}


def _assert_port_runtime_contract(saved: dict[str, Any], current: dict[str, Any]) -> None:
    expected = _port_runtime_contract(saved)
    observed = _port_runtime_contract(current)
    mismatches = {
        key: {"checkpoint": expected[key], "runtime": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    if set(mismatches) == {"retention"}:
        saved_retention = expected["retention"]
        current_retention = observed["retention"]
        shared_warmup_fork = bool(get_by_path(saved, "train.fork_after_head_warmup", False))
        explicit_override = bool(
            get_by_path(current, "checkpoint.allow_shared_warmup_retention_fork", False)
        )
        component_safe_target = (
            isinstance(current_retention, dict)
            and current_retention.get("enabled") is True
            and current_retention.get("mode") == "component_safe_v3"
            and current_retention.get("soft_kl") is False
        )
        retention_disabled_at_fork = saved_retention in (None, {"enabled": False})
        if shared_warmup_fork and explicit_override and component_safe_target and retention_disabled_at_fork:
            mismatches = {}
    if mismatches:
        raise RuntimeError(f"27B checkpoint/runtime contract mismatch: {json.dumps(mismatches, sort_keys=True)}")


def _file_entries(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": str(path.relative_to(root)),
            "byte_size": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if path.name != "checkpoint_manifest.json"
    ]


def _atomic_replace(staging: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    moved = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved = True
        os.replace(staging, destination)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if moved and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise


def _write_heads(path: Path, model: Any, metrics: dict[str, Any], *, architecture_id: str) -> None:
    if not hasattr(model, "frame_projector"):
        raise RuntimeError("27B v3 checkpoint requires frame_projector")
    torch.save(
        {
            "architecture": architecture_id,
            "frame_projector": model.frame_projector.state_dict(),
            "set_encoder": model.set_encoder.state_dict(),
            "stage_head": model.stage_head.state_dict(),
            "pair_head": None if model.pair_head is None else model.pair_head.state_dict(),
            "hidden_size": int(model.hidden_size),
            "model_dim": int(model.model_dim),
            "pooling_mode": str(model.pooling_mode),
            "metrics": metrics,
        },
        path,
    )


def _processor_manifest_sha(root: Path) -> str:
    values = [
        {
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
        }
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    ]
    if not values:
        raise RuntimeError("Processor/tokenizer manifest is empty")
    return _canonical_sha256(values)


def build_v3_binding(
    model: Any,
    cfg: dict[str, Any],
    root: Path,
    prompt_fingerprint: dict[str, Any],
    *,
    calibration_sha256: str | None,
) -> dict[str, Any]:
    backbone = getattr(model, "backbone", None)
    if backbone is None or getattr(backbone, "config", None) is None:
        raise RuntimeError("27B v3 checkpoint requires a live backbone config")
    architecture = validate_qwen35_27b_architecture(backbone.config)
    base_config = (
        backbone.config.to_dict()
        if callable(getattr(backbone.config, "to_dict", None))
        else vars(backbone.config)
    )
    adapter_dir = root / "adapter"
    adapter_weights = [
        candidate
        for candidate in (adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin")
        if candidate.is_file()
    ]
    if len(adapter_weights) != 1:
        raise RuntimeError("27B v3 checkpoint requires exactly one adapter weight file")
    return {
        "architecture": str(get_by_path(cfg, "architecture.id")),
        "base_model_path": str(get_by_path(cfg, "backbone.base_model_path")),
        "base_model_revision": get_by_path(cfg, "backbone.revision", None),
        "backbone_config_sha256": _canonical_sha256(base_config),
        "hidden_size": int(architecture["hidden_size"]),
        "language_layers": int(architecture["num_hidden_layers"]),
        "block_counts": dict(architecture["layer_type_counts"]),
        "quantization": quantization_contract(cfg),
        "processor_tokenizer_manifest_sha256": _processor_manifest_sha(root / "processor"),
        "prompt_fingerprint_sha256": file_sha256(root / "prompt_fingerprint.json"),
        "prompt_sha256": str(prompt_fingerprint["rendered_prompt_sha256"]),
        "anchor_token_ids": list(prompt_fingerprint["anchor_token_ids"]),
        "image_policy": str(get_by_path(cfg, "data.image_policy")),
        "text_policy": str(get_by_path(cfg, "data.text_policy")),
        "permutation_table_sha256": file_sha256(root / "permutations.json"),
        "adapter_sha256": file_sha256(adapter_weights[0]),
        "heads_sha256": file_sha256(root / "heads.pt"),
        "calibration_sha256": calibration_sha256,
        "source_git_head": _git_value("rev-parse", "HEAD"),
        "source_git_tree": _git_value("rev-parse", "HEAD^{tree}"),
    }


def save_stage_pair_checkpoint_v3(
    path: str | Path,
    model: Any,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    *,
    processor: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
    minimal: bool = False,
    prompt_fingerprint: dict[str, Any] | None = None,
    training_progress: dict[str, Any] | None = None,
) -> None:
    from .stage_pair_checkpoint import (
        _package_versions,
        _requires_adapter,
        _write_exact_adapter_config,
    )

    raw = unwrap_model(model)
    architecture_id = str(get_by_path(cfg, "architecture.id", ""))
    if architecture_id not in SUPPORTED_ARCHITECTURE_IDS:
        raise RuntimeError("v3 checkpoint is reserved for the 27B Champion port")
    if not str(get_by_path(cfg, "backbone.revision", "") or ""):
        raise RuntimeError("v3 checkpoint requires a pinned 27B base revision")
    if int(raw.hidden_size) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("v3 checkpoint hidden width must be 5120")
    if processor is None or not hasattr(processor, "save_pretrained"):
        raise RuntimeError("v3 checkpoint requires a save_pretrained-capable processor")
    if prompt_fingerprint is None:
        raise RuntimeError("v3 checkpoint requires a live prompt fingerprint")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_heads(staging / "heads.pt", raw, metrics, architecture_id=architecture_id)
        backbone = raw.backbone
        manifest = getattr(backbone, "_stage_pair_lora_target_manifest", None)
        if not _requires_adapter(cfg) or not isinstance(manifest, list):
            raise RuntimeError("v3 checkpoint requires one finalized LoRA adapter manifest")
        validate_lora_manifest(
            manifest,
            vision_merger_enabled=False,
            expected_language_layers=64,
            expected_full_attention_layers=16,
            expected_linear_attention_layers=48,
        )
        backbone.save_pretrained(staging / "adapter", safe_serialization=True)
        _write_exact_adapter_config(staging / "adapter", manifest)
        write_json(staging / "lora_target_manifest.json", manifest)
        processor.save_pretrained(staging / "processor")
        write_json(staging / "prompt_fingerprint.json", prompt_fingerprint)
        write_json(staging / "config.json", cfg)
        write_json(staging / "metrics.json", metrics)
        write_json(staging / "permutations.json", {"perms": [list(perm) for perm in PERMS]})
        if extra is not None:
            write_json(staging / "extra.json", extra)
        if not minimal:
            state: dict[str, Any] = {}
            if optimizer is not None:
                state["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                state["scheduler"] = scheduler.state_dict()
            if training_progress is not None:
                state["training_progress"] = dict(training_progress)
            if state:
                torch.save(state, staging / "training_state.pt")
        calibration_sha = None if extra is None else extra.get("calibration_sha256")
        binding = build_v3_binding(raw, cfg, staging, prompt_fingerprint, calibration_sha256=calibration_sha)
        checkpoint_manifest = {
            "checkpoint_format_version": FORMAT_VERSION,
            "experiment_id": str(get_by_path(cfg, "experiment.id")),
            "runtime_contract": _port_runtime_contract(cfg),
            "binding": binding,
            "lora_groups": lora_target_summary(manifest),
            "package_versions": _package_versions(),
            "files": _file_entries(staging),
        }
        write_json(staging / "checkpoint_manifest.json", checkpoint_manifest)
        verify_stage_pair_checkpoint_v3(staging, runtime_cfg=cfg, processor=processor, model=raw)
        _atomic_replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_stage_pair_checkpoint_v3(
    path: str | Path,
    *,
    runtime_cfg: dict[str, Any] | None = None,
    processor: Any | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    from .stage_pair_checkpoint import _validate_adapter

    root = Path(path)
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("v3 checkpoint manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("checkpoint_format_version", -1)) != FORMAT_VERSION:
        raise RuntimeError("checkpoint is not format v3")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("v3 manifest has no files")
    declared = {str(entry["relative_path"]) for entry in entries}
    for entry in entries:
        file_path = root / str(entry["relative_path"])
        if not file_path.is_file() or int(file_path.stat().st_size) != int(entry["byte_size"]):
            raise RuntimeError(f"v3 manifest file missing/size mismatch: {file_path}")
        if file_sha256(file_path) != str(entry["sha256"]):
            raise RuntimeError(f"v3 manifest checksum mismatch: {file_path}")
    required = {
        "heads.pt",
        "config.json",
        "metrics.json",
        "permutations.json",
        "prompt_fingerprint.json",
        "lora_target_manifest.json",
        "adapter/adapter_config.json",
    }
    if not required <= declared:
        raise RuntimeError(f"v3 manifest is missing required checksums: {sorted(required - declared)}")
    saved_cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if int(get_by_path(saved_cfg, "checkpoint.format_version", -1)) != FORMAT_VERSION:
        raise RuntimeError("saved config is not checkpoint v3")
    saved_architecture = str(get_by_path(saved_cfg, "architecture.id", ""))
    if saved_architecture not in SUPPORTED_ARCHITECTURE_IDS:
        raise RuntimeError("saved architecture id mismatch")
    if runtime_cfg is not None:
        _assert_port_runtime_contract(saved_cfg, runtime_cfg)
    permutations = json.loads((root / "permutations.json").read_text(encoding="utf-8"))
    if permutations != {"perms": [list(perm) for perm in PERMS]}:
        raise RuntimeError("v3 permutation mapping mismatch")
    lora_manifest = json.loads((root / "lora_target_manifest.json").read_text(encoding="utf-8"))
    validate_lora_manifest(
        lora_manifest,
        vision_merger_enabled=False,
        expected_language_layers=64,
        expected_full_attention_layers=16,
        expected_linear_attention_layers=48,
    )
    _validate_adapter(root / "adapter", lora_manifest)
    if processor is not None:
        saved_fingerprint = json.loads((root / "prompt_fingerprint.json").read_text(encoding="utf-8"))
        current = build_prompt_fingerprint(runtime_cfg or saved_cfg, processor, format_version=2)
        assert_prompt_fingerprint_match(saved_fingerprint, current)
    binding = manifest.get("binding")
    if not isinstance(binding, dict) or binding.get("architecture") != saved_architecture:
        raise RuntimeError("v3 architecture binding is missing")
    if int(binding.get("hidden_size", -1)) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("v3 hidden-size binding mismatch")
    if binding.get("quantization") != quantization_contract(saved_cfg):
        raise RuntimeError("v3 quantization binding mismatch")
    if file_sha256(root / "heads.pt") != binding.get("heads_sha256"):
        raise RuntimeError("v3 heads binding mismatch")
    adapter_weights = [
        candidate
        for candidate in (
            root / "adapter" / "adapter_model.safetensors",
            root / "adapter" / "adapter_model.bin",
        )
        if candidate.is_file()
    ]
    if len(adapter_weights) != 1 or file_sha256(adapter_weights[0]) != binding.get("adapter_sha256"):
        raise RuntimeError("v3 adapter binding mismatch")
    if file_sha256(root / "prompt_fingerprint.json") != binding.get("prompt_fingerprint_sha256"):
        raise RuntimeError("v3 prompt binding mismatch")
    if file_sha256(root / "permutations.json") != binding.get("permutation_table_sha256"):
        raise RuntimeError("v3 permutation binding mismatch")
    if _processor_manifest_sha(root / "processor") != binding.get("processor_tokenizer_manifest_sha256"):
        raise RuntimeError("v3 processor/tokenizer binding mismatch")
    if model is not None:
        architecture = validate_qwen35_27b_architecture(model.backbone.config)
        if int(architecture["hidden_size"]) != int(binding["hidden_size"]):
            raise RuntimeError("runtime backbone/config binding mismatch")
        base_config = (
            model.backbone.config.to_dict()
            if callable(getattr(model.backbone.config, "to_dict", None))
            else vars(model.backbone.config)
        )
        if _canonical_sha256(base_config) != binding.get("backbone_config_sha256"):
            raise RuntimeError("runtime backbone config SHA mismatch")
    return manifest


def load_stage_pair_checkpoint_v3(
    path: str | Path,
    model: Any,
    *,
    is_trainable: bool,
    cfg: dict[str, Any],
    processor: Any,
) -> tuple[Any, dict[str, Any]]:
    from .stage_pair_checkpoint import _load_adapter

    raw = unwrap_model(model)
    verify_stage_pair_checkpoint_v3(path, runtime_cfg=cfg, processor=processor, model=raw)
    root = Path(path)
    adapter_name = _load_adapter(raw, root / "adapter", is_trainable=is_trainable)
    lora_manifest = json.loads((root / "lora_target_manifest.json").read_text(encoding="utf-8"))
    finalized, summary = finalize_peft_lora_manifest(
        raw.backbone,
        lora_manifest,
        adapter_name=adapter_name,
        require_trainable=is_trainable,
    )
    raw.backbone._stage_pair_lora_target_manifest = finalized
    raw.backbone._stage_pair_lora_target_summary = summary
    payload = torch.load(root / "heads.pt", map_location="cpu", weights_only=False)
    required = {
        "architecture",
        "frame_projector",
        "set_encoder",
        "stage_head",
        "pair_head",
        "hidden_size",
        "model_dim",
        "pooling_mode",
        "metrics",
    }
    if set(payload) != required:
        raise RuntimeError("v3 heads schema mismatch")
    if payload["architecture"] != str(get_by_path(cfg, "architecture.id")) or int(payload["hidden_size"]) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("v3 heads identity mismatch")
    raw.frame_projector.load_state_dict(payload["frame_projector"], strict=True)
    raw.set_encoder.load_state_dict(payload["set_encoder"], strict=True)
    raw.stage_head.load_state_dict(payload["stage_head"], strict=True)
    if raw.pair_head is None or payload["pair_head"] is None:
        raise RuntimeError("v3 pair-head architecture mismatch")
    raw.pair_head.load_state_dict(payload["pair_head"], strict=True)
    return raw, payload
