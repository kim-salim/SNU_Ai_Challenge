from __future__ import annotations

import importlib.metadata
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
from .stage_pair_prompt import (
    ANCHOR_POOLING_MODE,
    StagePairPromptSpec,
    assert_prompt_fingerprint_match,
    build_prompt_fingerprint,
)


FORMAT_VERSION = 2


def _adapter_names(backbone: Any, fallback: str = "default") -> list[str]:
    active = getattr(backbone, "active_adapters", None)
    if isinstance(active, (list, tuple)) and active:
        return [str(value) for value in active]
    active_one = getattr(backbone, "active_adapter", None)
    if isinstance(active_one, str) and active_one:
        return [active_one]
    return [fallback]


def _requires_adapter(cfg: dict[str, Any]) -> bool:
    return bool(get_by_path(cfg, "lora.enabled", False))


def _package_versions() -> dict[str, str]:
    versions = {"torch": torch.__version__}
    for name in ("transformers", "peft", "bitsandbytes", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _runtime_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = (
        "experiment.id",
        "backbone.base_model_path",
        "backbone.revision",
        "backbone.hidden_size",
        "quantization.enabled",
        "quantization.bits",
        "quantization.quant_type",
        "quantization.double_quant",
        "quantization.compute_dtype",
        "prompt.enable_thinking",
        "prompt.add_generation_prompt",
        "prompt.anchor_text",
        "prompt.strict_template",
        "pooling.mode",
        "lora.enabled",
        "lora.full_attention",
        "lora.linear_attention",
        "vision_merger_lora",
        "model.model_dim",
        "model.set_layers",
        "model.set_heads",
        "model.set_ffn_dim",
        "model.use_set_encoder",
        "model.use_pairwise",
        "score.stage_weight",
        "score.pair_weight",
    )
    return {path: get_by_path(cfg, path, None) for path in paths}


def _assert_runtime_contract(saved_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> None:
    saved = _runtime_contract(saved_cfg)
    current = _runtime_contract(runtime_cfg)
    mismatches = {
        key: {"checkpoint": saved[key], "runtime": current[key]}
        for key in saved
        if saved[key] != current[key]
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint/runtime config mismatch: {json.dumps(mismatches, sort_keys=True)}")


def _adapter_weight_file(adapter_dir: Path) -> Path:
    candidates = [
        path
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (path := adapter_dir / name).is_file() and path.stat().st_size > 0
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one adapter weight file in {adapter_dir}, found {candidates}")
    return candidates[0]


def _adapter_weight_keys(adapter_dir: Path) -> set[str]:
    weight_file = _adapter_weight_file(adapter_dir)
    if weight_file.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            return set(handle.keys())
    payload = torch.load(weight_file, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Adapter payload is not a state dict: {weight_file}")
    return {str(key) for key in payload}


def _write_exact_adapter_config(adapter_dir: Path, manifest: list[dict[str, Any]]) -> None:
    path = adapter_dir / "adapter_config.json"
    if not path.is_file():
        raise RuntimeError(f"PEFT did not create adapter_config.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_modules"] = [str(entry["module_name"]) for entry in manifest]
    payload["rank_pattern"] = {str(entry["module_name"]): int(entry["rank"]) for entry in manifest}
    payload["alpha_pattern"] = {str(entry["module_name"]): int(entry["alpha"]) for entry in manifest}
    write_json(path, payload)


def _validate_adapter(adapter_dir: Path, manifest: list[dict[str, Any]]) -> None:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file() or config_path.stat().st_size <= 0:
        raise RuntimeError(f"Missing adapter config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_names = [str(entry["module_name"]) for entry in manifest]
    if set(config.get("target_modules", [])) != set(expected_names) or len(config.get("target_modules", [])) != len(
        expected_names
    ):
        raise RuntimeError("adapter_config target_modules differ from the exact LoRA manifest")
    for key, entry_key in (("rank_pattern", "rank"), ("alpha_pattern", "alpha")):
        expected = {str(entry["module_name"]): int(entry[entry_key]) for entry in manifest}
        if config.get(key) != expected:
            raise RuntimeError(f"adapter_config {key} differs from the exact LoRA manifest")
    weight_keys = _adapter_weight_keys(adapter_dir)
    matched_keys: set[str] = set()
    for entry in manifest:
        raw_name = str(entry["module_name"])
        for side in ("lora_A", "lora_B"):
            matches = [key for key in weight_keys if raw_name in key and f".{side}." in key]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {side} tensor for {raw_name}, found {matches}")
            matched_keys.add(matches[0])
    if matched_keys != weight_keys:
        raise RuntimeError(f"Unexpected adapter tensors: {sorted(weight_keys - matched_keys)}")


def _write_heads(path: Path, model: Any, metrics: dict[str, Any]) -> None:
    torch.save(
        {
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


def _manifest_files(root: Path) -> list[dict[str, Any]]:
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


def save_stage_pair_checkpoint_v2(
    path: str | Path,
    model: Any,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    *,
    processor: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
    minimal: bool = False,
    prompt_fingerprint: dict[str, Any] | None = None,
) -> None:
    model = unwrap_model(model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_heads(staging / "heads.pt", model, metrics)
        backbone = getattr(model, "backbone", None)
        manifest = getattr(backbone, "_stage_pair_lora_target_manifest", None)
        if _requires_adapter(cfg):
            if backbone is None or not hasattr(backbone, "save_pretrained"):
                raise RuntimeError("LoRA checkpoint requires a save_pretrained-capable backbone")
            if not isinstance(manifest, list):
                raise RuntimeError("LoRA checkpoint is missing its finalized target manifest")
            vision_enabled = bool(get_by_path(cfg, "vision_merger_lora.enabled", False))
            validate_lora_manifest(manifest, vision_merger_enabled=vision_enabled)
            backbone.save_pretrained(staging / "adapter", safe_serialization=True)
            _write_exact_adapter_config(staging / "adapter", manifest)
            write_json(staging / "lora_target_manifest.json", manifest)
        if processor is None or not hasattr(processor, "save_pretrained"):
            raise RuntimeError("v2 checkpoint requires a save_pretrained-capable processor")
        processor.save_pretrained(staging / "processor")

        spec = StagePairPromptSpec.from_config(cfg)
        if spec.pooling_mode == ANCHOR_POOLING_MODE and prompt_fingerprint is None:
            raise RuntimeError("Anchor checkpoint requires prompt_fingerprint")
        if prompt_fingerprint is not None:
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
            if state:
                torch.save(state, staging / "training_state.pt")

        checkpoint_manifest = {
            "checkpoint_format_version": FORMAT_VERSION,
            "base_model_revision": str(get_by_path(cfg, "backbone.revision")),
            "experiment_id": str(get_by_path(cfg, "experiment.id")),
            "pooling_mode": spec.pooling_mode,
            "lora_groups": None if not isinstance(manifest, list) else lora_target_summary(manifest),
            "git_commit": _git_commit(),
            "package_versions": _package_versions(),
            "runtime_contract": _runtime_contract(cfg),
            "files": _manifest_files(staging),
        }
        write_json(staging / "checkpoint_manifest.json", checkpoint_manifest)
        verify_stage_pair_checkpoint_files(staging, runtime_cfg=cfg, processor=processor)
        _atomic_replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _save_stage_pair_checkpoint_v1(
    path: str | Path,
    model: Any,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    *,
    processor: Any | None,
    optimizer: Any | None,
    scheduler: Any | None,
    extra: dict[str, Any] | None,
    minimal: bool,
) -> None:
    model = unwrap_model(model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_heads(staging / "heads.pt", model, metrics)
        backbone = getattr(model, "backbone", None)
        if backbone is not None and hasattr(backbone, "save_pretrained"):
            backbone.save_pretrained(staging / "adapter")
        if processor is not None:
            if not hasattr(processor, "save_pretrained"):
                raise RuntimeError("Processor does not implement save_pretrained")
            processor.save_pretrained(staging / "processor")
        write_json(staging / "config.json", cfg)
        write_json(staging / "metrics.json", metrics)
        write_json(staging / "permutations.json", {"perms": [list(perm) for perm in PERMS]})
        if extra is not None:
            write_json(staging / "extra.json", extra)
        if not minimal:
            state = {}
            if optimizer is not None:
                state["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                state["scheduler"] = scheduler.state_dict()
            if state:
                torch.save(state, staging / "training_state.pt")
        _atomic_replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def save_stage_pair_checkpoint(
    path: str | Path,
    model: Any,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    *,
    processor: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
    minimal: bool = False,
    prompt_fingerprint: dict[str, Any] | None = None,
) -> None:
    if int(get_by_path(cfg, "checkpoint.format_version", 1)) == 3:
        from .stage_pair_checkpoint_v3 import save_stage_pair_checkpoint_v3

        save_stage_pair_checkpoint_v3(
            path,
            model,
            cfg,
            metrics,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            extra=extra,
            minimal=minimal,
            prompt_fingerprint=prompt_fingerprint,
        )
        return
    if int(get_by_path(cfg, "checkpoint.format_version", 1)) == FORMAT_VERSION:
        save_stage_pair_checkpoint_v2(
            path,
            model,
            cfg,
            metrics,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            extra=extra,
            minimal=minimal,
            prompt_fingerprint=prompt_fingerprint,
        )
        return
    _save_stage_pair_checkpoint_v1(
        path,
        model,
        cfg,
        metrics,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        extra=extra,
        minimal=minimal,
    )


def verify_stage_pair_checkpoint_files(
    path: str | Path,
    *,
    runtime_cfg: dict[str, Any] | None = None,
    processor: Any | None = None,
) -> dict[str, Any]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"v2 checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_format_version = int(manifest.get("checkpoint_format_version", -1))
    if checkpoint_format_version == 3:
        from .stage_pair_checkpoint_v3 import verify_stage_pair_checkpoint_v3

        return verify_stage_pair_checkpoint_v3(
            root,
            runtime_cfg=runtime_cfg,
            processor=processor,
        )
    if checkpoint_format_version != FORMAT_VERSION:
        raise RuntimeError(f"Unsupported checkpoint format: {manifest.get('checkpoint_format_version')}")
    file_entries = manifest.get("files")
    if not isinstance(file_entries, list) or not file_entries:
        raise RuntimeError("Checkpoint manifest has no file checksum entries")
    declared_paths = [str(entry.get("relative_path", "")) for entry in file_entries]
    if not all(declared_paths) or len(declared_paths) != len(set(declared_paths)):
        raise RuntimeError("Checkpoint manifest contains empty or duplicate file paths")
    for entry in file_entries:
        file_path = root / str(entry["relative_path"])
        if not file_path.is_file():
            raise RuntimeError(f"Manifest file is missing: {file_path}")
        if int(file_path.stat().st_size) != int(entry["byte_size"]):
            raise RuntimeError(f"Manifest size mismatch: {file_path}")
        actual_hash = file_sha256(file_path)
        if actual_hash != str(entry["sha256"]):
            raise RuntimeError(f"Manifest checksum mismatch: {file_path}")
    for required in (
        "heads.pt",
        "config.json",
        "metrics.json",
        "permutations.json",
        "prompt_fingerprint.json",
    ):
        file_path = root / required
        if not file_path.is_file() or file_path.stat().st_size <= 0:
            raise RuntimeError(f"Required checkpoint artifact is missing or empty: {file_path}")
        if required not in declared_paths:
            raise RuntimeError(f"Required checkpoint artifact lacks a checksum entry: {required}")
    processor_files = [path for path in (root / "processor").rglob("*") if path.is_file()]
    if not processor_files:
        raise RuntimeError(f"Checkpoint processor directory is empty: {root / 'processor'}")
    undeclared_processor_files = [
        str(path.relative_to(root)) for path in processor_files if str(path.relative_to(root)) not in declared_paths
    ]
    if undeclared_processor_files:
        raise RuntimeError(f"Processor files lack checksum entries: {undeclared_processor_files}")
    saved_cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if int(get_by_path(saved_cfg, "checkpoint.format_version", -1)) != FORMAT_VERSION:
        raise RuntimeError("Checkpoint config is not v2")
    if _requires_adapter(saved_cfg):
        manifest_file = root / "lora_target_manifest.json"
        if not manifest_file.is_file():
            raise RuntimeError(f"LoRA target manifest is missing: {manifest_file}")
        adapter_config_path = root / "adapter" / "adapter_config.json"
        if not adapter_config_path.is_file() or adapter_config_path.stat().st_size <= 0:
            raise RuntimeError(f"Missing adapter config: {adapter_config_path}")
        adapter_weight_path = _adapter_weight_file(root / "adapter")
        required_adapter_paths = {"lora_target_manifest.json", "adapter/adapter_config.json"}
        required_adapter_paths.add(str(adapter_weight_path.relative_to(root)))
        missing_checksums = sorted(required_adapter_paths - set(declared_paths))
        if missing_checksums:
            raise RuntimeError(f"LoRA artifacts lack checksum entries: {missing_checksums}")
        lora_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        validate_lora_manifest(
            lora_manifest,
            vision_merger_enabled=bool(get_by_path(saved_cfg, "vision_merger_lora.enabled", False)),
        )
        _validate_adapter(root / "adapter", lora_manifest)
    if runtime_cfg is not None:
        _assert_runtime_contract(saved_cfg, runtime_cfg)
    if processor is not None:
        saved_fingerprint = json.loads((root / "prompt_fingerprint.json").read_text(encoding="utf-8"))
        fingerprint_version = int(saved_fingerprint.get("format_version", 1))
        current_fingerprint = build_prompt_fingerprint(
            runtime_cfg or saved_cfg,
            processor,
            format_version=fingerprint_version,
        )
        assert_prompt_fingerprint_match(saved_fingerprint, current_fingerprint)
    return manifest


def _align_adapter_devices(backbone: Any, adapter_name: str) -> None:
    for module in backbone.modules():
        weight = getattr(module, "weight", None)
        device = getattr(weight, "device", None)
        if device is None:
            continue
        for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
            adapters = getattr(module, attr, None)
            if adapters is not None and adapter_name in adapters:
                adapters[adapter_name].to(device)


def _load_adapter(model: Any, adapter_dir: Path, *, is_trainable: bool) -> str:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise RuntimeError("Checkpoint contains an adapter but model.backbone is missing")
    peft_config = getattr(backbone, "peft_config", None)
    if isinstance(peft_config, dict) and peft_config:
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        adapter_name = _adapter_names(backbone)[0]
        state = load_peft_weights(str(adapter_dir), device="cpu")
        result = set_peft_model_state_dict(
            backbone,
            state,
            adapter_name=adapter_name,
            ignore_mismatched_sizes=False,
        )
        missing_lora = [key for key in result.missing_keys if "lora_" in key]
        if missing_lora or result.unexpected_keys:
            raise RuntimeError(
                f"Adapter state mismatch: missing={missing_lora}, unexpected={result.unexpected_keys}"
            )
        if hasattr(backbone, "set_adapter"):
            backbone.set_adapter(adapter_name)
        for module in backbone.modules():
            for attr in ("lora_A", "lora_B"):
                adapters = getattr(module, attr, None)
                if adapters is not None and adapter_name in adapters:
                    for parameter in adapters[adapter_name].parameters():
                        parameter.requires_grad = is_trainable
    else:
        from peft import PeftModel

        model.backbone = PeftModel.from_pretrained(
            backbone,
            str(adapter_dir),
            is_trainable=is_trainable,
        )
        adapter_name = _adapter_names(model.backbone)[0]
    _align_adapter_devices(model.backbone, adapter_name)
    return adapter_name


def _load_v1(path: Path, model: Any, *, strict: bool, is_trainable: bool) -> tuple[Any, dict[str, Any]]:
    adapter_dir = path / "adapter"
    if adapter_dir.exists():
        _load_adapter(model, adapter_dir, is_trainable=is_trainable)
    heads_path = path / "heads.pt"
    if not heads_path.is_file():
        raise FileNotFoundError(f"Stage-pair heads checkpoint not found: {heads_path}")
    payload = torch.load(heads_path, map_location="cpu", weights_only=False)
    model.set_encoder.load_state_dict(payload["set_encoder"], strict=strict)
    model.stage_head.load_state_dict(payload["stage_head"], strict=strict)
    if model.pair_head is not None and payload.get("pair_head") is not None:
        model.pair_head.load_state_dict(payload["pair_head"], strict=strict)
    return model, payload


def load_stage_pair_checkpoint(
    path: str | Path,
    model: Any,
    *,
    strict: bool = True,
    is_trainable: bool = False,
    cfg: dict[str, Any] | None = None,
    processor: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    model = unwrap_model(model)
    root = Path(path)
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.exists():
        if cfg is not None and int(get_by_path(cfg, "checkpoint.format_version", 1)) == FORMAT_VERSION:
            raise RuntimeError("v2 runtime config cannot load a legacy checkpoint without a manifest")
        return _load_v1(root, model, strict=strict, is_trainable=is_trainable)
    manifest_preview = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest_preview.get("checkpoint_format_version", -1)) == 3:
        if not strict or cfg is None or processor is None:
            raise RuntimeError("v3 checkpoint loading requires strict=True, runtime cfg, and processor")
        from .stage_pair_checkpoint_v3 import load_stage_pair_checkpoint_v3

        return load_stage_pair_checkpoint_v3(
            root,
            model,
            is_trainable=is_trainable,
            cfg=cfg,
            processor=processor,
        )
    if not strict:
        raise RuntimeError("v2 checkpoint loading requires strict=True")
    if cfg is None or processor is None:
        raise RuntimeError("v2 checkpoint loading requires runtime cfg and processor for fingerprint verification")
    verify_stage_pair_checkpoint_files(root, runtime_cfg=cfg, processor=processor)
    saved_cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    adapter_name = "default"
    if _requires_adapter(saved_cfg):
        adapter_name = _load_adapter(model, root / "adapter", is_trainable=is_trainable)
        lora_manifest = json.loads((root / "lora_target_manifest.json").read_text(encoding="utf-8"))
        finalized, summary = finalize_peft_lora_manifest(
            model.backbone,
            lora_manifest,
            adapter_name=adapter_name,
            require_trainable=is_trainable,
        )
        model.backbone._stage_pair_lora_target_manifest = finalized
        model.backbone._stage_pair_lora_target_summary = summary
    payload = torch.load(root / "heads.pt", map_location="cpu", weights_only=False)
    required_keys = {"set_encoder", "stage_head", "pair_head", "hidden_size", "model_dim", "pooling_mode", "metrics"}
    if set(payload) != required_keys:
        raise RuntimeError(
            f"heads.pt keys differ from v2 schema: missing={sorted(required_keys - set(payload))}, "
            f"unexpected={sorted(set(payload) - required_keys)}"
        )
    if int(payload["hidden_size"]) != int(model.hidden_size) or int(payload["model_dim"]) != int(model.model_dim):
        raise RuntimeError("Checkpoint head dimensions do not match the runtime model")
    if str(payload["pooling_mode"]) != str(model.pooling_mode):
        raise RuntimeError("Checkpoint pooling mode does not match the runtime model")
    if (payload["pair_head"] is None) != (model.pair_head is None):
        raise RuntimeError("Checkpoint/runtime pair head architecture mismatch")
    model.set_encoder.load_state_dict(payload["set_encoder"], strict=True)
    model.stage_head.load_state_dict(payload["stage_head"], strict=True)
    if model.pair_head is not None:
        model.pair_head.load_state_dict(payload["pair_head"], strict=True)
    return model, payload
