from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from snu_order.utils.config import get_by_path


EXPECTED_LANGUAGE_LAYERS = 32
EXPECTED_FULL_ATTENTION_LAYERS = 8
EXPECTED_LINEAR_ATTENTION_LAYERS = 24
FULL_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
LINEAR_ATTENTION_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "out_proj")
VISION_MERGER_SUFFIXES = ("visual.merger.linear_fc1", "visual.merger.linear_fc2")
TEXT_FULL_GROUP = "text_full_attention"
TEXT_LINEAR_GROUP = "text_linear_attention"
VISION_MERGER_GROUP = "vision_merger"


@dataclass(frozen=True)
class LoraTargetSpec:
    module_name: str
    module_class: str
    group: str
    layer_index: int | None
    layer_type: str
    projection_name: str
    rank: int
    alpha: int
    dropout: float
    parameter_count: int


def is_structured_lora_config(cfg: dict[str, Any]) -> bool:
    return isinstance(get_by_path(cfg, "lora.full_attention", None), dict) and isinstance(
        get_by_path(cfg, "lora.linear_attention", None), dict
    )


def _text_config(model: nn.Module) -> Any:
    config = getattr(model, "config", None)
    for candidate in (
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
        config,
    ):
        if candidate is not None and getattr(candidate, "layer_types", None) is not None:
            return candidate
    raise RuntimeError("Qwen backbone config does not expose text layer_types")


def _group_config(
    cfg: dict[str, Any],
    path: str,
    expected_modules: tuple[str, ...],
) -> tuple[tuple[str, ...], int, int, float]:
    group_cfg = get_by_path(cfg, path, None)
    if not isinstance(group_cfg, dict):
        raise RuntimeError(f"Missing structured LoRA group: {path}")
    modules = tuple(str(value) for value in group_cfg.get("modules", []))
    if modules != expected_modules:
        raise RuntimeError(f"{path}.modules must be {expected_modules}, got {modules}")
    rank = int(group_cfg.get("rank", 0))
    alpha = int(group_cfg.get("alpha", 0))
    dropout = float(group_cfg.get("dropout", 0.0))
    if rank <= 0 or alpha <= 0:
        raise RuntimeError(f"{path} rank and alpha must be positive")
    return modules, rank, alpha, dropout


def _language_layer_container(model: nn.Module, layer_types: list[str]) -> tuple[str, nn.ModuleList]:
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) != len(layer_types):
            continue
        if "language_model" not in name or not name.endswith(".layers"):
            continue
        candidates.append((name, module))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one language backbone ModuleList ending in '.layers'; "
            f"found {[name for name, _ in candidates]}"
        )
    return candidates[0]


def _parameter_count(module: nn.Module, rank: int, module_name: str) -> int:
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if in_features is None or out_features is None:
        raise RuntimeError(f"LoRA target is not a linear projection: {module_name}")
    return int(rank) * (int(in_features) + int(out_features))


def _raise_plan_error(
    message: str,
    entries: list[LoraTargetSpec],
    *,
    missing: list[str] | None = None,
    unexpected_vision: list[str] | None = None,
) -> None:
    counts = Counter(entry.group for entry in entries)
    projection_counts = Counter(entry.projection_name for entry in entries)
    diagnostic = {
        "message": message,
        "actual_matches": [
            {"module_path": entry.module_name, "module_class": entry.module_class, "group": entry.group}
            for entry in entries
        ],
        "group_counts": dict(counts),
        "projection_counts": dict(projection_counts),
        "missing_expected_group": missing or [],
        "unexpected_vision_match": unexpected_vision or [],
    }
    raise RuntimeError(f"Qwen3.5 LoRA target planning failed:\n{json.dumps(diagnostic, indent=2, sort_keys=True)}")


def discover_qwen35_lora_targets(model: nn.Module, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    full_modules, full_rank, full_alpha, full_dropout = _group_config(
        cfg, "lora.full_attention", FULL_ATTENTION_PROJECTIONS
    )
    linear_modules, linear_rank, linear_alpha, linear_dropout = _group_config(
        cfg, "lora.linear_attention", LINEAR_ATTENTION_PROJECTIONS
    )
    if full_dropout != linear_dropout:
        raise RuntimeError("A single PEFT adapter requires identical LoRA dropout across text groups")

    text_config = _text_config(model)
    layer_types = [str(value) for value in getattr(text_config, "layer_types", [])]
    configured_layers = int(getattr(text_config, "num_hidden_layers", len(layer_types)))
    type_counts = Counter(layer_types)
    expected_type_counts = Counter(
        {"full_attention": EXPECTED_FULL_ATTENTION_LAYERS, "linear_attention": EXPECTED_LINEAR_ATTENTION_LAYERS}
    )
    if configured_layers != EXPECTED_LANGUAGE_LAYERS or len(layer_types) != EXPECTED_LANGUAGE_LAYERS:
        _raise_plan_error(
            f"Expected 32 language layers, got num_hidden_layers={configured_layers}, layer_types={len(layer_types)}",
            [],
            missing=["language_layers"],
        )
    if type_counts != expected_type_counts:
        _raise_plan_error(
            f"Expected layer type distribution {dict(expected_type_counts)}, got {dict(type_counts)}",
            [],
            missing=[name for name, count in expected_type_counts.items() if type_counts[name] != count],
        )

    layer_prefix, layers = _language_layer_container(model, layer_types)
    named_modules = dict(model.named_modules())
    entries: list[LoraTargetSpec] = []
    missing: list[str] = []
    for layer_index, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            branch_name = "self_attn"
            projections = full_modules
            group = TEXT_FULL_GROUP
            rank, alpha, dropout = full_rank, full_alpha, full_dropout
        elif layer_type == "linear_attention":
            branch_name = "linear_attn"
            projections = linear_modules
            group = TEXT_LINEAR_GROUP
            rank, alpha, dropout = linear_rank, linear_alpha, linear_dropout
        else:
            missing.append(f"unsupported_layer_type:{layer_index}:{layer_type}")
            continue
        for projection_name in projections:
            module_name = f"{layer_prefix}.{layer_index}.{branch_name}.{projection_name}"
            module = named_modules.get(module_name)
            if module is None or not hasattr(module, "weight"):
                missing.append(module_name)
                continue
            entries.append(
                LoraTargetSpec(
                    module_name=module_name,
                    module_class=module.__class__.__name__,
                    group=group,
                    layer_index=layer_index,
                    layer_type=layer_type,
                    projection_name=projection_name,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    parameter_count=_parameter_count(module, rank, module_name),
                )
            )

    unexpected_vision = [
        entry.module_name
        for entry in entries
        if any(marker in entry.module_name.lower() for marker in (".visual.", ".vision_", "vision_tower"))
    ]
    if missing or unexpected_vision:
        _raise_plan_error(
            "Text projection discovery did not match the pinned Qwen3.5 revision",
            entries,
            missing=missing,
            unexpected_vision=unexpected_vision,
        )

    merger_cfg = get_by_path(cfg, "vision_merger_lora", {})
    merger_enabled = bool(merger_cfg.get("enabled", False)) if isinstance(merger_cfg, dict) else False
    if merger_enabled:
        suffixes = tuple(str(value) for value in merger_cfg.get("target_suffixes", VISION_MERGER_SUFFIXES))
        if suffixes != VISION_MERGER_SUFFIXES:
            _raise_plan_error(
                f"vision_merger_lora.target_suffixes must be {VISION_MERGER_SUFFIXES}, got {suffixes}", entries
            )
        merger_rank = int(merger_cfg.get("rank", 0))
        merger_alpha = int(merger_cfg.get("alpha", 0))
        merger_dropout = float(merger_cfg.get("dropout", full_dropout))
        if merger_rank <= 0 or merger_alpha <= 0 or merger_dropout != full_dropout:
            _raise_plan_error(
                "Vision merger rank/alpha must be positive and dropout must match the single adapter", entries
            )
        for suffix in suffixes:
            matches = [
                (name, module)
                for name, module in named_modules.items()
                if (name == suffix or name.endswith(f".{suffix}")) and hasattr(module, "weight")
            ]
            if len(matches) != 1:
                _raise_plan_error(
                    f"Expected one exact vision merger match for {suffix}, found {[name for name, _ in matches]}",
                    entries,
                    missing=[suffix],
                )
            module_name, module = matches[0]
            if ".visual.merger." not in module_name or ".visual.blocks." in module_name:
                _raise_plan_error("Vision merger target escaped the merger boundary", entries, unexpected_vision=[module_name])
            entries.append(
                LoraTargetSpec(
                    module_name=module_name,
                    module_class=module.__class__.__name__,
                    group=VISION_MERGER_GROUP,
                    layer_index=None,
                    layer_type="vision_merger",
                    projection_name=suffix.rsplit(".", 1)[-1],
                    rank=merger_rank,
                    alpha=merger_alpha,
                    dropout=merger_dropout,
                    parameter_count=_parameter_count(module, merger_rank, module_name),
                )
            )

    manifest = [asdict(entry) for entry in entries]
    validate_lora_manifest(manifest, vision_merger_enabled=merger_enabled)
    return manifest


def validate_lora_manifest(
    manifest: list[dict[str, Any]],
    *,
    vision_merger_enabled: bool | None = None,
) -> None:
    full = [entry for entry in manifest if entry["group"] == TEXT_FULL_GROUP]
    linear = [entry for entry in manifest if entry["group"] == TEXT_LINEAR_GROUP]
    merger = [entry for entry in manifest if entry["group"] == VISION_MERGER_GROUP]
    errors: list[str] = []
    if len(full) != 32:
        errors.append(f"full modules={len(full)} expected=32")
    if len(linear) != 72:
        errors.append(f"linear modules={len(linear)} expected=72")
    if len({int(entry["layer_index"]) for entry in full}) != 8:
        errors.append("full layers must equal 8")
    if len({int(entry["layer_index"]) for entry in linear}) != 24:
        errors.append("linear layers must equal 24")
    covered = {int(entry["layer_index"]) for entry in full + linear}
    if covered != set(range(32)):
        errors.append(f"covered language layers={sorted(covered)}")
    expected_projection_counts = {
        "q_proj": 8,
        "k_proj": 8,
        "v_proj": 8,
        "o_proj": 8,
        "in_proj_qkv": 24,
        "in_proj_z": 24,
        "out_proj": 24,
    }
    actual_projection_counts = Counter(str(entry["projection_name"]) for entry in full + linear)
    if dict(actual_projection_counts) != expected_projection_counts:
        errors.append(
            f"projection counts={dict(actual_projection_counts)} expected={expected_projection_counts}"
        )
    if vision_merger_enabled is True and len(merger) != 2:
        errors.append(f"vision merger modules={len(merger)} expected=2")
    if vision_merger_enabled is False and merger:
        errors.append(f"vision merger disabled but selected {[entry['module_name'] for entry in merger]}")
    names = [str(entry["module_name"]) for entry in manifest]
    if len(names) != len(set(names)):
        errors.append("duplicate module paths")
    for entry in full:
        if ".language_model.layers." not in entry["module_name"] or ".self_attn." not in entry["module_name"]:
            errors.append(f"invalid full path={entry['module_name']}")
    for entry in linear:
        if ".language_model.layers." not in entry["module_name"] or ".linear_attn." not in entry["module_name"]:
            errors.append(f"invalid linear path={entry['module_name']}")
    for entry in merger:
        if ".visual.merger." not in entry["module_name"] or ".visual.blocks." in entry["module_name"]:
            errors.append(f"invalid merger path={entry['module_name']}")
    if errors:
        _raise_plan_error("; ".join(errors), [LoraTargetSpec(**entry) for entry in manifest])


def enforce_lora_trainability(backbone: nn.Module, manifest: list[dict[str, Any]]) -> None:
    raw_paths = [str(entry["module_name"]) for entry in manifest]
    trainable = 0
    for name, parameter in backbone.named_parameters():
        is_planned_lora = "lora_" in name and any(path in name for path in raw_paths)
        parameter.requires_grad = bool(is_planned_lora)
        if is_planned_lora:
            trainable += int(parameter.numel())
    if trainable <= 0:
        raise RuntimeError("Structured LoRA plan produced no trainable adapter parameters")


def finalize_peft_lora_manifest(
    backbone: nn.Module,
    manifest: list[dict[str, Any]],
    *,
    adapter_name: str = "default",
    require_trainable: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    named_modules = list(backbone.named_modules())
    finalized: list[dict[str, Any]] = []
    for source in manifest:
        entry = dict(source)
        raw_name = str(entry["module_name"])
        matches = [
            (name, module)
            for name, module in named_modules
            if (name == raw_name or name.endswith(f".{raw_name}"))
            and hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one PEFT wrapper for {raw_name}, found {[name for name, _ in matches]}")
        peft_name, module = matches[0]
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            raise RuntimeError(f"Adapter {adapter_name!r} is missing from {peft_name}")
        lora_a = module.lora_A[adapter_name]
        lora_b = module.lora_B[adapter_name]
        actual_rank = int(lora_a.out_features)
        actual_alpha = int(module.lora_alpha[adapter_name])
        if actual_rank != int(entry["rank"]) or actual_alpha != int(entry["alpha"]):
            raise RuntimeError(
                f"LoRA rank/alpha mismatch for {peft_name}: expected "
                f"{entry['rank']}/{entry['alpha']}, got {actual_rank}/{actual_alpha}"
            )
        params = list(lora_a.parameters()) + list(lora_b.parameters())
        if require_trainable and not all(parameter.requires_grad for parameter in params):
            raise RuntimeError(f"Planned LoRA tensors are frozen: {peft_name}")
        entry["peft_module_name"] = peft_name
        entry["parameter_count"] = sum(int(parameter.numel()) for parameter in params)
        finalized.append(entry)

    vision_enabled = any(entry["group"] == VISION_MERGER_GROUP for entry in finalized)
    validate_lora_manifest(finalized, vision_merger_enabled=vision_enabled)
    expected_wrappers = {str(entry["peft_module_name"]) for entry in finalized}
    actual_wrappers = {
        name
        for name, module in named_modules
        if hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and adapter_name in module.lora_A
        and adapter_name in module.lora_B
    }
    if actual_wrappers != expected_wrappers:
        raise RuntimeError(
            "PEFT wrapper set differs from the exact plan: "
            f"missing={sorted(expected_wrappers - actual_wrappers)}, "
            f"unexpected={sorted(actual_wrappers - expected_wrappers)}"
        )
    return finalized, lora_target_summary(finalized)


def lora_target_summary(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for group in (TEXT_FULL_GROUP, TEXT_LINEAR_GROUP, VISION_MERGER_GROUP):
        entries = [entry for entry in manifest if entry["group"] == group]
        groups[group] = {
            "module_count": len(entries),
            "module_paths": [str(entry["module_name"]) for entry in entries],
            "rank": None if not entries else int(entries[0]["rank"]),
            "alpha": None if not entries else int(entries[0]["alpha"]),
            "trainable_parameter_count": sum(int(entry["parameter_count"]) for entry in entries),
        }
    return {
        "groups": groups,
        "text_full_attention_match_count": groups[TEXT_FULL_GROUP]["module_count"],
        "text_linear_attention_match_count": groups[TEXT_LINEAR_GROUP]["module_count"],
        "vision_match_count": groups[VISION_MERGER_GROUP]["module_count"],
        "lora_trainable_parameter_count": sum(int(entry["parameter_count"]) for entry in manifest),
    }


def assert_no_cpu_disk_offload(backbone: nn.Module) -> dict[str, Any]:
    raw_map = getattr(backbone, "hf_device_map", None)
    device_map = {str(key): str(value) for key, value in raw_map.items()} if isinstance(raw_map, dict) else {}
    offloaded = {
        key: value
        for key, value in device_map.items()
        if value.lower() in {"cpu", "disk", "meta"} or value.lower().startswith("cpu:")
    }
    if offloaded:
        raise RuntimeError(f"CPU/disk offload is forbidden for E1: {offloaded}")
    parameter_devices = sorted({str(parameter.device) for parameter in backbone.parameters()})
    forbidden_parameter_devices = [
        device for device in parameter_devices if device in {"cpu", "meta"} or device.startswith("cpu:")
    ]
    if forbidden_parameter_devices and torch.cuda.is_available():
        raise RuntimeError(f"Backbone parameters remain off GPU under single-GPU E1: {parameter_devices}")
    if not device_map and len(parameter_devices) == 1:
        device_map = {"<all_parameters>": parameter_devices[0]}
    return {
        "actual_device_map": device_map,
        "parameter_devices": parameter_devices,
        "cpu_or_disk_offload": bool(offloaded),
    }


def validate_gradient_contract(stage_pair_model: nn.Module) -> dict[str, int]:
    backbone = getattr(stage_pair_model, "backbone", None)
    manifest = getattr(backbone, "_stage_pair_lora_target_manifest", None)
    if backbone is None or not isinstance(manifest, list):
        raise RuntimeError("Gradient contract requires a finalized structured LoRA manifest")
    group_counts = Counter()
    for entry in manifest:
        raw_name = str(entry["module_name"])
        matches = [
            (name, parameter)
            for name, parameter in backbone.named_parameters()
            if raw_name in name and (".lora_A." in name or ".lora_B." in name)
        ]
        if len(matches) != 2 or any(parameter.grad is None for _, parameter in matches):
            raise RuntimeError(f"Missing LoRA gradients for {raw_name}: {[name for name, _ in matches]}")
        group_counts[str(entry["group"])] += sum(int(parameter.numel()) for _, parameter in matches)
    for name, parameter in backbone.named_parameters():
        if "lora_" not in name and (parameter.requires_grad or parameter.grad is not None):
            raise RuntimeError(f"Frozen base parameter violated the gradient contract: {name}")
    head_count = 0
    for module_name in ("set_encoder", "stage_head", "pair_head"):
        module = getattr(stage_pair_model, module_name, None)
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                raise RuntimeError(f"Missing head gradient: {module_name}.{name}")
            head_count += int(parameter.numel())
    return {
        "text_full_attention": int(group_counts[TEXT_FULL_GROUP]),
        "text_linear_attention": int(group_counts[TEXT_LINEAR_GROUP]),
        "vision_merger": int(group_counts[VISION_MERGER_GROUP]),
        "heads": head_count,
    }
