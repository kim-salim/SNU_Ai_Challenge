from __future__ import annotations

import inspect
import os
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from snu_order.utils.config import get_by_path
from snu_order.utils.io import write_json

from .lora_targets import (
    assert_no_cpu_disk_offload,
    discover_qwen35_lora_targets,
    enforce_lora_trainability,
    finalize_peft_lora_manifest,
    is_structured_lora_config,
)


def torch_dtype_from_string(value: str | None) -> torch.dtype | None:
    if value is None:
        return None
    normalized = str(value).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [B,L], got {tuple(attention_mask.shape)}")
    mask = attention_mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("attention_mask contains an all-padding row")
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    return positions.masked_fill(~mask, -1).max(dim=1).values.long()


class LastNonPaddingPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have shape [B,L,H], got {tuple(hidden_states.shape)}")
        indices = last_non_padding_indices(attention_mask)
        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_idx, indices]


class PermutationClassifierHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 24, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = pooled.to(self.norm.weight.dtype)
        return self.linear(self.dropout(self.norm(x)))


class Qwen3VL24WayClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int,
        num_classes: int = 24,
        classifier_dropout: float = 0.1,
        backbone_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.pooler = LastNonPaddingPooler()
        self.classifier = PermutationClassifierHead(
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            dropout=classifier_dropout,
        )
        self.backbone_trainable = bool(backbone_trainable)

    def _backbone_forward(self, inputs: dict[str, Any]) -> Any:
        kwargs = dict(inputs)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        kwargs["use_cache"] = False
        return self.backbone(**kwargs)

    def forward(self, **inputs: Any) -> torch.Tensor:
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("attention_mask is required for last_non_padding_token pooling")
        with torch.set_grad_enabled(self.backbone_trainable and self.training):
            outputs = self._backbone_forward(inputs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            last_hidden = getattr(outputs, "last_hidden_state", None)
        else:
            last_hidden = hidden_states[-1]
        if last_hidden is None:
            raise ValueError("Backbone output does not contain hidden_states or last_hidden_state")
        pooled = self.pooler(last_hidden, attention_mask.to(last_hidden.device))
        return self.classifier(pooled)


def _hidden_size_from_config(config: Any) -> int:
    for obj in (config, getattr(config, "text_config", None), getattr(config, "llm_config", None)):
        if obj is None:
            continue
        value = getattr(obj, "hidden_size", None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden_size from model config")


def _bnb_config(cfg: dict[str, Any]) -> Any | None:
    quant_cfg = cfg.get("quantization", {})
    if not bool(quant_cfg.get("enabled", False)):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("4-bit QLoRA requires transformers BitsAndBytesConfig and bitsandbytes") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype_from_string(str(quant_cfg.get("compute_dtype", "bf16"))),
        bnb_4bit_quant_type=str(quant_cfg.get("quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(quant_cfg.get("double_quant", True)),
    )


def _resolve_model_id(cfg: dict[str, Any]) -> str:
    return str(get_by_path(cfg, "model.local_dir", None) or get_by_path(cfg, "model.name", "Qwen/Qwen3-VL-8B-Instruct"))


def load_qwen3_processor(cfg: dict[str, Any]) -> Any:
    from transformers import AutoProcessor

    model_cfg = cfg.get("model", {})
    processor_kwargs: dict[str, Any] = {
        "local_files_only": bool(get_by_path(cfg, "model.local_files_only", True)),
        "trust_remote_code": bool(get_by_path(cfg, "model.trust_remote_code", True)),
    }
    if model_cfg.get("revision"):
        processor_kwargs["revision"] = str(model_cfg["revision"])
    processor_cfg = cfg.get("processor", {})
    if processor_cfg.get("min_pixels") is not None:
        processor_kwargs["min_pixels"] = int(processor_cfg["min_pixels"])
    if processor_cfg.get("max_pixels") is not None:
        processor_kwargs["max_pixels"] = int(processor_cfg["max_pixels"])
    return AutoProcessor.from_pretrained(_resolve_model_id(cfg), **processor_kwargs)


def _qwen_model_class(cfg: dict[str, Any]) -> Any:
    import transformers

    model_type = str(get_by_path(cfg, "model.type", "qwen3_vl")).lower()
    if model_type in {"qwen3_5", "qwen3_5_vl", "qwen3_6", "qwen3_6_vl"}:
        candidates = (
            "Qwen3_5ForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForMultimodalLM",
        )
    else:
        candidates = (
            "Qwen3VLForConditionalGeneration",
            "AutoModelForMultimodalLM",
            "AutoModelForImageTextToText",
        )
    for name in candidates:
        model_cls = getattr(transformers, name, None)
        if model_cls is not None:
            return model_cls
    raise ImportError(f"Could not find a transformers model class for {model_type}; tried {candidates}")


def _ddp_local_rank() -> int | None:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    if world_size <= 1 or "LOCAL_RANK" not in os.environ:
        return None
    return int(os.environ["LOCAL_RANK"])


def load_qwen3_backbone(cfg: dict[str, Any]) -> nn.Module:
    model_cls = _qwen_model_class(cfg)
    model_cfg = cfg.get("model", {})
    local_rank = _ddp_local_rank()
    model_kwargs: dict[str, Any] = {
        "local_files_only": bool(get_by_path(cfg, "model.local_files_only", True)),
        "trust_remote_code": bool(get_by_path(cfg, "model.trust_remote_code", True)),
        "device_map": {"": local_rank} if local_rank is not None else get_by_path(cfg, "model.device_map", "auto"),
    }
    if model_cfg.get("revision"):
        model_kwargs["revision"] = str(model_cfg["revision"])
    dtype = torch_dtype_from_string(str(get_by_path(cfg, "model.torch_dtype", "bfloat16")))
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    quantization_config = _bnb_config(cfg)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if bool(get_by_path(cfg, "attention.use_flash_attention_2", False)):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = model_cls.from_pretrained(_resolve_model_id(cfg), **model_kwargs)
        _disable_deepstack_visual_features_if_needed(model, cfg)
        return model
    except Exception as exc:
        if model_kwargs.get("attn_implementation") == "flash_attention_2" and bool(
            get_by_path(cfg, "attention.fallback_to_sdpa", True)
        ):
            warnings.warn(
                f"flash_attention_2 load failed ({exc}); retrying with sdpa/default attention",
                RuntimeWarning,
                stacklevel=2,
            )
            model_kwargs.pop("attn_implementation", None)
            try:
                model_kwargs["attn_implementation"] = "sdpa"
                model = model_cls.from_pretrained(_resolve_model_id(cfg), **model_kwargs)
                _disable_deepstack_visual_features_if_needed(model, cfg)
                return model
            except Exception:
                model_kwargs.pop("attn_implementation", None)
                model = model_cls.from_pretrained(_resolve_model_id(cfg), **model_kwargs)
                _disable_deepstack_visual_features_if_needed(model, cfg)
                return model
        raise


def _disable_deepstack_visual_features_if_needed(model: nn.Module, cfg: dict[str, Any]) -> None:
    if not bool(get_by_path(cfg, "model.disable_deepstack_visual_features", False)):
        return
    for module in model.modules():
        if hasattr(module, "deepstack_visual_indexes"):
            try:
                module.deepstack_visual_indexes = []
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to disable deepstack_visual_indexes on {module.__class__.__name__}"
                ) from exc
        module_cfg = getattr(module, "config", None)
        vision_cfg = getattr(module_cfg, "vision_config", None)
        if vision_cfg is not None and hasattr(vision_cfg, "deepstack_visual_indexes"):
            vision_cfg.deepstack_visual_indexes = []
        if module_cfg is not None and hasattr(module_cfg, "deepstack_visual_indexes"):
            module_cfg.deepstack_visual_indexes = []


def matching_language_lora_module_names(model: nn.Module, target_modules: list[str]) -> list[str]:
    targets = {str(item) for item in target_modules}
    names: list[str] = []
    for name, module in model.named_modules():
        if ".language_model.layers." not in name or not hasattr(module, "weight"):
            continue
        for target in targets:
            if "." in target:
                matched = name == target or name.endswith(f".{target}")
            else:
                matched = name.endswith(f".self_attn.{target}")
            if matched:
                names.append(name)
                break
    return names


def _freeze_vision_parameters(model: nn.Module) -> None:
    vision_markers = ("visual", "vision", "vision_tower", "image_tower", "image_encoder")
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(marker in lowered for marker in vision_markers) and "lora_" not in lowered:
            parameter.requires_grad = False


def apply_lora_if_enabled(backbone: nn.Module, cfg: dict[str, Any]) -> nn.Module:
    if not bool(get_by_path(cfg, "lora.enabled", True)):
        return backbone
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("QLoRA training requires peft") from exc

    structured_manifest: list[dict[str, Any]] | None = None
    rank_pattern: dict[str, int] | None = None
    alpha_pattern: dict[str, int] | None = None
    if is_structured_lora_config(cfg):
        parameters = inspect.signature(LoraConfig).parameters
        missing_features = [name for name in ("rank_pattern", "alpha_pattern") if name not in parameters]
        if missing_features:
            try:
                import peft

                peft_version = getattr(peft, "__version__", "unknown")
            except ImportError:
                peft_version = "unavailable"
            raise RuntimeError(
                f"Installed PEFT {peft_version} lacks required LoraConfig fields: {missing_features}"
            )
        structured_manifest = discover_qwen35_lora_targets(backbone, cfg)
        target_modules = [str(entry["module_name"]) for entry in structured_manifest]
        rank_pattern = {str(entry["module_name"]): int(entry["rank"]) for entry in structured_manifest}
        alpha_pattern = {str(entry["module_name"]): int(entry["alpha"]) for entry in structured_manifest}
        lora_rank = int(get_by_path(cfg, "lora.full_attention.rank"))
        lora_alpha = int(get_by_path(cfg, "lora.full_attention.alpha"))
        lora_dropout = float(get_by_path(cfg, "lora.full_attention.dropout"))
    else:
        requested = list(get_by_path(cfg, "lora.target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
        target_modules = matching_language_lora_module_names(backbone, requested)
        if not target_modules:
            raise ValueError(f"No exact language-backbone LoRA targets matched {requested}")
        lora_rank = int(get_by_path(cfg, "lora.rank", 16))
        lora_alpha = int(get_by_path(cfg, "lora.alpha", 32))
        lora_dropout = float(get_by_path(cfg, "lora.dropout", 0.05))

    if bool(get_by_path(cfg, "quantization.enabled", True)):
        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing=bool(get_by_path(cfg, "train.gradient_checkpointing", True)),
        )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    lora_kwargs: dict[str, Any] = {
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "lora_dropout": lora_dropout,
        "bias": str(get_by_path(cfg, "lora.bias", "none")),
        "task_type": "CAUSAL_LM",
    }
    if rank_pattern is not None and alpha_pattern is not None:
        lora_kwargs["rank_pattern"] = rank_pattern
        lora_kwargs["alpha_pattern"] = alpha_pattern
    lora_cfg = LoraConfig(**lora_kwargs)
    backbone = get_peft_model(backbone, lora_cfg)
    if structured_manifest is not None:
        enforce_lora_trainability(backbone, structured_manifest)
    elif bool(get_by_path(cfg, "model.freeze_vision_encoder", True)):
        _freeze_vision_parameters(backbone)
    trainable_matches = [name for name, param in backbone.named_parameters() if param.requires_grad]
    if not trainable_matches:
        raise ValueError("LoRA is enabled but no backbone parameters are trainable")
    if structured_manifest is not None:
        finalized_manifest, summary = finalize_peft_lora_manifest(backbone, structured_manifest)
        backbone._stage_pair_lora_target_manifest = finalized_manifest
        backbone._stage_pair_lora_target_summary = summary
    if bool(get_by_path(cfg, "runtime.require_no_cpu_disk_offload", False)):
        backbone._stage_pair_device_report = assert_no_cpu_disk_offload(backbone)
    return backbone


def build_qwen3vl_lora24_model(cfg: dict[str, Any], *, frozen_probe: bool = False) -> tuple[Qwen3VL24WayClassifier, Any]:
    processor = load_qwen3_processor(cfg)
    backbone = load_qwen3_backbone(cfg)
    if bool(get_by_path(cfg, "train.gradient_checkpointing", True)) and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if bool(get_by_path(cfg, "model.freeze_base_model", True)):
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    if not frozen_probe:
        backbone = apply_lora_if_enabled(backbone, cfg)
    else:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    hidden_size = _hidden_size_from_config(backbone.config)
    model = Qwen3VL24WayClassifier(
        backbone,
        hidden_size=hidden_size,
        num_classes=int(get_by_path(cfg, "model.num_classes", 24)),
        classifier_dropout=float(get_by_path(cfg, "model.classifier_dropout", 0.1)),
        backbone_trainable=not frozen_probe and bool(get_by_path(cfg, "lora.enabled", True)),
    )
    if not bool(get_by_path(cfg, "model.train_classifier", True)):
        for parameter in model.classifier.parameters():
            parameter.requires_grad = False
    return model, processor


def trainable_parameter_report(model: nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    names: list[str] = []
    for name, param in model.named_parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
            names.append(name)
    return {
        "total": total,
        "trainable": trainable,
        "ratio": trainable / max(total, 1),
        "trainable_names": names,
    }


def write_trainable_report(path: str | Path, model: nn.Module) -> dict[str, Any]:
    report = trainable_parameter_report(model)
    write_json(path, report)
    return report


def dump_version_report(path: str | Path) -> None:
    report: dict[str, Any] = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    try:
        import transformers

        report["transformers"] = transformers.__version__
    except Exception as exc:
        report["transformers_error"] = repr(exc)
    try:
        import peft

        report["peft"] = peft.__version__
    except Exception as exc:
        report["peft_error"] = repr(exc)
    try:
        import bitsandbytes as bnb

        report["bitsandbytes"] = getattr(bnb, "__version__", "unknown")
    except Exception as exc:
        report["bitsandbytes_error"] = repr(exc)
    write_json(path, report)
