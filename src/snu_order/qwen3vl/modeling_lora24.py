from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from snu_order.utils.config import get_by_path
from snu_order.utils.io import write_json


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

    processor_kwargs: dict[str, Any] = {
        "local_files_only": bool(get_by_path(cfg, "model.local_files_only", True)),
        "trust_remote_code": bool(get_by_path(cfg, "model.trust_remote_code", True)),
    }
    processor_cfg = cfg.get("processor", {})
    if processor_cfg.get("min_pixels") is not None:
        processor_kwargs["min_pixels"] = int(processor_cfg["min_pixels"])
    if processor_cfg.get("max_pixels") is not None:
        processor_kwargs["max_pixels"] = int(processor_cfg["max_pixels"])
    return AutoProcessor.from_pretrained(_resolve_model_id(cfg), **processor_kwargs)


def _ddp_local_rank() -> int | None:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    if world_size <= 1 or "LOCAL_RANK" not in os.environ:
        return None
    return int(os.environ["LOCAL_RANK"])


def load_qwen3_backbone(cfg: dict[str, Any]) -> nn.Module:
    import transformers

    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_cls is None:
        raise ImportError("Qwen3-VL requires Qwen3VLForConditionalGeneration or AutoModelForMultimodalLM")

    local_rank = _ddp_local_rank()
    model_kwargs: dict[str, Any] = {
        "local_files_only": bool(get_by_path(cfg, "model.local_files_only", True)),
        "trust_remote_code": bool(get_by_path(cfg, "model.trust_remote_code", True)),
        "device_map": {"": local_rank} if local_rank is not None else get_by_path(cfg, "model.device_map", "auto"),
    }
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
            except Exception:
                pass
        module_cfg = getattr(module, "config", None)
        vision_cfg = getattr(module_cfg, "vision_config", None)
        if vision_cfg is not None and hasattr(vision_cfg, "deepstack_visual_indexes"):
            vision_cfg.deepstack_visual_indexes = []
        if module_cfg is not None and hasattr(module_cfg, "deepstack_visual_indexes"):
            module_cfg.deepstack_visual_indexes = []


def _module_tail(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def matching_lora_module_names(model: nn.Module, target_modules: Iterable[str]) -> list[str]:
    targets = {str(item) for item in target_modules}
    names: list[str] = []
    for name, module in model.named_modules():
        if _module_tail(name) in targets and hasattr(module, "weight"):
            names.append(name)
    return names


def _freeze_vision_parameters(model: nn.Module) -> None:
    vision_markers = ("visual", "vision", "vision_tower", "image_tower", "image_encoder")
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(marker in lowered for marker in vision_markers):
            parameter.requires_grad = False


def apply_lora_if_enabled(backbone: nn.Module, cfg: dict[str, Any]) -> nn.Module:
    if not bool(get_by_path(cfg, "lora.enabled", True)):
        return backbone
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("QLoRA training requires peft") from exc

    target_modules = list(get_by_path(cfg, "lora.target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
    matches = matching_lora_module_names(backbone, target_modules)
    if not matches:
        raise ValueError(f"No LoRA target modules matched {target_modules}")

    if bool(get_by_path(cfg, "quantization.enabled", True)):
        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing=bool(get_by_path(cfg, "train.gradient_checkpointing", True)),
        )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    lora_cfg = LoraConfig(
        r=int(get_by_path(cfg, "lora.rank", 16)),
        lora_alpha=int(get_by_path(cfg, "lora.alpha", 32)),
        target_modules=target_modules,
        lora_dropout=float(get_by_path(cfg, "lora.dropout", 0.05)),
        bias=str(get_by_path(cfg, "lora.bias", "none")),
        task_type="CAUSAL_LM",
    )
    backbone = get_peft_model(backbone, lora_cfg)
    if bool(get_by_path(cfg, "model.freeze_vision_encoder", True)):
        _freeze_vision_parameters(backbone)
    trainable_matches = [name for name, param in backbone.named_parameters() if param.requires_grad]
    if not trainable_matches:
        raise ValueError("LoRA is enabled but no backbone parameters are trainable")
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
