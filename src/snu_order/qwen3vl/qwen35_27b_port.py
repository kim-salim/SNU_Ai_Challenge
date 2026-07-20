from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from snu_order.utils.config import get_by_path

from .modeling_stage_pair import (
    ANCHOR_POOLING_MODE,
    Qwen3VLStagePairModel,
)
from .stage_pair_scorer import structured_permutation_logits


ARCHITECTURE_ID = "qwen35_27b_stage_pair_e1_int4_v1"
RETENTION_ARCHITECTURE_ID = "qwen35_27b_stage_pair_e1_champion_retention_v2"
SUPPORTED_ARCHITECTURE_IDS = frozenset({ARCHITECTURE_ID, RETENTION_ARCHITECTURE_ID})
EXPECTED_HIDDEN_SIZE = 5120
EXPECTED_LANGUAGE_LAYERS = 64
EXPECTED_FULL_ATTENTION_LAYERS = 16
EXPECTED_LINEAR_ATTENTION_LAYERS = 48
EXPECTED_MODEL_TYPE_MARKER = "qwen3_5"


def is_27b_port_config(cfg: dict[str, Any]) -> bool:
    return str(get_by_path(cfg, "architecture.id", "")) in SUPPORTED_ARCHITECTURE_IDS


def _text_config(config: Any) -> Any:
    for candidate in (
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
        config,
    ):
        if candidate is not None and getattr(candidate, "layer_types", None) is not None:
            return candidate
    raise RuntimeError("Qwen3.5-27B config does not expose text layer_types")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_qwen35_27b_architecture(config: Any) -> dict[str, Any]:
    text_config = _text_config(config)
    model_type = str(getattr(config, "model_type", "")).lower()
    hidden_size = int(getattr(text_config, "hidden_size", -1))
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers", -1))
    layer_types = [str(value) for value in getattr(text_config, "layer_types", [])]
    counts = Counter(layer_types)
    vision_config = getattr(config, "vision_config", None)
    failures: list[str] = []
    if EXPECTED_MODEL_TYPE_MARKER not in model_type:
        failures.append(f"model_type={model_type!r} does not contain {EXPECTED_MODEL_TYPE_MARKER!r}")
    if vision_config is None:
        failures.append("vision_config is missing")
    if hidden_size != EXPECTED_HIDDEN_SIZE:
        failures.append(f"hidden_size={hidden_size} expected={EXPECTED_HIDDEN_SIZE}")
    if num_hidden_layers != EXPECTED_LANGUAGE_LAYERS or len(layer_types) != EXPECTED_LANGUAGE_LAYERS:
        failures.append(
            f"language_layers={num_hidden_layers}/{len(layer_types)} expected={EXPECTED_LANGUAGE_LAYERS}"
        )
    expected_counts = {
        "full_attention": EXPECTED_FULL_ATTENTION_LAYERS,
        "linear_attention": EXPECTED_LINEAR_ATTENTION_LAYERS,
    }
    if dict(counts) != expected_counts:
        failures.append(f"layer_types={dict(counts)} expected={expected_counts}")
    report = {
        "architecture_id": ARCHITECTURE_ID,
        "model_type": model_type,
        "multimodal_vision_config_present": vision_config is not None,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "layer_type_counts": dict(counts),
        "config_sha256": _canonical_sha256(
            config.to_dict() if callable(getattr(config, "to_dict", None)) else vars(config)
        ),
        "status": "PASS" if not failures else "HOLD_QWEN_ARCH_UNSUPPORTED",
        "failures": failures,
    }
    if failures:
        raise RuntimeError(f"HOLD_QWEN_ARCH_UNSUPPORTED: {json.dumps(report, sort_keys=True)}")
    return report


def build_strict_nf4_config(cfg: dict[str, Any]) -> Any:
    if not is_27b_port_config(cfg):
        raise RuntimeError("Strict 27B NF4 builder requires the 27B port architecture id")
    expected = {
        "enabled": True,
        "bits": 4,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "bf16",
    }
    observed = {key: get_by_path(cfg, f"quantization.{key}", None) for key in expected}
    if observed != expected:
        raise RuntimeError(f"HOLD_INT4_BACKEND_INCOMPATIBLE: quantization={observed} expected={expected}")
    dtype_value = get_by_path(
        cfg,
        "backbone.torch_dtype",
        get_by_path(cfg, "model.torch_dtype", ""),
    )
    if str(dtype_value).lower() not in {"bf16", "bfloat16"}:
        raise RuntimeError("HOLD_INT4_BACKEND_INCOMPATIBLE: backbone.torch_dtype must be bfloat16")
    device_map = get_by_path(cfg, "backbone.device_map", get_by_path(cfg, "model.device_map", None))
    if not isinstance(device_map, dict) or set(device_map) != {""} or not isinstance(device_map[""], int):
        raise RuntimeError("HOLD_INT4_BACKEND_INCOMPATIBLE: final device_map must be {'': <cuda-index>}")
    if not bool(get_by_path(cfg, "runtime.require_no_cpu_disk_offload", False)):
        raise RuntimeError("HOLD_INT4_BACKEND_INCOMPATIBLE: CPU/disk offload guard must be enabled")
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("HOLD_INT4_BACKEND_INCOMPATIBLE: BitsAndBytesConfig is unavailable") from exc
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    actual = {
        "load_in_4bit": bool(quant.load_in_4bit),
        "bnb_4bit_quant_type": str(quant.bnb_4bit_quant_type),
        "bnb_4bit_use_double_quant": bool(quant.bnb_4bit_use_double_quant),
        "bnb_4bit_compute_dtype": quant.bnb_4bit_compute_dtype,
    }
    required = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": torch.bfloat16,
    }
    if actual != required:
        raise RuntimeError(f"HOLD_INT4_BACKEND_INCOMPATIBLE: BitsAndBytesConfig={actual}")
    return quant


def quantization_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "backend": "bitsandbytes",
        "load_in_4bit": bool(get_by_path(cfg, "quantization.enabled", False)),
        "bits": int(get_by_path(cfg, "quantization.bits", 0)),
        "quant_type": str(get_by_path(cfg, "quantization.quant_type", "")),
        "double_quant": bool(get_by_path(cfg, "quantization.double_quant", False)),
        "compute_dtype": str(get_by_path(cfg, "quantization.compute_dtype", "")),
        "torch_dtype": str(
            get_by_path(cfg, "backbone.torch_dtype", get_by_path(cfg, "model.torch_dtype", ""))
        ),
        "cpu_disk_offload": False,
        "device_map": get_by_path(
            cfg, "backbone.device_map", get_by_path(cfg, "model.device_map", None)
        ),
    }
    return {**contract, "sha256": _canonical_sha256(contract)}


class FrameProjector(nn.Module):
    """Fresh 27B projector matching the Champion input normalization/projection topology."""

    def __init__(self, in_features: int = EXPECTED_HIDDEN_SIZE, out_features: int = 512) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError(
                f"27B FrameProjector input must be {EXPECTED_HIDDEN_SIZE}, got {self.in_features}"
            )
        self.input_norm = nn.LayerNorm(self.in_features)
        self.proj = nn.Linear(self.in_features, self.out_features)

    def forward(self, frame_hidden: torch.Tensor) -> torch.Tensor:
        if frame_hidden.ndim != 3 or frame_hidden.shape[1] != 4:
            raise ValueError(f"frame_hidden must have shape [B,4,5120], got {tuple(frame_hidden.shape)}")
        if int(frame_hidden.shape[-1]) != self.in_features:
            raise RuntimeError(
                f"27B cache width mismatch: got {int(frame_hidden.shape[-1])}, expected {self.in_features}"
            )
        x = self.input_norm(frame_hidden.to(self.input_norm.weight.dtype))
        return self.proj(x.to(self.proj.weight.dtype))


class PositionFreeContextEncoder(nn.Module):
    """Champion Set Transformer after its width-changing input projection."""

    def __init__(
        self,
        model_dim: int = 512,
        *,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_set_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.use_set_encoder = bool(use_set_encoder)
        if self.use_set_encoder:
            layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=int(nhead),
                dim_feedforward=int(dim_feedforward),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        else:
            self.encoder = nn.Identity()
        self.output_norm = nn.LayerNorm(self.model_dim)

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.ndim != 3 or tuple(projected.shape[1:]) != (4, self.model_dim):
            raise ValueError(f"projected must have shape [B,4,{self.model_dim}], got {tuple(projected.shape)}")
        return self.output_norm(self.encoder(projected))


@dataclass(frozen=True)
class HiddenOnlyResolution:
    owner_class: str
    core_class: str
    path: str


def resolve_hidden_only_core(backbone: nn.Module) -> tuple[nn.Module, HiddenOnlyResolution]:
    root = backbone.get_base_model() if callable(getattr(backbone, "get_base_model", None)) else backbone
    queue: deque[tuple[str, nn.Module]] = deque([("<root>", root)])
    seen: set[int] = set()
    while queue:
        path, module = queue.popleft()
        if id(module) in seen:
            continue
        seen.add(id(module))
        child = getattr(module, "model", None)
        if isinstance(child, nn.Module) and isinstance(getattr(module, "lm_head", None), nn.Module):
            return child, HiddenOnlyResolution(module.__class__.__name__, child.__class__.__name__, f"{path}.model")
        for name in ("base_model", "model"):
            candidate = getattr(module, name, None)
            if isinstance(candidate, nn.Module) and candidate is not module:
                queue.append((f"{path}.{name}", candidate))
    raise RuntimeError("HOLD_HIDDEN_STATE_CONTRACT_FAILURE: no multimodal core below a CausalLM lm_head")


def hidden_only_multimodal_forward(backbone: nn.Module, inputs: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    core, resolution = resolve_hidden_only_core(backbone)
    kwargs = dict(inputs)
    if kwargs.pop("output_hidden_states", False):
        raise RuntimeError("HOLD_HIDDEN_STATE_CONTRACT_FAILURE: caller requested all hidden states")
    if kwargs.pop("use_cache", False):
        raise RuntimeError("HOLD_HIDDEN_STATE_CONTRACT_FAILURE: caller requested KV cache")
    kwargs["output_hidden_states"] = False
    kwargs["return_dict"] = True
    kwargs["use_cache"] = False
    outputs = core(**kwargs)
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None and isinstance(outputs, (tuple, list)) and outputs:
        last_hidden = outputs[0]
    if not torch.is_tensor(last_hidden) or last_hidden.ndim != 3:
        raise RuntimeError("HOLD_HIDDEN_STATE_CONTRACT_FAILURE: inner backbone did not return [B,L,H]")
    if int(last_hidden.shape[-1]) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(
            f"HOLD_HIDDEN_STATE_CONTRACT_FAILURE: hidden width={last_hidden.shape[-1]} expected=5120"
        )
    if getattr(outputs, "past_key_values", None) is not None:
        raise RuntimeError("HOLD_HIDDEN_STATE_CONTRACT_FAILURE: KV cache was produced")
    return last_hidden, {
        "owner_class": resolution.owner_class,
        "core_class": resolution.core_class,
        "core_path": resolution.path,
        "output_hidden_states": False,
        "use_cache": False,
        "lm_head_called": False,
    }


class Qwen35_27BStagePairE1Model(Qwen3VLStagePairModel):
    def __init__(self, backbone: nn.Module | None, **kwargs: Any) -> None:
        hidden_size = int(kwargs.pop("hidden_size"))
        if hidden_size != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError(f"27B model hidden_size must be 5120, got {hidden_size}")
        super().__init__(backbone, hidden_size=hidden_size, **kwargs)
        model_dim = self.model_dim
        original = self.set_encoder
        self.frame_projector = FrameProjector(hidden_size, model_dim)
        self.set_encoder = PositionFreeContextEncoder(
            model_dim,
            num_layers=len(original.encoder.layers) if isinstance(original.encoder, nn.TransformerEncoder) else 0,
            nhead=original.encoder.layers[0].self_attn.num_heads
            if isinstance(original.encoder, nn.TransformerEncoder)
            else 8,
            dim_feedforward=original.encoder.layers[0].linear1.out_features
            if isinstance(original.encoder, nn.TransformerEncoder)
            else 2048,
            dropout=float(original.encoder.layers[0].dropout.p)
            if isinstance(original.encoder, nn.TransformerEncoder)
            else 0.1,
            use_set_encoder=original.use_set_encoder,
        )
        self._last_hidden_only_contract: dict[str, Any] | None = None
        self._freeze_backbone_forward = False

    def set_backbone_forward_frozen(self, frozen: bool) -> None:
        self._freeze_backbone_forward = bool(frozen)

    def _extract_pooled_rows(
        self,
        inputs: dict[str, Any],
        *,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backbone is None:
            raise ValueError("backbone is required for live Qwen frame representation extraction")
        if self.pooling_mode != ANCHOR_POOLING_MODE or anchor_mask is None:
            raise RuntimeError("27B Champion port requires STATE anchor_span_mean pooling")
        kwargs = dict(inputs)
        embedded_anchor = kwargs.pop("anchor_mask", None)
        if embedded_anchor is not None:
            if anchor_mask is not None:
                raise RuntimeError("anchor_mask was supplied twice")
            anchor_mask = embedded_anchor
        with torch.set_grad_enabled(
            self.backbone_trainable and self.training and not self._freeze_backbone_forward
        ):
            last_hidden, contract = hidden_only_multimodal_forward(self.backbone, kwargs)
        self._last_hidden_only_contract = contract
        return self.anchor_pooler(last_hidden, anchor_mask)

    def forward(
        self,
        *,
        frame_hidden: torch.Tensor | None = None,
        inputs: dict[str, Any] | None = None,
        batch_size: int | None = None,
        anchor_mask: torch.Tensor | None = None,
        frame_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if frame_hidden is None:
            if inputs is None or batch_size is None:
                raise ValueError("Either frame_hidden or inputs+batch_size must be provided")
            frame_hidden = self.extract_frame_representations(
                inputs,
                batch_size=int(batch_size),
                anchor_mask=anchor_mask,
                frame_chunk_size=frame_chunk_size,
            )
        projected = self.frame_projector(frame_hidden)
        contextual = self.set_encoder(projected)
        stage_logits = self.stage_head(contextual)
        pair_logits = self.pair_head(contextual) if self.pair_head is not None else None
        final_logits = structured_permutation_logits(
            stage_logits,
            pair_logits,
            stage_weight=self.stage_weight,
            pair_weight=self.pair_score_weight if pair_logits is not None else 0.0,
        )
        return {
            "frame_hidden": frame_hidden,
            "projected": projected,
            "contextual": contextual,
            "stage_logits": stage_logits,
            "pair_logits": pair_logits
            if pair_logits is not None
            else stage_logits.new_zeros((stage_logits.shape[0], 6)),
            "final_logits": final_logits,
        }


def assert_27b_cache_compatible(payload: dict[str, Any], expected_identity: dict[str, Any] | None = None) -> None:
    frame_hidden = payload.get("frame_hidden")
    if not torch.is_tensor(frame_hidden) or frame_hidden.ndim != 3 or frame_hidden.shape[1] != 4:
        raise RuntimeError("27B cache must contain frame_hidden with shape [N,4,5120]")
    if int(frame_hidden.shape[-1]) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(
            f"27B cache width mismatch: got {int(frame_hidden.shape[-1])}, expected {EXPECTED_HIDDEN_SIZE}"
        )
    identity = payload.get("cache_identity")
    if expected_identity is not None:
        if not isinstance(identity, dict):
            raise RuntimeError("27B cache is missing cache_identity")
        mismatches = {
            key: {"cache": identity.get(key), "runtime": value}
            for key, value in expected_identity.items()
            if identity.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"27B cache identity mismatch: {json.dumps(mismatches, sort_keys=True)}")


def load_migrated_champion_heads(path: str | Any, model: nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "architecture",
        "hidden_size",
        "model_dim",
        "pooling_mode",
        "frame_projector",
        "set_encoder",
        "stage_head",
        "pair_head",
        "migration_source_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("Migrated Champion head payload schema mismatch")
    if payload["architecture"] not in SUPPORTED_ARCHITECTURE_IDS or int(payload["hidden_size"]) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("Migrated Champion head identity mismatch")
    if int(payload["model_dim"]) != int(getattr(model, "model_dim", -1)):
        raise RuntimeError("Migrated Champion model_dim mismatch")
    if str(payload["pooling_mode"]) != str(getattr(model, "pooling_mode", "")):
        raise RuntimeError("Migrated Champion pooling mismatch")
    for name in ("frame_projector", "set_encoder", "stage_head", "pair_head"):
        module = getattr(model, name, None)
        if module is None or payload[name] is None:
            raise RuntimeError(f"Migrated Champion payload/model is missing {name}")
        module.load_state_dict(payload[name], strict=True)
    return {
        "status": "PASS",
        "source_sha256": str(payload["migration_source_sha256"]),
        "fresh_frame_projector_loaded": True,
        "compatible_champion_heads_loaded": True,
    }
