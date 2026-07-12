from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from snu_order.utils.config import get_by_path
from snu_order.utils.io import write_json

from .checkpoint import load_lora24_checkpoint, unwrap_model
from .modeling_lora24 import (
    LastNonPaddingPooler,
    _hidden_size_from_config,
    apply_lora_if_enabled,
    dump_version_report,
    load_qwen3_backbone,
    load_qwen3_processor,
    trainable_parameter_report,
)
from .permutations import PAIRS, PERMS
from .stage_pair_scorer import structured_permutation_logits


class PositionFreeSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 512,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_set_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.use_set_encoder = bool(use_set_encoder)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.proj = nn.Linear(self.input_dim, self.model_dim)
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

    def forward(self, frame_hidden: torch.Tensor) -> torch.Tensor:
        if frame_hidden.ndim != 3 or frame_hidden.shape[1] != 4:
            raise ValueError(f"frame_hidden must have shape [B,4,H], got {tuple(frame_hidden.shape)}")
        x = self.input_norm(frame_hidden.to(self.input_norm.weight.dtype))
        x = self.proj(x.to(self.proj.weight.dtype))
        x = self.encoder(x)
        return self.output_norm(x)


class StageHead(nn.Module):
    def __init__(self, model_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(model_dim), nn.Dropout(float(dropout)), nn.Linear(model_dim, 4))

    def forward(self, contextual: torch.Tensor) -> torch.Tensor:
        if contextual.ndim != 3 or contextual.shape[1] != 4:
            raise ValueError(f"contextual must have shape [B,4,D], got {tuple(contextual.shape)}")
        return self.net(contextual)


class AntiSymmetricPairwiseHead(nn.Module):
    def __init__(self, model_dim: int = 512, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * int(model_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def _raw(self, ci: torch.Tensor, cj: torch.Tensor) -> torch.Tensor:
        features = torch.cat([ci, cj, cj - ci, ci * cj], dim=-1)
        return self.mlp(features).squeeze(-1)

    def directional_logits(self, contextual: torch.Tensor) -> torch.Tensor:
        if contextual.ndim != 3 or contextual.shape[1] != 4:
            raise ValueError(f"contextual must have shape [B,4,D], got {tuple(contextual.shape)}")
        bsz = contextual.shape[0]
        out = contextual.new_zeros((bsz, 4, 4))
        for i, j in PAIRS:
            raw_ij = self._raw(contextual[:, i], contextual[:, j])
            raw_ji = self._raw(contextual[:, j], contextual[:, i])
            dij = 0.5 * (raw_ij - raw_ji)
            out[:, i, j] = dij
            out[:, j, i] = -dij
        return out

    def forward(self, contextual: torch.Tensor) -> torch.Tensor:
        directional = self.directional_logits(contextual)
        return torch.stack([directional[:, i, j] for i, j in PAIRS], dim=1)


class Qwen3VLStagePairModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module | None,
        *,
        hidden_size: int,
        model_dim: int = 512,
        set_layers: int = 2,
        set_heads: int = 8,
        set_ffn_dim: int = 2048,
        dropout: float = 0.1,
        use_set_encoder: bool = True,
        use_pairwise: bool = True,
        stage_weight: float = 1.0,
        pair_score_weight: float = 0.3,
        backbone_trainable: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.model_dim = int(model_dim)
        self.use_pairwise = bool(use_pairwise)
        self.stage_weight = float(stage_weight)
        self.pair_score_weight = float(pair_score_weight)
        self.backbone_trainable = bool(backbone_trainable)
        self.pooler = LastNonPaddingPooler()
        self.set_encoder = PositionFreeSetEncoder(
            self.hidden_size,
            model_dim=self.model_dim,
            num_layers=set_layers,
            nhead=set_heads,
            dim_feedforward=set_ffn_dim,
            dropout=dropout,
            use_set_encoder=use_set_encoder,
        )
        self.stage_head = StageHead(self.model_dim, dropout=dropout)
        self.pair_head = AntiSymmetricPairwiseHead(self.model_dim, self.model_dim, dropout=dropout) if self.use_pairwise else None

    def extract_frame_representations(self, inputs: dict[str, Any], *, batch_size: int) -> torch.Tensor:
        if self.backbone is None:
            raise ValueError("backbone is required for live Qwen frame representation extraction")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("attention_mask is required for last non-padding pooling")
        kwargs = dict(inputs)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        kwargs["use_cache"] = False
        with torch.set_grad_enabled(self.backbone_trainable and self.training):
            outputs = self.backbone(**kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        last_hidden = hidden_states[-1] if hidden_states is not None else getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("Backbone output does not contain hidden states")
        pooled = self.pooler(last_hidden, attention_mask.to(last_hidden.device))
        expected = int(batch_size) * 4
        if pooled.shape[0] != expected:
            raise ValueError(f"Expected B*4={expected} frame representations, got {pooled.shape[0]}")
        return pooled.reshape(int(batch_size), 4, pooled.shape[-1])

    def forward(
        self,
        *,
        frame_hidden: torch.Tensor | None = None,
        inputs: dict[str, Any] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if frame_hidden is None:
            if inputs is None or batch_size is None:
                raise ValueError("Either frame_hidden or inputs+batch_size must be provided")
            frame_hidden = self.extract_frame_representations(inputs, batch_size=int(batch_size))
        contextual = self.set_encoder(frame_hidden)
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
            "contextual": contextual,
            "stage_logits": stage_logits,
            "pair_logits": pair_logits if pair_logits is not None else stage_logits.new_zeros((stage_logits.shape[0], len(PAIRS))),
            "final_logits": final_logits,
        }


def build_stage_pair_head_from_config(cfg: dict[str, Any], *, hidden_size: int, backbone: nn.Module | None = None) -> Qwen3VLStagePairModel:
    return Qwen3VLStagePairModel(
        backbone,
        hidden_size=hidden_size,
        model_dim=int(get_by_path(cfg, "model.model_dim", 512)),
        set_layers=int(get_by_path(cfg, "model.set_layers", 2)),
        set_heads=int(get_by_path(cfg, "model.set_heads", 8)),
        set_ffn_dim=int(get_by_path(cfg, "model.set_ffn_dim", 2048)),
        dropout=float(get_by_path(cfg, "model.dropout", 0.1)),
        use_set_encoder=bool(get_by_path(cfg, "model.use_set_encoder", True)),
        use_pairwise=bool(get_by_path(cfg, "model.use_pairwise", True)),
        stage_weight=float(get_by_path(cfg, "score.stage_weight", 1.0)),
        pair_score_weight=float(get_by_path(cfg, "score.pair_weight", 0.3)),
        backbone_trainable=not bool(get_by_path(cfg, "backbone.frozen", True)),
    )


def _load_existing_adapter(backbone: nn.Module, cfg: dict[str, Any], *, is_trainable: bool) -> nn.Module:
    adapter_root = Path(str(get_by_path(cfg, "backbone.existing_lora_path", "weights/qwen3vl_lora24/best")))
    adapter_dir = adapter_root / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"existing LoRA adapter not found: {adapter_dir}")
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading existing LoRA adapter requires peft") from exc
    return PeftModel.from_pretrained(backbone, str(adapter_dir), is_trainable=is_trainable)


def build_stage_pair_model_from_config(cfg: dict[str, Any], *, live_backbone: bool) -> tuple[Qwen3VLStagePairModel, Any | None]:
    if not live_backbone:
        hidden_size = int(get_by_path(cfg, "cache.hidden_size", get_by_path(cfg, "backbone.hidden_size", 4096)))
        return build_stage_pair_head_from_config(cfg, hidden_size=hidden_size, backbone=None), None

    processor = load_qwen3_processor({"model": {"local_dir": get_by_path(cfg, "backbone.base_model_path"), "local_files_only": True, "trust_remote_code": True}, "processor": cfg.get("processor", {})})
    backbone_cfg = {
        "model": {
            "local_dir": get_by_path(cfg, "backbone.base_model_path", "Qwen/Qwen3-VL-8B-Instruct"),
            "local_files_only": bool(get_by_path(cfg, "backbone.local_files_only", True)),
            "trust_remote_code": bool(get_by_path(cfg, "backbone.trust_remote_code", True)),
            "torch_dtype": get_by_path(cfg, "backbone.torch_dtype", "bfloat16"),
            "device_map": get_by_path(cfg, "backbone.device_map", "auto"),
            "freeze_vision_encoder": True,
            "disable_deepstack_visual_features": bool(get_by_path(cfg, "backbone.disable_deepstack_visual_features", True)),
        },
        "attention": cfg.get("attention", {}),
        "quantization": cfg.get("quantization", {}),
        "train": cfg.get("train", {}),
        "lora": cfg.get("lora", {}),
    }
    backbone = load_qwen3_backbone(backbone_cfg)
    source = str(get_by_path(cfg, "backbone.source", "base"))
    frozen = bool(get_by_path(cfg, "backbone.frozen", True))
    if source == "existing_lora":
        backbone = _load_existing_adapter(backbone, cfg, is_trainable=not frozen)
    elif source == "base" and not frozen:
        backbone = apply_lora_if_enabled(backbone, backbone_cfg)
    elif source != "base":
        raise ValueError(f"Unsupported backbone.source: {source}")
    if frozen:
        for param in backbone.parameters():
            param.requires_grad = False
    hidden_size = _hidden_size_from_config(backbone.config)
    model = build_stage_pair_head_from_config(cfg, hidden_size=hidden_size, backbone=backbone)
    return model, processor


def save_stage_pair_checkpoint(
    path: str | Path,
    model: nn.Module,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    *,
    processor: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
    minimal: bool = False,
) -> None:
    model = unwrap_model(model)
    ckpt = Path(path)
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "set_encoder": model.set_encoder.state_dict(),
            "stage_head": model.stage_head.state_dict(),
            "pair_head": None if model.pair_head is None else model.pair_head.state_dict(),
            "hidden_size": int(model.hidden_size),
            "model_dim": int(model.model_dim),
            "metrics": metrics,
        },
        ckpt / "heads.pt",
    )
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "save_pretrained"):
        try:
            backbone.save_pretrained(ckpt / "adapter")
        except Exception:
            pass
    if processor is not None and hasattr(processor, "save_pretrained"):
        try:
            processor.save_pretrained(ckpt / "processor")
        except Exception:
            pass
    write_json(ckpt / "config.json", cfg)
    write_json(ckpt / "metrics.json", metrics)
    write_json(ckpt / "permutations.json", {"perms": [list(p) for p in PERMS]})
    if extra:
        write_json(ckpt / "extra.json", extra)
    if not minimal:
        state: dict[str, Any] = {}
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if state:
            torch.save(state, ckpt / "training_state.pt")


def _adapter_names(backbone: nn.Module, fallback: str) -> list[str]:
    active = getattr(backbone, "active_adapters", None)
    if isinstance(active, (list, tuple)) and active:
        return [str(v) for v in active]
    active_one = getattr(backbone, "active_adapter", None)
    if isinstance(active_one, str) and active_one:
        return [active_one]
    return [fallback]


def _align_lora_adapter_devices(backbone: nn.Module, adapter_name: str = "stage_pair") -> None:
    names = set(_adapter_names(backbone, adapter_name))
    names.add(adapter_name)
    for module in backbone.modules():
        weight = getattr(module, "weight", None)
        device = getattr(weight, "device", None)
        if device is None:
            continue
        for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
            adapters = getattr(module, attr, None)
            if adapters is None:
                continue
            for name in names:
                if name in adapters:
                    adapters[name].to(device)


def load_stage_pair_checkpoint(path: str | Path, model: nn.Module, *, strict: bool = True, is_trainable: bool = False) -> tuple[nn.Module, dict[str, Any]]:
    model = unwrap_model(model)
    ckpt = Path(path)
    adapter_dir = ckpt / "adapter"
    loaded_adapter_name = "stage_pair"
    if adapter_dir.exists() and getattr(model, "backbone", None) is not None:
        try:
            if hasattr(model.backbone, "load_adapter"):
                model.backbone.load_adapter(str(adapter_dir), adapter_name=loaded_adapter_name, is_trainable=is_trainable)
                if hasattr(model.backbone, "set_adapter"):
                    model.backbone.set_adapter(loaded_adapter_name)
            else:
                from peft import PeftModel

                model.backbone = PeftModel.from_pretrained(model.backbone, str(adapter_dir), is_trainable=is_trainable)
                loaded_adapter_name = "default"
            _align_lora_adapter_devices(model.backbone, loaded_adapter_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load stage-pair adapter from {adapter_dir}") from exc
    payload = torch.load(ckpt / "heads.pt", map_location="cpu")
    model.set_encoder.load_state_dict(payload["set_encoder"], strict=strict)
    model.stage_head.load_state_dict(payload["stage_head"], strict=strict)
    if model.pair_head is not None and payload.get("pair_head") is not None:
        model.pair_head.load_state_dict(payload["pair_head"], strict=strict)
    return model, payload


def write_stage_pair_trainable_report(path: str | Path, model: nn.Module) -> dict[str, Any]:
    report = trainable_parameter_report(model)
    write_json(path, report)
    return report
