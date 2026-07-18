from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from snu_order.utils.config import get_by_path
from snu_order.utils.io import write_json

from .checkpoint import unwrap_model
from .frame_chunking import normalize_frame_chunk_size, slice_frame_multimodal_inputs
from .lora_targets import (
    assert_no_cpu_disk_offload,
    lora_target_summary,
)
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
from .stage_pair_prompt import (
    ANCHOR_POOLING_MODE,
    LEGACY_POOLING_MODE,
    AnchorSpanMeanPooler,
    StagePairPromptSpec,
)
from .stage_pair_checkpoint import load_stage_pair_checkpoint, save_stage_pair_checkpoint
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
        pooling_mode: str = LEGACY_POOLING_MODE,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.model_dim = int(model_dim)
        self.use_pairwise = bool(use_pairwise)
        self.stage_weight = float(stage_weight)
        self.pair_score_weight = float(pair_score_weight)
        self.backbone_trainable = bool(backbone_trainable)
        if pooling_mode not in {LEGACY_POOLING_MODE, ANCHOR_POOLING_MODE}:
            raise RuntimeError(f"Unsupported pooling mode: {pooling_mode}")
        self.pooling_mode = pooling_mode
        self.pooler = LastNonPaddingPooler()
        self.anchor_pooler = AnchorSpanMeanPooler()
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

    def _extract_pooled_rows(
        self,
        inputs: dict[str, Any],
        *,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backbone is None:
            raise ValueError("backbone is required for live Qwen frame representation extraction")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("attention_mask is required for last non-padding pooling")
        kwargs = dict(inputs)
        embedded_anchor_mask = kwargs.pop("anchor_mask", None)
        if embedded_anchor_mask is not None:
            if anchor_mask is not None:
                raise RuntimeError("anchor_mask was supplied both inside inputs and as an explicit argument")
            anchor_mask = embedded_anchor_mask
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        kwargs["use_cache"] = False
        with torch.set_grad_enabled(self.backbone_trainable and self.training):
            outputs = self.backbone(**kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        last_hidden = hidden_states[-1] if hidden_states is not None else getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("Backbone output does not contain hidden states")
        if self.pooling_mode == ANCHOR_POOLING_MODE:
            if anchor_mask is None:
                raise RuntimeError("anchor_span_mean pooling requires an explicit anchor_mask")
            pooled = self.anchor_pooler(last_hidden, anchor_mask)
        else:
            if anchor_mask is not None:
                raise RuntimeError("last_non_padding pooling must not receive anchor_mask")
            pooled = self.pooler(last_hidden, attention_mask.to(last_hidden.device))
        del outputs, hidden_states, last_hidden
        return pooled

    def extract_frame_representations(
        self,
        inputs: dict[str, Any],
        *,
        batch_size: int,
        anchor_mask: torch.Tensor | None = None,
        frame_chunk_size: int | None = None,
    ) -> torch.Tensor:
        chunk_size = normalize_frame_chunk_size(frame_chunk_size)
        expected = int(batch_size) * 4
        if chunk_size in {None, 4}:
            pooled = self._extract_pooled_rows(inputs, anchor_mask=anchor_mask)
        else:
            if int(batch_size) != 1:
                raise RuntimeError(
                    "frame_chunk_size=1/2 currently requires evaluation batch_size=1 to preserve "
                    "the exact four-frame sample boundary"
                )
            if anchor_mask is not None and tuple(anchor_mask.shape[:1]) != (expected,):
                raise RuntimeError(
                    f"anchor_mask must contain {expected} frame rows, got {tuple(anchor_mask.shape)}"
                )
            chunks: list[torch.Tensor] = []
            for start in range(0, expected, chunk_size):
                end = min(start + chunk_size, expected)
                chunk_inputs = slice_frame_multimodal_inputs(
                    inputs,
                    start=start,
                    end=end,
                    total_frames=expected,
                )
                chunk_anchor = None if anchor_mask is None else anchor_mask[start:end]
                chunks.append(self._extract_pooled_rows(chunk_inputs, anchor_mask=chunk_anchor))
            pooled = torch.cat(chunks, dim=0)
        if pooled.shape[0] != expected:
            raise ValueError(f"Expected B*4={expected} frame representations, got {pooled.shape[0]}")
        return pooled.reshape(int(batch_size), 4, pooled.shape[-1])

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
    architecture_id = str(get_by_path(cfg, "architecture.id", ""))
    model_class: type[Qwen3VLStagePairModel] = Qwen3VLStagePairModel
    if architecture_id == "qwen35_27b_stage_pair_e1_int4_v1":
        from .qwen35_27b_port import Qwen35_27BStagePairE1Model

        model_class = Qwen35_27BStagePairE1Model
    return model_class(
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
        pooling_mode=StagePairPromptSpec.from_config(cfg).pooling_mode,
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


def build_stage_pair_model_from_config(
    cfg: dict[str, Any],
    *,
    live_backbone: bool,
    processor_path: str | Path | None = None,
) -> tuple[Qwen3VLStagePairModel, Any | None]:
    if not live_backbone:
        configured_hidden = get_by_path(cfg, "cache.hidden_size", get_by_path(cfg, "backbone.hidden_size", None))
        if configured_hidden is None:
            raise RuntimeError("Head-only construction requires an explicit verified hidden size")
        hidden_size = int(configured_hidden)
        return build_stage_pair_head_from_config(cfg, hidden_size=hidden_size, backbone=None), None

    processor_model_cfg = {
        "local_dir": (
            str(processor_path)
            if processor_path is not None
            else get_by_path(cfg, "backbone.base_model_path")
        ),
        "local_files_only": bool(get_by_path(cfg, "backbone.local_files_only", True)),
        "trust_remote_code": bool(get_by_path(cfg, "backbone.trust_remote_code", True)),
        "type": get_by_path(cfg, "backbone.model_type", "qwen3_vl"),
    }
    if processor_path is None and get_by_path(cfg, "backbone.revision", None):
        processor_model_cfg["revision"] = get_by_path(cfg, "backbone.revision")
    processor = load_qwen3_processor({"model": processor_model_cfg, "processor": cfg.get("processor", {})})
    backbone_cfg = {
        "architecture": cfg.get("architecture", {}),
        "model": {
            "local_dir": get_by_path(cfg, "backbone.base_model_path", "Qwen/Qwen3-VL-8B-Instruct"),
            "local_files_only": bool(get_by_path(cfg, "backbone.local_files_only", True)),
            "trust_remote_code": bool(get_by_path(cfg, "backbone.trust_remote_code", True)),
            "type": get_by_path(cfg, "backbone.model_type", "qwen3_vl"),
            "torch_dtype": get_by_path(cfg, "backbone.torch_dtype", "bfloat16"),
            "device_map": get_by_path(cfg, "backbone.device_map", "auto"),
            "freeze_vision_encoder": True,
            "disable_deepstack_visual_features": bool(get_by_path(cfg, "backbone.disable_deepstack_visual_features", True)),
        },
        "attention": cfg.get("attention", {}),
        "quantization": cfg.get("quantization", {}),
        "train": cfg.get("train", {}),
        "lora": cfg.get("lora", {}),
        "vision_merger_lora": cfg.get("vision_merger_lora", {}),
        "runtime": cfg.get("runtime", {}),
    }
    if get_by_path(cfg, "backbone.revision", None):
        backbone_cfg["model"]["revision"] = get_by_path(cfg, "backbone.revision")
    backbone = load_qwen3_backbone(backbone_cfg)
    if str(get_by_path(cfg, "architecture.id", "")) == "qwen35_27b_stage_pair_e1_int4_v1":
        from .qwen35_27b_port import validate_qwen35_27b_architecture

        validate_qwen35_27b_architecture(backbone.config)
    if bool(get_by_path(cfg, "runtime.require_no_cpu_disk_offload", False)):
        backbone._stage_pair_device_report = assert_no_cpu_disk_offload(backbone)
    source = str(get_by_path(cfg, "backbone.source", "base"))
    frozen = bool(get_by_path(cfg, "backbone.frozen", True))
    if source == "existing_lora":
        backbone = _load_existing_adapter(backbone, cfg, is_trainable=not frozen)
    elif source == "base" and (
        not frozen
        or (
            int(get_by_path(cfg, "checkpoint.format_version", 1)) in {2, 3}
            and bool(get_by_path(cfg, "lora.enabled", False))
        )
    ):
        backbone = apply_lora_if_enabled(backbone, backbone_cfg)
    elif source != "base":
        raise ValueError(f"Unsupported backbone.source: {source}")
    if frozen:
        for param in backbone.parameters():
            param.requires_grad = False
    hidden_size = _hidden_size_from_config(backbone.config)
    model = build_stage_pair_head_from_config(cfg, hidden_size=hidden_size, backbone=backbone)
    return model, processor


def write_stage_pair_trainable_report(path: str | Path, model: nn.Module) -> dict[str, Any]:
    raw = unwrap_model(model)
    report = trainable_parameter_report(raw)
    backbone = getattr(raw, "backbone", None)
    manifest = getattr(backbone, "_stage_pair_lora_target_manifest", None)
    summary = lora_target_summary(manifest) if isinstance(manifest, list) else {
        "groups": {},
        "text_full_attention_match_count": 0,
        "text_linear_attention_match_count": 0,
        "vision_match_count": 0,
        "lora_trainable_parameter_count": 0,
    }
    visual_base_parameters = 0
    vision_trainable_parameters = 0
    base_model_parameters = 0
    backbone_parameters = 0
    if backbone is not None:
        for name, parameter in backbone.named_parameters():
            count = int(parameter.numel())
            lowered = name.lower()
            backbone_parameters += count
            if "lora_" not in lowered:
                base_model_parameters += count
            if any(marker in lowered for marker in ("visual", "vision", "image")):
                if "lora_" in lowered and parameter.requires_grad:
                    vision_trainable_parameters += count
                elif "lora_" not in lowered:
                    visual_base_parameters += count
                    if parameter.requires_grad:
                        raise RuntimeError(f"Visual base parameter is unexpectedly trainable: {name}")
    report.update(summary)
    report.update(
        {
            "base_model_total_parameters": base_model_parameters,
            "backbone_parameter_total": backbone_parameters,
            "trainable_parameter_total": int(report["trainable"]),
            "frozen_visual_base_parameter_count": visual_base_parameters,
            "vision_trainable_parameter_count": vision_trainable_parameters,
            "actual_device_map": getattr(backbone, "_stage_pair_device_report", {}).get("actual_device_map", {}),
            "parameter_devices": getattr(backbone, "_stage_pair_device_report", {}).get("parameter_devices", []),
            "cpu_or_disk_offload": getattr(backbone, "_stage_pair_device_report", {}).get(
                "cpu_or_disk_offload", False
            ),
        }
    )
    write_json(path, report)
    return report
