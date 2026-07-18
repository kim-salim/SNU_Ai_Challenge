from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from snu_order.qwen3vl.lora_targets import (
    discover_qwen35_lora_targets,
    validate_gradient_contract,
)
from snu_order.qwen3vl.modeling_stage_pair import (
    AntiSymmetricPairwiseHead,
    build_stage_pair_head_from_config,
)
from snu_order.qwen3vl.permutations import PERMS
from snu_order.qwen3vl.qwen35_27b_port import (
    ARCHITECTURE_ID,
    FrameProjector,
    assert_27b_cache_compatible,
    build_strict_nf4_config,
    hidden_only_multimodal_forward,
    validate_qwen35_27b_architecture,
)
from snu_order.qwen3vl.stage_pair_scorer import structured_permutation_logits
from snu_order.qwen3vl.stage_pair_checkpoint_v3 import _assert_port_runtime_contract
from snu_order.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/exp/qwen35_27b_stage_pair_e1_int4_champion_port_v1.yaml"


def _config() -> dict:
    return load_config(CONFIG)


def _fake_hf_config(layer_types: list[str] | None = None, *, hidden_size: int = 5120):
    values = layer_types or ["linear_attention"] * 48 + ["full_attention"] * 16
    text = SimpleNamespace(
        hidden_size=hidden_size,
        num_hidden_layers=len(values),
        layer_types=values,
    )
    return SimpleNamespace(
        model_type="qwen3_5",
        text_config=text,
        vision_config=SimpleNamespace(model_type="qwen3_5_vision"),
        to_dict=lambda: {
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": hidden_size,
                "num_hidden_layers": len(values),
                "layer_types": values,
            },
            "vision_config": {"model_type": "qwen3_5_vision"},
        },
    )


class _FullLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(4, 4, bias=False))


class _LinearLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_attn = nn.Module()
        for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
            setattr(self.linear_attn, name, nn.Linear(4, 4, bias=False))


class _FakeBackbone(nn.Module):
    def __init__(self, layer_types: list[str]) -> None:
        super().__init__()
        self.config = _fake_hf_config(layer_types)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [_FullLayer() if value == "full_attention" else _LinearLayer() for value in layer_types]
        )


def test_config_uses_27b_and_verified_90_10_contract() -> None:
    cfg = _config()
    assert cfg["architecture"]["id"] == ARCHITECTURE_ID
    assert cfg["backbone"]["hidden_size"] == 5120
    assert cfg["data"]["split_contract"] == "full_train_90_10_v1"
    assert cfg["data"]["train_split"].endswith("full_train_90_10_v1/train_90_v1.csv")
    assert cfg["data"]["valid_split"].endswith("full_train_90_10_v1/valid_10_v1.csv")


def test_qwen35_27b_architecture_exact_contract() -> None:
    report = validate_qwen35_27b_architecture(_fake_hf_config())
    assert report["hidden_size"] == 5120
    assert report["layer_type_counts"] == {"linear_attention": 48, "full_attention": 16}
    with pytest.raises(RuntimeError, match="HOLD_QWEN_ARCH_UNSUPPORTED"):
        validate_qwen35_27b_architecture(_fake_hf_config(hidden_size=4096))


def test_projector_shape_and_4096_cache_rejection() -> None:
    projector = FrameProjector(5120, 512)
    assert projector(torch.randn(2, 4, 5120)).shape == (2, 4, 512)
    with pytest.raises(RuntimeError, match="cache width mismatch"):
        projector(torch.randn(2, 4, 4096))
    with pytest.raises(RuntimeError, match="cache width mismatch"):
        assert_27b_cache_compatible({"frame_hidden": torch.randn(2, 4, 4096)})


def test_nf4_bitsandbytes_config_is_exact_and_fail_closed() -> None:
    cfg = _config()
    quant = build_strict_nf4_config(cfg)
    assert quant.load_in_4bit is True
    assert quant.bnb_4bit_quant_type == "nf4"
    assert quant.bnb_4bit_use_double_quant is True
    assert quant.bnb_4bit_compute_dtype is torch.bfloat16
    changed = json.loads(json.dumps(cfg))
    changed["quantization"]["double_quant"] = False
    with pytest.raises(RuntimeError, match="HOLD_INT4_BACKEND_INCOMPATIBLE"):
        build_strict_nf4_config(changed)


def test_lora_manifest_has_16_attention_and_48_deltanet_layers() -> None:
    layer_types = ["linear_attention"] * 48 + ["full_attention"] * 16
    manifest = discover_qwen35_lora_targets(_FakeBackbone(layer_types), _config())
    full = [entry for entry in manifest if entry["group"] == "text_full_attention"]
    linear = [entry for entry in manifest if entry["group"] == "text_linear_attention"]
    assert len(full) == 64
    assert len(linear) == 144
    assert {entry["rank"] for entry in full} == {16}
    assert {entry["rank"] for entry in linear} == {8}
    assert not any("mlp" in entry["module_name"] for entry in manifest)
    assert not any("visual" in entry["module_name"] for entry in manifest)


def test_wrong_layer_count_is_rejected() -> None:
    cfg = _config()
    with pytest.raises(RuntimeError, match="Expected 64 language layers"):
        discover_qwen35_lora_targets(_FakeBackbone(["linear_attention"] * 47 + ["full_attention"] * 16), cfg)


def test_hidden_only_path_never_calls_lm_head_and_disables_cache() -> None:
    class Core(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kwargs = None

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                last_hidden_state=torch.zeros(1, 3, 5120),
                past_key_values=None,
            )

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def forward(self, value):
            self.called = True
            return value

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Core()
            self.lm_head = Head()

    wrapper = Wrapper()
    hidden, contract = hidden_only_multimodal_forward(wrapper, {"input_ids": torch.ones(1, 3)})
    assert hidden.shape == (1, 3, 5120)
    assert wrapper.lm_head.called is False
    assert wrapper.model.kwargs["use_cache"] is False
    assert wrapper.model.kwargs["output_hidden_states"] is False
    assert contract["lm_head_called"] is False


def test_gradient_contract_includes_fresh_frame_projector() -> None:
    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_a = nn.Parameter(torch.ones(2, 2))
            self.lora_b = nn.Parameter(torch.ones(2, 2))
            self._stage_pair_lora_target_manifest = [
                {
                    "module_name": "model.language_model.layers.0.self_attn.q_proj",
                    "group": "text_full_attention",
                }
            ]

        def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
            del prefix, recurse, remove_duplicate
            yield (
                "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
                self.lora_a,
            )
            yield (
                "model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight",
                self.lora_b,
            )

    class StagePair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Backbone()
            self.frame_projector = nn.Linear(2, 2)
            self.set_encoder = nn.Linear(2, 2)
            self.stage_head = nn.Linear(2, 2)
            self.pair_head = nn.Linear(2, 2)

    model = StagePair()
    for parameter in model.parameters():
        parameter.requires_grad = True
        parameter.grad = torch.ones_like(parameter)
    report = validate_gradient_contract(model)
    expected_head_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    )
    assert report["heads"] == expected_head_parameters
    model.frame_projector.weight.grad = None
    with pytest.raises(RuntimeError, match="Missing head gradient: frame_projector.weight"):
        validate_gradient_contract(model)


def test_stage_pair_semantics_and_equivariance_are_preserved() -> None:
    cfg = _config()
    model = build_stage_pair_head_from_config(cfg, hidden_size=5120, backbone=None).eval()
    frame_hidden = torch.randn(2, 4, 5120)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.inference_mode():
        original = model(frame_hidden=frame_hidden)
        shuffled = model(frame_hidden=frame_hidden[:, permutation])
    assert torch.allclose(original["contextual"][:, permutation], shuffled["contextual"], atol=1e-5)
    pair = AntiSymmetricPairwiseHead(512, 512, dropout=0.0).eval()
    directional = pair.directional_logits(torch.randn(2, 4, 512))
    assert torch.allclose(directional, -directional.transpose(1, 2), atol=1e-6)
    reconstructed = structured_permutation_logits(
        original["stage_logits"], original["pair_logits"], stage_weight=1.0, pair_weight=0.3
    )
    assert torch.equal(original["final_logits"], reconstructed)
    assert len(PERMS) == 24


def test_legacy_9b_head_builder_keeps_state_dict_keys() -> None:
    cfg = load_config(ROOT / "configs/exp/qwen35_9b_stage_pair_v2_text_anchor.yaml")
    model = build_stage_pair_head_from_config(cfg, hidden_size=4096, backbone=None)
    assert not hasattr(model, "frame_projector")
    assert "set_encoder.proj.weight" in model.state_dict()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("architecture", "id"), "wrong_architecture"),
        (("backbone", "hidden_size"), 4096),
        (("quantization", "quant_type"), "fp4"),
        (("prompt", "anchor_text"), "OTHER:"),
        (("data", "image_policy"), "dynamic_resize"),
        (("score", "pair_weight"), 0.7),
    ],
)
def test_v3_checkpoint_runtime_contract_rejects_semantic_mismatch(path, value) -> None:
    saved = _config()
    runtime = json.loads(json.dumps(saved))
    cursor = runtime
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(RuntimeError, match="checkpoint/runtime contract mismatch"):
        _assert_port_runtime_contract(saved, runtime)
