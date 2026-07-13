from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from snu_order.qwen3vl.lora_targets import (
    TEXT_FULL_GROUP,
    TEXT_LINEAR_GROUP,
    VISION_MERGER_GROUP,
    discover_qwen35_lora_targets,
    enforce_lora_trainability,
    finalize_peft_lora_manifest,
)


class FullLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = nn.Linear(4, 2, bias=False)
        self.self_attn.v_proj = nn.Linear(4, 2, bias=False)
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)


class LinearLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_qkv = nn.Linear(4, 8, bias=False)
        self.linear_attn.in_proj_z = nn.Linear(4, 4, bias=False)
        self.linear_attn.out_proj = nn.Linear(4, 4, bias=False)
        self.out_proj = nn.Linear(4, 4, bias=False)


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        layer_types = ["full_attention" if index % 4 == 3 else "linear_attention" for index in range(32)]
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(layer_types=layer_types, num_hidden_layers=32)
        )
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [FullLayer() if layer_type == "full_attention" else LinearLayer() for layer_type in layer_types]
        )
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Module()
        self.model.visual.merger.linear_fc1 = nn.Linear(4, 4, bias=False)
        self.model.visual.merger.linear_fc2 = nn.Linear(4, 4, bias=False)
        block = nn.Module()
        block.mlp = nn.Module()
        block.mlp.linear_fc1 = nn.Linear(4, 4, bias=False)
        block.mlp.linear_fc2 = nn.Linear(4, 4, bias=False)
        self.model.visual.blocks = nn.ModuleList([block])
        self.model.unrelated_out_proj = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 10, bias=False)


def _cfg(merger_enabled=False):
    return {
        "lora": {
            "full_attention": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "linear_attention": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "modules": ["in_proj_qkv", "in_proj_z", "out_proj"],
            },
        },
        "vision_merger_lora": {
            "enabled": merger_enabled,
            "rank": 4,
            "alpha": 8,
            "dropout": 0.05,
            "target_suffixes": ["visual.merger.linear_fc1", "visual.merger.linear_fc2"],
        },
    }


def test_exact_text_target_plan_has_expected_counts_and_rank_patterns():
    manifest = discover_qwen35_lora_targets(FakeQwen(), _cfg(False))
    full = [entry for entry in manifest if entry["group"] == TEXT_FULL_GROUP]
    linear = [entry for entry in manifest if entry["group"] == TEXT_LINEAR_GROUP]
    assert len(full) == 32
    assert len(linear) == 72
    assert len(manifest) == 104
    assert {entry["layer_index"] for entry in manifest} == set(range(32))
    assert all(entry["rank"] == 16 and entry["alpha"] == 32 for entry in full)
    assert all(entry["rank"] == 8 and entry["alpha"] == 16 for entry in linear)
    assert not any("visual" in entry["module_name"] or "lm_head" in entry["module_name"] for entry in manifest)
    assert not any("unrelated_out_proj" in entry["module_name"] for entry in manifest)
    assert not any(".self_attn." in entry["module_name"] for entry in linear)
    assert not any(".linear_attn." in entry["module_name"] for entry in full)
    rank_pattern = {entry["module_name"]: entry["rank"] for entry in manifest}
    assert len(rank_pattern) == 104


def test_target_count_mismatch_fails_with_diagnostics():
    model = FakeQwen()
    del model.model.language_model.layers[0].linear_attn.in_proj_z
    with pytest.raises(RuntimeError, match="(?s)actual_matches.*missing_expected_group"):
        discover_qwen35_lora_targets(model, _cfg(False))


def _attach_fake_lora(model, manifest):
    modules = dict(model.named_modules())
    for entry in manifest:
        module = modules[entry["module_name"]]
        rank = int(entry["rank"])
        module.lora_A = nn.ModuleDict({"default": nn.Linear(module.in_features, rank, bias=False)})
        module.lora_B = nn.ModuleDict({"default": nn.Linear(rank, module.out_features, bias=False)})
        module.lora_alpha = {"default": int(entry["alpha"])}


def test_vision_merger_disabled_has_zero_vision_lora_trainable_parameters():
    model = FakeQwen()
    manifest = discover_qwen35_lora_targets(model, _cfg(False))
    _attach_fake_lora(model, manifest)
    enforce_lora_trainability(model, manifest)
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "visual" in name
    )


def test_vision_merger_enabled_matches_only_two_merger_layers_and_freezes_bases():
    model = FakeQwen()
    manifest = discover_qwen35_lora_targets(model, _cfg(True))
    merger = [entry for entry in manifest if entry["group"] == VISION_MERGER_GROUP]
    assert len(merger) == 2
    assert all(".visual.merger." in entry["module_name"] for entry in merger)
    assert not any("visual.blocks" in entry["module_name"] for entry in manifest)
    _attach_fake_lora(model, manifest)
    enforce_lora_trainability(model, manifest)
    finalized, summary = finalize_peft_lora_manifest(model, manifest)
    assert len(finalized) == 106
    assert summary["vision_match_count"] == 2
    for name, parameter in model.named_parameters():
        if ".visual.merger." in name and "lora_" in name:
            assert parameter.requires_grad
        elif ".visual." in name:
            assert not parameter.requires_grad
