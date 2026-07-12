from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from snu_order.qwen3vl.checkpoint import load_classifier_weights, save_lora24_checkpoint
from snu_order.qwen3vl.modeling_lora24 import Qwen3VL24WayClassifier


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.proj = nn.Linear(1, 4)

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.proj(input_ids.float().unsqueeze(-1))
        return SimpleNamespace(hidden_states=(hidden,))


def test_checkpoint_save_load_round_trip(tmp_path):
    model = Qwen3VL24WayClassifier(TinyBackbone(), hidden_size=4, backbone_trainable=False)
    with torch.no_grad():
        model.classifier.linear.bias.fill_(3.0)
    save_lora24_checkpoint(tmp_path / "ckpt", model, None, {"x": 1}, {"valid_a_exact_match": 0.5}, minimal=True)
    fresh = Qwen3VL24WayClassifier(TinyBackbone(), hidden_size=4, backbone_trainable=False)
    load_classifier_weights(tmp_path / "ckpt", fresh)
    assert torch.allclose(fresh.classifier.linear.bias, torch.full_like(fresh.classifier.linear.bias, 3.0))
