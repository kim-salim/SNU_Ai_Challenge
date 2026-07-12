from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from snu_order.qwen3vl.modeling_lora24 import Qwen3VL24WayClassifier, last_non_padding_indices, trainable_parameter_report
from snu_order.qwen3vl.permutations import perm_index_to_answer


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, input_ids, attention_mask, **kwargs):
        x = input_ids.float().unsqueeze(-1)
        hidden = self.proj(x)
        return SimpleNamespace(hidden_states=(hidden,))


def test_last_non_padding_pooling_left_and_right_padding():
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    left = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    assert last_non_padding_indices(right).tolist() == [1, 2]
    assert last_non_padding_indices(left).tolist() == [3, 3]


def test_classifier_output_shape():
    model = Qwen3VL24WayClassifier(FakeBackbone(), hidden_size=8, backbone_trainable=False)
    logits = model(input_ids=torch.ones((2, 5), dtype=torch.long), attention_mask=torch.ones((2, 5), dtype=torch.long))
    assert tuple(logits.shape) == (2, 24)


def test_trainable_parameters_can_be_limited_to_classifier():
    backbone = FakeBackbone()
    for param in backbone.parameters():
        param.requires_grad = False
    model = Qwen3VL24WayClassifier(backbone, hidden_size=8, backbone_trainable=False)
    report = trainable_parameter_report(model)
    assert report["trainable"] > 0
    assert all(name.startswith("classifier.") for name in report["trainable_names"])


def test_argmax_class_converts_to_valid_answer():
    logits = torch.zeros((1, 24))
    logits[0, 17] = 10
    answer = perm_index_to_answer(int(logits.argmax(dim=1).item()))
    assert sorted(answer) == [1, 2, 3, 4]
