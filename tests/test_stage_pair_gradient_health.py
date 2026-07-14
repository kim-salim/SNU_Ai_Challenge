import json

import torch
from torch import nn

from snu_order.qwen3vl.gradient_health import GradientHealthMonitor
from snu_order.qwen3vl.lora_targets import TEXT_FULL_GROUP, TEXT_LINEAR_GROUP, VISION_MERGER_GROUP


class _LoraLinear(nn.Module):
    def __init__(self, size: int = 4, rank: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(size), requires_grad=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(size, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, size, bias=False)})
        nn.init.kaiming_uniform_(self.lora_A["default"].weight)
        nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight.T + self.lora_B["default"](self.lora_A["default"](value))


class _FakeHealthModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.model = nn.Module()
        self.backbone.model.language_model = nn.Module()
        full = nn.Module()
        full.self_attn = nn.Module()
        full.self_attn.q_proj = _LoraLinear()
        linear = nn.Module()
        linear.linear_attn = nn.Module()
        linear.linear_attn.out_proj = _LoraLinear()
        self.backbone.model.language_model.layers = nn.ModuleList([full, linear])
        self.backbone.model.visual = nn.Module()
        self.backbone.model.visual.merger = nn.Module()
        self.backbone.model.visual.merger.linear_fc1 = _LoraLinear()
        self.backbone._stage_pair_lora_target_manifest = [
            {
                "module_name": "model.language_model.layers.0.self_attn.q_proj",
                "group": TEXT_FULL_GROUP,
            },
            {
                "module_name": "model.language_model.layers.1.linear_attn.out_proj",
                "group": TEXT_LINEAR_GROUP,
            },
            {
                "module_name": "model.visual.merger.linear_fc1",
                "group": VISION_MERGER_GROUP,
            },
        ]
        self.set_encoder = nn.Linear(4, 4)
        self.stage_head = nn.Linear(4, 4)
        self.pair_head = nn.Linear(4, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        layers = self.backbone.model.language_model.layers
        value = layers[0].self_attn.q_proj(value)
        value = layers[1].linear_attn.out_proj(value)
        value = self.backbone.model.visual.merger.linear_fc1(value)
        value = torch.tanh(self.set_encoder(value))
        return self.stage_head(value).sum() + self.pair_head(value).sum()


def test_gradient_health_confirms_merger_b_then_a_updates_and_frozen_vision_base(tmp_path):
    torch.manual_seed(42)
    model = _FakeHealthModel()
    output = tmp_path / "gradient_health.json"
    monitor = GradientHealthMonitor(model, output)
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
    for completed_step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(value).square()
        loss.backward()
        pending = monitor.capture_before_step(completed_step)
        optimizer.step()
        monitor.capture_after_step(pending)
    monitor.assert_complete()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["captured_completed_optimizer_steps"] == 2
    step_one = payload["steps"][0]["representative_weight_delta"]
    step_two = payload["steps"][1]["representative_weight_delta"]
    merger_b = [value for name, value in step_one.items() if ".visual.merger." in name and ".lora_B." in name]
    merger_a = [value for name, value in step_two.items() if ".visual.merger." in name and ".lora_A." in name]
    vision_base = [value for name, value in step_two.items() if ".visual." in name and "lora_" not in name]
    assert max(value["cumulative_max_abs_delta"] for value in merger_b) > 0
    assert max(value["cumulative_max_abs_delta"] for value in merger_a) > 0
    assert all(value["cumulative_max_abs_delta"] == 0 for value in vision_base)
