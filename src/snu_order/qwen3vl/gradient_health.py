from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from snu_order.utils.io import write_json

from .lora_targets import TEXT_FULL_GROUP, TEXT_LINEAR_GROUP, VISION_MERGER_GROUP


HEALTH_GROUPS = (
    "lora.full_attention",
    "lora.linear_attention",
    "lora.vision_merger",
    "head.set",
    "head.stage",
    "head.pair",
    "vision.base",
)

_MANIFEST_TO_HEALTH = {
    TEXT_FULL_GROUP: "lora.full_attention",
    TEXT_LINEAR_GROUP: "lora.linear_attention",
    VISION_MERGER_GROUP: "lora.vision_merger",
}


def _is_visual_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in (".visual.", ".vision_", "vision_tower"))


class GradientHealthMonitor:
    def __init__(self, model: nn.Module, output_path: str | Path) -> None:
        self.model = model
        self.output_path = Path(output_path)
        backbone = getattr(model, "backbone", None)
        manifest = getattr(backbone, "_stage_pair_lora_target_manifest", None)
        if backbone is None or not isinstance(manifest, list):
            raise RuntimeError("Gradient health requires a finalized structured LoRA manifest")
        named = dict(model.named_parameters())
        groups: dict[str, list[tuple[str, nn.Parameter]]] = {group: [] for group in HEALTH_GROUPS}
        for entry in manifest:
            health_group = _MANIFEST_TO_HEALTH.get(str(entry.get("group")))
            if health_group is None:
                raise RuntimeError(f"Unknown LoRA group in gradient health manifest: {entry.get('group')}")
            raw_name = str(entry["module_name"])
            matches = [
                (name, parameter)
                for name, parameter in named.items()
                if raw_name in name and (".lora_A." in name or ".lora_B." in name)
            ]
            if len(matches) != 2:
                raise RuntimeError(f"Expected two LoRA tensors for {raw_name}, found {[name for name, _ in matches]}")
            groups[health_group].extend(matches)
        for name, parameter in named.items():
            if name.startswith("set_encoder."):
                groups["head.set"].append((name, parameter))
            elif name.startswith("stage_head."):
                groups["head.stage"].append((name, parameter))
            elif name.startswith("pair_head."):
                groups["head.pair"].append((name, parameter))
            elif name.startswith("backbone.") and _is_visual_name(name) and "lora_" not in name:
                groups["vision.base"].append((name, parameter))
        missing = [group for group in HEALTH_GROUPS if not groups[group]]
        if missing:
            raise RuntimeError(f"Gradient health groups are empty: {missing}")
        trainable_vision_base = [name for name, parameter in groups["vision.base"] if parameter.requires_grad]
        if trainable_vision_base:
            raise RuntimeError(f"Vision base parameters must be frozen: {trainable_vision_base[:8]}")
        self.groups = groups
        representative_names: set[str] = set()
        for group, values in groups.items():
            representative_names.add(values[0][0])
            if group.startswith("lora."):
                for side in (".lora_A.", ".lora_B."):
                    match = next((name for name, _ in values if side in name), None)
                    if match is not None:
                        representative_names.add(match)
        self.representative_names = sorted(representative_names)
        self.initial = {
            name: named[name].detach().float().cpu().clone() for name in self.representative_names
        }
        self.steps: list[dict[str, Any]] = []
        self._write(status="IN_PROGRESS")

    def _parameter_report(self, values: list[tuple[str, nn.Parameter]]) -> dict[str, Any]:
        trainable_numel = sum(int(parameter.numel()) for _, parameter in values if parameter.requires_grad)
        no_grad_numel = 0
        finite_nonzero_numel = 0
        finite_zero_numel = 0
        nonfinite_numel = 0
        norm_squared = 0.0
        for _, parameter in values:
            gradient = parameter.grad
            if gradient is None:
                no_grad_numel += int(parameter.numel())
                continue
            grad = gradient.detach().float()
            finite = torch.isfinite(grad)
            nonfinite_numel += int((~finite).sum().item())
            finite_values = grad[finite]
            finite_nonzero_numel += int(finite_values.ne(0).sum().item())
            finite_zero_numel += int(finite_values.eq(0).sum().item())
            norm_squared += float(finite_values.square().sum().item())
        return {
            "expected_tensor_count": len(values),
            "discovered_tensor_count": len(values),
            "trainable_numel": trainable_numel,
            "no_grad_numel": no_grad_numel,
            "finite_nonzero_numel": finite_nonzero_numel,
            "finite_zero_numel": finite_zero_numel,
            "nonfinite_numel": nonfinite_numel,
            "gradient_norm": math.sqrt(norm_squared),
        }

    def capture_before_step(self, completed_step: int) -> dict[str, Any]:
        if completed_step not in {1, 2}:
            return {}
        reports = {group: self._parameter_report(values) for group, values in self.groups.items()}
        for group, report in reports.items():
            if report["nonfinite_numel"]:
                raise RuntimeError(f"Non-finite gradient detected in {group}: {report}")
            if group != "vision.base" and report["finite_nonzero_numel"] <= 0:
                raise RuntimeError(f"No finite nonzero gradient detected in {group}: {report}")
        if reports["vision.base"]["no_grad_numel"] != sum(
            int(parameter.numel()) for _, parameter in self.groups["vision.base"]
        ):
            raise RuntimeError("Frozen vision base unexpectedly received a gradient")
        named = dict(self.model.named_parameters())
        return {
            "completed_optimizer_step": completed_step,
            "groups": reports,
            "pre_step": {
                name: named[name].detach().float().cpu().clone() for name in self.representative_names
            },
        }

    def capture_after_step(self, pending: dict[str, Any]) -> None:
        if not pending:
            return
        named = dict(self.model.named_parameters())
        representative_deltas: dict[str, dict[str, float]] = {}
        for name in self.representative_names:
            current = named[name].detach().float().cpu()
            representative_deltas[name] = {
                "step_max_abs_delta": float((current - pending["pre_step"][name]).abs().max().item()),
                "cumulative_max_abs_delta": float((current - self.initial[name]).abs().max().item()),
            }
        pending = {key: value for key, value in pending.items() if key != "pre_step"}
        pending["representative_weight_delta"] = representative_deltas
        step = int(pending["completed_optimizer_step"])
        merger_b = [
            delta["cumulative_max_abs_delta"]
            for name, delta in representative_deltas.items()
            if ".visual.merger." in name and ".lora_B." in name
        ]
        merger_a = [
            delta["cumulative_max_abs_delta"]
            for name, delta in representative_deltas.items()
            if ".visual.merger." in name and ".lora_A." in name
        ]
        vision_base = [
            delta["cumulative_max_abs_delta"]
            for name, delta in representative_deltas.items()
            if _is_visual_name(name) and "lora_" not in name
        ]
        if not merger_b or max(merger_b) <= 0:
            raise RuntimeError("Vision merger lora_B did not update on a completed optimizer step")
        if step >= 2 and (not merger_a or max(merger_a) <= 0):
            raise RuntimeError("Vision merger lora_A did not update by the second completed optimizer step")
        if any(value != 0 for value in vision_base):
            raise RuntimeError(f"Frozen vision base weight changed: {vision_base}")
        self.steps.append(pending)
        self._write(status="PASS" if len(self.steps) >= 2 else "IN_PROGRESS")

    def assert_complete(self) -> None:
        if len(self.steps) < 2:
            raise RuntimeError(f"Gradient health captured only {len(self.steps)} completed optimizer steps")
        self._write(status="PASS")

    def _write(self, *, status: str) -> None:
        payload = {
            "status": status,
            "required_completed_optimizer_steps": 2,
            "captured_completed_optimizer_steps": len(self.steps),
            "steps": self.steps,
        }
        write_json(self.output_path, json.loads(json.dumps(payload, allow_nan=False)))
