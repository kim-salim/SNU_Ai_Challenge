from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from snu_order.utils.config import get_by_path

from .permutations import PAIRS, PERMS
from .stage_pair_scorer import structured_permutation_logits


CACHE_FORMAT = "qwen35_e1_champion_teacher_v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_shuffle(shuffle_idx: Sequence[int] | None) -> tuple[int, int, int, int]:
    if shuffle_idx is None:
        return (0, 1, 2, 3)
    values = tuple(int(value) for value in shuffle_idx)
    if len(values) != 4 or sorted(values) != [0, 1, 2, 3]:
        raise ValueError(f"Invalid frame shuffle: {values}")
    return values


def remap_teacher_logits_for_input_shuffle(
    stage_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    shuffle_idx: Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express canonical teacher decisions in the current shuffled input slots."""
    shuffle = _validate_shuffle(shuffle_idx)
    if tuple(stage_logits.shape) != (4, 4):
        raise ValueError(f"teacher stage logits must be [4,4], got {tuple(stage_logits.shape)}")
    if tuple(pair_logits.shape) != (len(PAIRS),):
        raise ValueError(f"teacher pair logits must be [{len(PAIRS)}], got {tuple(pair_logits.shape)}")
    stage = stage_logits[list(shuffle)]
    pair_by_slots = {pair: pair_logits[index] for index, pair in enumerate(PAIRS)}
    remapped: list[torch.Tensor] = []
    for new_left, new_right in PAIRS:
        old_left = shuffle[new_left]
        old_right = shuffle[new_right]
        if old_left < old_right:
            remapped.append(pair_by_slots[(old_left, old_right)])
        else:
            remapped.append(-pair_by_slots[(old_right, old_left)])
    return stage, torch.stack(remapped)


class ChampionTeacherStore:
    def __init__(
        self,
        path: str | Path,
        *,
        expected_ids: Sequence[str],
        expected_split_sha256: str,
    ) -> None:
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        required = {
            "cache_format",
            "ids",
            "stage_logits",
            "pair_logits",
            "target_perm_idx",
            "answer",
            "teacher_correct",
            "identity",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise RuntimeError("Champion teacher cache schema mismatch")
        if payload["cache_format"] != CACHE_FORMAT:
            raise RuntimeError(f"Unsupported Champion teacher cache: {payload['cache_format']!r}")
        ids = [str(value) for value in payload["ids"]]
        expected = [str(value) for value in expected_ids]
        if ids != expected or len(ids) != len(set(ids)):
            raise RuntimeError("Champion teacher cache ID ordering does not match the training split")
        identity = payload["identity"]
        if not isinstance(identity, dict) or identity.get("train_split_sha256") != expected_split_sha256:
            raise RuntimeError("Champion teacher cache train-split binding mismatch")
        n = len(ids)
        stage = payload["stage_logits"].float().contiguous()
        pair = payload["pair_logits"].float().contiguous()
        answer = payload["answer"].long().contiguous()
        targets = payload["target_perm_idx"].long().contiguous()
        correct = payload["teacher_correct"].bool().contiguous()
        if tuple(stage.shape) != (n, 4, 4) or tuple(pair.shape) != (n, len(PAIRS)):
            raise RuntimeError("Champion teacher cache logit shape mismatch")
        if tuple(answer.shape) != (n, 4) or tuple(targets.shape) != (n,) or tuple(correct.shape) != (n,):
            raise RuntimeError("Champion teacher cache target shape mismatch")
        self.ids = ids
        self.stage_logits = stage
        self.pair_logits = pair
        self.answer = answer
        self.target_perm_idx = targets
        self.teacher_correct = correct
        self.identity = identity
        self._index = {sample_id: index for index, sample_id in enumerate(ids)}

    def sample(self, sample_id: str, shuffle_idx: Sequence[int] | None) -> dict[str, torch.Tensor]:
        index = self._index[str(sample_id)]
        stage, pair = remap_teacher_logits_for_input_shuffle(
            self.stage_logits[index], self.pair_logits[index], shuffle_idx
        )
        final = structured_permutation_logits(stage.unsqueeze(0), pair.unsqueeze(0), stage_weight=1.0, pair_weight=0.3)[0]
        return {
            "teacher_stage_logits": stage,
            "teacher_pair_logits": pair,
            "teacher_final_logits": final,
            "teacher_correct": self.teacher_correct[index],
        }


@dataclass(frozen=True)
class RetentionLossOutput:
    loss: torch.Tensor
    stage_kd: torch.Tensor
    pair_kd: torch.Tensor
    permutation_kd: torch.Tensor
    active_samples: int


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("masked retention loss expects one scalar per sample")
    denominator = weights.sum()
    if float(denominator.detach().cpu()) == 0.0:
        return values.new_zeros(())
    return (values * weights).sum() / denominator


def champion_retention_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> RetentionLossOutput:
    enabled = bool(get_by_path(cfg, "retention.enabled", False))
    zero = outputs["final_logits"].new_zeros(())
    if not enabled:
        return RetentionLossOutput(zero, zero, zero, zero, 0)
    required = ("teacher_stage_logits", "teacher_pair_logits", "teacher_final_logits", "teacher_correct")
    missing = [key for key in required if key not in batch]
    if missing:
        raise RuntimeError(f"Champion retention batch is missing: {missing}")
    temperature = float(get_by_path(cfg, "retention.temperature", 2.0))
    if temperature <= 0:
        raise ValueError("retention.temperature must be positive")
    mask = batch["teacher_correct"].bool().view(-1)
    stage_teacher = batch["teacher_stage_logits"].to(outputs["stage_logits"].dtype)
    pair_teacher = batch["teacher_pair_logits"].to(outputs["pair_logits"].dtype)
    final_teacher = batch["teacher_final_logits"].to(outputs["final_logits"].dtype)

    stage_per = F.kl_div(
        F.log_softmax(outputs["stage_logits"] / temperature, dim=-1),
        F.softmax(stage_teacher / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean(dim=-1) * temperature**2
    pair_per = F.binary_cross_entropy_with_logits(
        outputs["pair_logits"] / temperature,
        torch.sigmoid(pair_teacher / temperature),
        reduction="none",
    ).mean(dim=-1) * temperature**2
    perm_per = F.kl_div(
        F.log_softmax(outputs["final_logits"] / temperature, dim=-1),
        F.softmax(final_teacher / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * temperature**2
    stage_kd = _masked_mean(stage_per, mask)
    pair_kd = _masked_mean(pair_per, mask)
    permutation_kd = _masked_mean(perm_per, mask)
    loss = (
        float(get_by_path(cfg, "retention.stage_kd_weight", 0.10)) * stage_kd
        + float(get_by_path(cfg, "retention.pair_kd_weight", 0.05)) * pair_kd
        + float(get_by_path(cfg, "retention.permutation_kd_weight", 0.10)) * permutation_kd
    )
    return RetentionLossOutput(loss, stage_kd, pair_kd, permutation_kd, int(mask.sum().item()))


class ChampionRetentionLRScheduler:
    """Pre-registered projector alignment, head stabilization, then joint QLoRA."""

    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, cfg: dict[str, Any]) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.projector_end = max(1, int(math.ceil(self.total_steps * float(get_by_path(cfg, "retention.schedule.projector_only_fraction", 1.0 / 12.0)))))
        self.stabilization_end = max(
            self.projector_end + 1,
            int(math.ceil(self.total_steps * float(get_by_path(cfg, "retention.schedule.stabilization_end_fraction", 0.25)))),
        )
        self.stabilization_end = min(self.stabilization_end, self.total_steps - 1)
        self.joint_warmup_ratio = float(get_by_path(cfg, "retention.schedule.joint_warmup_ratio", 0.05))
        self.learning_rates = {
            "projector_only": {
                "backbone_lora": 0.0,
                "frame_projector": float(get_by_path(cfg, "retention.schedule.projector_only_lr", 3e-4)),
                "set_encoder": 0.0,
                "stage_head": 0.0,
                "pair_head": 0.0,
            },
            "head_stabilization": {
                "backbone_lora": 0.0,
                "frame_projector": float(get_by_path(cfg, "retention.schedule.stabilization_projector_lr", 1e-4)),
                "set_encoder": float(get_by_path(cfg, "retention.schedule.stabilization_head_lr", 5e-5)),
                "stage_head": float(get_by_path(cfg, "retention.schedule.stabilization_head_lr", 5e-5)),
                "pair_head": float(get_by_path(cfg, "retention.schedule.stabilization_head_lr", 5e-5)),
            },
            "joint_qlora": {
                "backbone_lora": float(get_by_path(cfg, "retention.schedule.joint_lora_lr", 2e-5)),
                "frame_projector": float(get_by_path(cfg, "retention.schedule.joint_head_lr", 3e-5)),
                "set_encoder": float(get_by_path(cfg, "retention.schedule.joint_head_lr", 3e-5)),
                "stage_head": float(get_by_path(cfg, "retention.schedule.joint_head_lr", 3e-5)),
                "pair_head": float(get_by_path(cfg, "retention.schedule.joint_head_lr", 3e-5)),
            },
        }
        self.step_index = 0
        self.apply(0)

    def phase_at(self, step: int) -> str:
        if step < self.projector_end:
            return "projector_only"
        if step < self.stabilization_end:
            return "head_stabilization"
        return "joint_qlora"

    def _joint_factor(self, step: int) -> float:
        joint_steps = max(1, self.total_steps - self.stabilization_end)
        joint_step = max(0, step - self.stabilization_end)
        warmup = max(1, int(math.ceil(joint_steps * self.joint_warmup_ratio)))
        if joint_step < warmup:
            return float(joint_step + 1) / float(warmup)
        progress = min(1.0, (joint_step - warmup) / max(1, joint_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    def apply(self, step: int) -> str:
        self.step_index = int(step)
        phase = self.phase_at(self.step_index)
        factor = self._joint_factor(self.step_index) if phase == "joint_qlora" else 1.0
        for group in self.optimizer.param_groups:
            name = str(group.get("group_name", ""))
            if name not in self.learning_rates[phase]:
                raise RuntimeError(f"Retention optimizer group is unbound: {name!r}")
            group["lr"] = self.learning_rates[phase][name] * factor
        return phase

    def step(self) -> None:
        self.apply(self.step_index + 1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "projector_end": self.projector_end,
            "stabilization_end": self.stabilization_end,
            "learning_rates": self.learning_rates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("total_steps", -1)) != self.total_steps:
            raise RuntimeError("Retention scheduler total-step mismatch")
        self.apply(int(state["step_index"]))


def retention_contract(cfg: dict[str, Any], scheduler: ChampionRetentionLRScheduler) -> dict[str, Any]:
    return {
        "enabled": bool(get_by_path(cfg, "retention.enabled", False)),
        "teacher_cache": str(get_by_path(cfg, "retention.teacher_cache")),
        "teacher_mask_policy": str(get_by_path(cfg, "retention.teacher_mask_policy", "teacher_correct_only")),
        "temperature": float(get_by_path(cfg, "retention.temperature", 2.0)),
        "weights": {
            "stage": float(get_by_path(cfg, "retention.stage_kd_weight", 0.10)),
            "pair": float(get_by_path(cfg, "retention.pair_kd_weight", 0.05)),
            "permutation": float(get_by_path(cfg, "retention.permutation_kd_weight", 0.10)),
        },
        "schedule": scheduler.state_dict(),
    }


def permutation_semantic_sha256() -> str:
    payload = json.dumps({"pairs": PAIRS, "perms": PERMS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
