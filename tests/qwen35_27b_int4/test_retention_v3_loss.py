from __future__ import annotations

import torch

from snu_order.qwen3vl.permutations import answer_to_perm_index
from snu_order.qwen3vl.retention_v3 import (
    apply_preregistered_component_coverage,
    component_safe_aux_scale,
    component_safe_retention_loss,
)


def _cfg() -> dict:
    return {
        "retention": {
            "enabled": True,
            "mode": "component_safe_v3",
            "soft_kl": False,
            "thresholds": {
                "teacher_permutation_margin_min": 0.1,
                "reference_stage_q30": 0.2,
                "reference_stage_q50": 0.5,
                "reference_pair_q30": 0.2,
                "reference_pair_q50": 0.5,
                "reference_permutation_q30": 0.2,
                "reference_permutation_q50": 0.5,
            },
            "weights": {
                "rescue_stage": 0.10,
                "rescue_pair": 0.05,
                "rescue_permutation": 0.10,
                "protect_stage": 0.05,
                "protect_pair": 0.025,
                "protect_permutation": 0.05,
            },
            "max_aux_to_base_loss_ratio": 0.20,
            "focal_gamma": 1.0,
            "per_sample_aux_weight_cap": 1.0,
        }
    }


def _batch() -> dict[str, torch.Tensor]:
    answer = torch.tensor([[1, 2, 3, 4]])
    target = torch.tensor([answer_to_perm_index(answer[0].tolist())])
    teacher_stage = torch.full((1, 4, 4), -2.0)
    teacher_stage[0, 0, 1] = 3.0  # Teacher Stage is wrong for frame 0.
    for frame in range(1, 4):
        teacher_stage[0, frame, frame] = 3.0
    teacher_pair = torch.full((1, 6), -2.0)
    teacher_pair[0, 0] = 2.0  # One teacher Pair component is correct.
    teacher_final = torch.zeros((1, 24))
    teacher_final[0, (int(target.item()) + 1) % 24] = 3.0  # Final teacher prediction is wrong.
    reference_stage = torch.zeros((1, 4, 4))
    reference_pair = torch.full((1, 6), -1.0)
    reference_final = torch.zeros((1, 24))
    return {
        "answer": answer,
        "target_perm_idx": target,
        "teacher_stage_logits": teacher_stage,
        "teacher_pair_logits": teacher_pair,
        "teacher_final_logits": teacher_final,
        "reference_stage_logits": reference_stage,
        "reference_pair_logits": reference_pair,
        "reference_final_logits": reference_final,
    }


def test_auxiliary_schedule_finishes_with_gt_only() -> None:
    assert component_safe_aux_scale(0.05) == 0.0
    assert component_safe_aux_scale(0.10) == 0.0
    assert component_safe_aux_scale(0.15) == 1.0
    assert component_safe_aux_scale(0.40) == 1.0
    assert 0.0 < component_safe_aux_scale(0.55) < 1.0
    assert component_safe_aux_scale(0.70) == 0.0
    assert component_safe_aux_scale(1.00) == 0.0


def test_preregistered_coverage_disables_only_low_coverage_rescue() -> None:
    weights = _cfg()["retention"]["weights"]
    policy = {
        "minimum_component_rescue_fraction": 0.03,
        "rescue_coverage": {"stage": 0.029, "pair": 0.03, "permutation": 0.08},
        "eligible_rescue_components": {"stage": False, "pair": True, "permutation": True},
    }
    effective = apply_preregistered_component_coverage(weights, policy)
    assert effective["rescue_stage"] == 0.0
    assert effective["rescue_pair"] == weights["rescue_pair"]
    assert effective["protect_stage"] == weights["protect_stage"]


def test_coverage_floor_change_is_rejected() -> None:
    policy = {
        "minimum_component_rescue_fraction": 0.02,
        "rescue_coverage": {"stage": 0.03, "pair": 0.03, "permutation": 0.03},
        "eligible_rescue_components": {"stage": True, "pair": True, "permutation": True},
    }
    try:
        apply_preregistered_component_coverage(_cfg()["retention"]["weights"], policy)
    except RuntimeError as exc:
        assert "differs from preregistration" in str(exc)
    else:
        raise AssertionError("Changing the preregistered rescue coverage floor must fail")


def test_masks_are_component_safe_not_final_teacher_safe() -> None:
    outputs = {
        "stage_logits": torch.zeros((1, 4, 4), requires_grad=True),
        "pair_logits": torch.zeros((1, 6), requires_grad=True),
        "final_logits": torch.zeros((1, 24), requires_grad=True),
    }
    result = component_safe_retention_loss(
        outputs, _batch(), _cfg(), base_loss=torch.tensor(2.0), progress=0.20
    )
    # Final teacher correctness must not enable an incorrect Stage component.
    assert result.active_counts["rescue_stage"] == 3
    # A correct Pair component remains usable even though final teacher top-1 is wrong.
    assert result.active_counts["rescue_pair"] == 1
    assert result.active_counts["rescue_permutation"] == 0


def test_inactive_components_receive_zero_auxiliary_gradient() -> None:
    outputs = {
        "stage_logits": torch.zeros((1, 4, 4), requires_grad=True),
        "pair_logits": torch.zeros((1, 6), requires_grad=True),
        "final_logits": torch.zeros((1, 24), requires_grad=True),
    }
    result = component_safe_retention_loss(
        outputs, _batch(), _cfg(), base_loss=torch.tensor(2.0), progress=0.20
    )
    result.loss.backward()
    assert torch.count_nonzero(outputs["stage_logits"].grad[0, 0]).item() == 0
    assert torch.count_nonzero(outputs["stage_logits"].grad[0, 1:]).item() > 0
    assert torch.count_nonzero(outputs["pair_logits"].grad[0, 1:]).item() == 0
    assert torch.count_nonzero(outputs["pair_logits"].grad[0, 0]).item() > 0


def test_teacher_confidence_weights_are_detached() -> None:
    batch = _batch()
    for key in ("teacher_stage_logits", "teacher_pair_logits", "teacher_final_logits"):
        batch[key] = batch[key].clone().requires_grad_(True)
    outputs = {
        "stage_logits": torch.zeros((1, 4, 4), requires_grad=True),
        "pair_logits": torch.zeros((1, 6), requires_grad=True),
        "final_logits": torch.zeros((1, 24), requires_grad=True),
    }
    result = component_safe_retention_loss(
        outputs, batch, _cfg(), base_loss=torch.tensor(2.0), progress=0.20
    )
    result.loss.backward()
    assert all(
        batch[key].grad is None
        for key in ("teacher_stage_logits", "teacher_pair_logits", "teacher_final_logits")
    )


def test_no_active_component_returns_exact_zero_without_nan() -> None:
    batch = _batch()
    batch["teacher_stage_logits"] = torch.full((1, 4, 4), -1.0)
    for frame in range(4):
        batch["teacher_stage_logits"][0, frame, (frame + 1) % 4] = 1.0
    batch["teacher_pair_logits"] = torch.full((1, 6), -1.0)
    batch["reference_stage_logits"] = torch.full((1, 4, 4), -1.0)
    for frame in range(4):
        batch["reference_stage_logits"][0, frame, (frame + 1) % 4] = 1.0
    outputs = {
        "stage_logits": torch.zeros((1, 4, 4), requires_grad=True),
        "pair_logits": torch.zeros((1, 6), requires_grad=True),
        "final_logits": torch.zeros((1, 24), requires_grad=True),
    }
    result = component_safe_retention_loss(
        outputs, batch, _cfg(), base_loss=torch.tensor(2.0), progress=0.20
    )
    assert torch.isfinite(result.loss)
    assert sum(result.active_counts.values()) == 0
    assert result.loss.item() == 0.0


def test_soft_kl_is_fail_closed() -> None:
    cfg = _cfg()
    cfg["retention"]["soft_kl"] = True
    outputs = {
        "stage_logits": torch.zeros((1, 4, 4)),
        "pair_logits": torch.zeros((1, 6)),
        "final_logits": torch.zeros((1, 24)),
    }
    try:
        component_safe_retention_loss(outputs, _batch(), cfg, base_loss=torch.tensor(1.0), progress=0.2)
    except RuntimeError as exc:
        assert "forbids soft KL" in str(exc)
    else:
        raise AssertionError("soft KL must be rejected")
