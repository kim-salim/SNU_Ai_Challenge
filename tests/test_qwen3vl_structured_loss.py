from __future__ import annotations

import torch

from snu_order.qwen3vl.structured_loss import StructuredPermutationLoss, build_pairwise_mask, build_position_mask


def test_position_and_pairwise_masks_shape_and_correctness():
    position = build_position_mask()
    pairwise = build_pairwise_mask()
    assert tuple(position.shape) == (24, 4, 4)
    assert tuple(pairwise.shape) == (24, 6)
    assert torch.all(position.sum(dim=2) == 1)
    assert pairwise.dtype == torch.bool


def test_structured_loss_finite_backward_and_high_correct_small():
    loss_fn = StructuredPermutationLoss(label_smoothing=0.0)
    logits = torch.zeros((2, 24), requires_grad=True)
    target = torch.tensor([0, 23])
    answers = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    out = loss_fn(logits, target, answers)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    confident = torch.full((1, 24), -20.0)
    confident[0, 0] = 20.0
    out2 = loss_fn(confident, torch.tensor([0]), torch.tensor([[1, 2, 3, 4]]))
    assert float(out2.loss) < 0.1


def test_uniform_logits_loss_is_finite():
    loss_fn = StructuredPermutationLoss()
    logits = torch.zeros((1, 24), requires_grad=True)
    out = loss_fn(logits, torch.tensor([5]), torch.tensor([[1, 4, 3, 2]]))
    assert torch.isfinite(out.loss)
