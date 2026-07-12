import torch

from snu_order.qwen3vl.modeling_stage_pair import AntiSymmetricPairwiseHead, Qwen3VLStagePairModel
from snu_order.qwen3vl.permutations import answer_to_perm_index, perm_index_to_answer
from snu_order.qwen3vl.stage_pair_scorer import (
    StagePairStructuredLoss,
    class_position_table,
    pair_sign_table,
    prediction_answers_from_logits,
    remap_logits_to_canonical,
    stage_scores_from_logits,
    structured_permutation_logits,
)


def test_stage_pair_output_shapes():
    model = Qwen3VLStagePairModel(None, hidden_size=8, model_dim=16, set_layers=1, set_heads=4, set_ffn_dim=32)
    out = model(frame_hidden=torch.randn(2, 4, 8))
    assert out["stage_logits"].shape == (2, 4, 4)
    assert out["pair_logits"].shape == (2, 6)
    assert out["final_logits"].shape == (2, 24)


def test_high_correct_stage_logits_selects_answer_permutation():
    answer = [2, 4, 1, 3]
    target_idx = answer_to_perm_index(answer)
    stage_logits = torch.full((1, 4, 4), -10.0)
    for frame_idx, stage in enumerate([v - 1 for v in answer]):
        stage_logits[0, frame_idx, stage] = 10.0
    logits = structured_permutation_logits(stage_logits, pair_logits=None, pair_weight=0.0)
    assert int(logits.argmax(dim=1).item()) == target_idx
    assert prediction_answers_from_logits(logits) == [answer]


def test_stage_score_uses_global_permutation_not_independent_argmax():
    stage_logits = torch.zeros((1, 4, 4))
    stage_logits[:, :, 0] = 10.0
    logits = structured_permutation_logits(stage_logits, pair_logits=None, pair_weight=0.0)
    pred_answer = prediction_answers_from_logits(logits)[0]
    assert sorted(pred_answer) == [1, 2, 3, 4]


def test_antisymmetric_pairwise_head():
    head = AntiSymmetricPairwiseHead(model_dim=8, hidden_dim=8, dropout=0.0)
    directional = head.directional_logits(torch.randn(3, 4, 8))
    assert torch.allclose(directional, -directional.transpose(1, 2), atol=1e-6)
    assert head(torch.randn(3, 4, 8)).shape == (3, 6)


def test_pair_sign_table_matches_answers():
    signs = pair_sign_table()
    positions = class_position_table()
    for class_idx in range(24):
        for pair_idx, (i, j) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
            expected = 1.0 if int(positions[class_idx, i]) < int(positions[class_idx, j]) else -1.0
            assert float(signs[class_idx, pair_idx]) == expected


def test_stage_score_manual_value():
    stage_logits = torch.zeros((1, 4, 4))
    scores = stage_scores_from_logits(stage_logits)
    assert scores.shape == (1, 24)
    assert torch.allclose(scores, torch.full((1, 24), -torch.log(torch.tensor(4.0))))


def test_structured_loss_finite_backward():
    model = Qwen3VLStagePairModel(None, hidden_size=8, model_dim=16, set_layers=1, set_heads=4, set_ffn_dim=32)
    out = model(frame_hidden=torch.randn(2, 4, 8))
    loss_fn = StagePairStructuredLoss()
    loss = loss_fn(out, torch.tensor([0, 1]), torch.tensor([[1, 2, 3, 4], [1, 2, 4, 3]])).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_canonical_remap_reverse():
    answer = [2, 4, 1, 3]
    idx = answer_to_perm_index(answer)
    logits = torch.full((1, 24), -10.0)
    logits[0, idx] = 10.0
    remapped = remap_logits_to_canonical(logits, [3, 2, 1, 0])
    pred = perm_index_to_answer(int(remapped.argmax(dim=1).item()))
    canonical = [0, 0, 0, 0]
    for new_slot, old_slot in enumerate([3, 2, 1, 0]):
        canonical[old_slot] = answer[new_slot]
    assert pred == canonical
