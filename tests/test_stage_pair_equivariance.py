import torch

from snu_order.qwen3vl.modeling_stage_pair import PositionFreeSetEncoder, Qwen3VLStagePairModel
from snu_order.qwen3vl.stage_pair_scorer import StagePairStructuredLoss, remap_logits_to_canonical


def test_position_free_set_encoder_equivariance():
    encoder = PositionFreeSetEncoder(8, model_dim=16, num_layers=1, nhead=4, dim_feedforward=32, dropout=0.0, use_set_encoder=True)
    encoder.eval()
    x = torch.randn(2, 4, 8)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        y = encoder(x)
        y_perm = encoder(x[:, perm])
    assert torch.allclose(y[:, perm], y_perm, atol=1e-5)


def test_stage_matrix_rows_move_with_input_shuffle():
    model = Qwen3VLStagePairModel(None, hidden_size=8, model_dim=16, set_layers=1, set_heads=4, set_ffn_dim=32, dropout=0.0)
    model.eval()
    x = torch.randn(1, 4, 8)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        a = model(frame_hidden=x)["stage_logits"]
        b = model(frame_hidden=x[:, perm])["stage_logits"]
    assert torch.allclose(a[:, perm], b, atol=1e-5)


def test_consistency_loss_finite_backward():
    model = Qwen3VLStagePairModel(None, hidden_size=8, model_dim=16, set_layers=1, set_heads=4, set_ffn_dim=32)
    x = torch.randn(2, 4, 8)
    out = model(frame_hidden=x)
    perm = torch.tensor([3, 2, 1, 0])
    shuf = model(frame_hidden=x[:, perm])
    consistency = remap_logits_to_canonical(shuf["final_logits"], perm)
    loss = StagePairStructuredLoss(consistency_weight=0.1)(
        out,
        torch.tensor([0, 1]),
        torch.tensor([[1, 2, 3, 4], [1, 2, 4, 3]]),
        consistency_logits=consistency,
    ).loss
    assert torch.isfinite(loss)
    loss.backward()
