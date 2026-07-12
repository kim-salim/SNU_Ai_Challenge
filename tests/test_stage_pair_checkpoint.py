import torch

from snu_order.qwen3vl.modeling_stage_pair import Qwen3VLStagePairModel, load_stage_pair_checkpoint, save_stage_pair_checkpoint


def test_stage_pair_checkpoint_round_trip(tmp_path):
    model = Qwen3VLStagePairModel(None, hidden_size=8, model_dim=16, set_layers=1, set_heads=4, set_ffn_dim=32)
    with torch.no_grad():
        for param in model.stage_head.parameters():
            param.add_(1.0)
    save_stage_pair_checkpoint(tmp_path / "ckpt", model, {"x": 1}, {"exact_match": 0.5}, minimal=True)
    fresh = Qwen3VLStagePairModel(None, hidden_size=8, model_dim=16, set_layers=1, set_heads=4, set_ffn_dim=32)
    load_stage_pair_checkpoint(tmp_path / "ckpt", fresh)
    for left, right in zip(model.stage_head.parameters(), fresh.stage_head.parameters(), strict=True):
        assert torch.allclose(left, right)


def test_all_predictions_convert_to_valid_answers():
    model = Qwen3VLStagePairModel(None, hidden_size=8, model_dim=16, set_layers=1, set_heads=4, set_ffn_dim=32)
    logits = model(frame_hidden=torch.randn(5, 4, 8))["final_logits"]
    for idx in logits.argmax(dim=1).tolist():
        answer = __import__("snu_order.qwen3vl.permutations", fromlist=["perm_index_to_answer"]).perm_index_to_answer(int(idx))
        assert sorted(answer) == [1, 2, 3, 4]
