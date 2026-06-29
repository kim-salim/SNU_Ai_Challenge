import pytest


def test_model_shapes():
    torch = pytest.importorskip("torch")
    from snu_order.models.frame_projector import FrameProjector
    from snu_order.models.losses import PermPairLoss
    from snu_order.models.pairwise_head import PairwiseHead
    from snu_order.models.permutation_ranker import PermutationRanker

    batch = 3
    dim = 16
    hidden = 32
    text_emb = torch.randn(batch, dim)
    frame_emb = torch.randn(batch, 4, dim)
    quality = torch.randn(batch, 4, 9)
    projector = FrameProjector(embedding_dim=dim, quality_dim=9, hidden_dim=hidden)
    ranker = PermutationRanker(hidden_dim=hidden, scorer_hidden_dims=(64, 32))
    pairwise = PairwiseHead(hidden_dim=hidden, pair_hidden_dim=32)

    frame_tokens = projector(text_emb, frame_emb, quality)
    assert frame_tokens.shape == (batch, 4, hidden)
    perm_scores = ranker(frame_tokens)
    assert perm_scores.shape == (batch, 24)
    pair_logits = pairwise(frame_tokens)
    assert pair_logits.shape == (batch, 6)

    loss_fn = PermPairLoss(pair_aux_weight=0.3, label_smoothing=0.05)
    loss, metrics = loss_fn(
        perm_scores,
        torch.zeros(batch, dtype=torch.long),
        pair_logits,
        torch.ones(batch, 6),
    )
    assert loss.ndim == 0
    assert metrics["loss"] >= 0
