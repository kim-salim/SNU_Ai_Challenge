from __future__ import annotations

from pathlib import Path
from typing import Any

from snu_order.utils.config import get_by_path


def build_modules(
    embedding_dim: int,
    quality_dim: int,
    cfg: dict[str, Any],
    device: str,
) -> tuple[Any, Any, Any | None]:
    import torch

    from snu_order.models.frame_projector import FrameProjector
    from snu_order.models.pairwise_head import PairwiseHead
    from snu_order.models.permutation_ranker import PermutationRanker

    hidden_dim = int(get_by_path(cfg, "model.frame_hidden_dim", 256))
    dropout = float(get_by_path(cfg, "model.dropout", 0.1))
    projector = FrameProjector(
        embedding_dim=embedding_dim,
        quality_dim=quality_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    ranker = PermutationRanker(hidden_dim=hidden_dim).to(device)
    pairwise = None
    if bool(get_by_path(cfg, "model.pairwise_aux.enabled", True)):
        pair_hidden = int(get_by_path(cfg, "model.pairwise_aux.hidden_dim", 512))
        pairwise = PairwiseHead(hidden_dim=hidden_dim, pair_hidden_dim=pair_hidden).to(device)
    return projector, ranker, pairwise


def save_checkpoint(
    path: str | Path,
    projector: Any,
    ranker: Any,
    pairwise: Any | None,
    cfg: dict[str, Any],
    embedding_dim: int,
    quality_dim: int,
    metrics: dict[str, float],
) -> None:
    import torch

    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector": projector.state_dict(),
            "ranker": ranker.state_dict(),
            "pairwise": None if pairwise is None else pairwise.state_dict(),
            "config": cfg,
            "embedding_dim": embedding_dim,
            "quality_dim": quality_dim,
            "metrics": metrics,
        },
        ckpt_path,
    )


def load_modules_from_checkpoint(path: str | Path, cfg: dict[str, Any] | None, device: str) -> tuple[Any, Any, Any | None, dict[str, Any]]:
    import torch

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    merged_cfg = cfg or checkpoint.get("config", {})
    embedding_dim = int(checkpoint["embedding_dim"])
    quality_dim = int(checkpoint["quality_dim"])
    projector, ranker, pairwise = build_modules(embedding_dim, quality_dim, merged_cfg, device)
    projector.load_state_dict(checkpoint["projector"])
    ranker.load_state_dict(checkpoint["ranker"])
    pair_state = checkpoint.get("pairwise")
    if pair_state is not None:
        if pairwise is None:
            merged_cfg.setdefault("model", {}).setdefault("pairwise_aux", {})["enabled"] = True
            projector, ranker, pairwise = build_modules(embedding_dim, quality_dim, merged_cfg, device)
            projector.load_state_dict(checkpoint["projector"])
            ranker.load_state_dict(checkpoint["ranker"])
        pairwise.load_state_dict(pair_state)
    elif pairwise is not None:
        pairwise = None
    return projector, ranker, pairwise, checkpoint

