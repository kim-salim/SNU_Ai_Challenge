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
    include_quality = bool(get_by_path(cfg, "features.include_quality", True))
    effective_quality_dim = int(quality_dim) if include_quality else 0
    projector = FrameProjector(
        embedding_dim=embedding_dim,
        quality_dim=effective_quality_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    scorer_hidden = get_by_path(cfg, "model.perm_scorer.hidden_dims", (1024, 256))
    scorer_hidden_dims = tuple(int(v) for v in scorer_hidden)
    if len(scorer_hidden_dims) != 2:
        raise ValueError(f"model.perm_scorer.hidden_dims must have length 2, got {scorer_hidden}")
    scorer_dropout = float(get_by_path(cfg, "model.perm_scorer.dropout", dropout))
    ranker = PermutationRanker(
        hidden_dim=hidden_dim,
        scorer_hidden_dims=scorer_hidden_dims,  # type: ignore[arg-type]
        dropout=scorer_dropout,
    ).to(device)
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
    try:
        projector.load_state_dict(checkpoint["projector"])
        ranker.load_state_dict(checkpoint["ranker"])
    except RuntimeError:
        if not bool(get_by_path(merged_cfg, "features.include_quality", True)) and quality_dim > 0:
            compat_cfg = dict(merged_cfg)
            compat_cfg.setdefault("features", {})["include_quality"] = True
            projector, ranker, pairwise = build_modules(embedding_dim, quality_dim, compat_cfg, device)
            projector.load_state_dict(checkpoint["projector"])
            ranker.load_state_dict(checkpoint["ranker"])
            merged_cfg = compat_cfg
        else:
            raise
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
