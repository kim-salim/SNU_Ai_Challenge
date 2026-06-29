from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_KEYS = ("ids", "text_emb", "frame_emb", "quality")
LABEL_KEYS = ("answer", "target_perm_idx", "pairwise_labels")


def save_feature_cache(path: str | Path, data: dict[str, Any]) -> None:
    validate_feature_cache(data, require_labels=all(key in data for key in LABEL_KEYS))
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def _require(data: dict[str, Any], key: str) -> np.ndarray:
    if key not in data:
        raise ValueError(f"Feature cache missing key: {key}")
    return np.asarray(data[key])


def _check_finite(name: str, arr: np.ndarray) -> None:
    if np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
        raise ValueError(f"Feature cache key {name} contains NaN or Inf")


def validate_feature_cache(data: dict[str, Any], require_labels: bool) -> None:
    ids = _require(data, "ids")
    text_emb = _require(data, "text_emb")
    frame_emb = _require(data, "frame_emb")
    quality = _require(data, "quality")

    if ids.ndim != 1:
        raise ValueError(f"ids must have rank 1, got {ids.shape}")
    n = ids.shape[0]
    if text_emb.ndim != 2 or text_emb.shape[0] != n:
        raise ValueError(f"text_emb must have shape [N,D], got {text_emb.shape}")
    if frame_emb.ndim != 3 or frame_emb.shape[0] != n or frame_emb.shape[1] != 4:
        raise ValueError(f"frame_emb must have shape [N,4,D], got {frame_emb.shape}")
    if quality.ndim != 3 or quality.shape[0] != n or quality.shape[1] != 4:
        raise ValueError(f"quality must have shape [N,4,Q], got {quality.shape}")
    if frame_emb.shape[2] != text_emb.shape[1]:
        raise ValueError("frame_emb last dimension must match text_emb dimension")

    for key in REQUIRED_KEYS:
        _check_finite(key, np.asarray(data[key]))

    if require_labels:
        answer = _require(data, "answer")
        target_perm_idx = _require(data, "target_perm_idx")
        pairwise_labels = _require(data, "pairwise_labels")
        if answer.shape != (n, 4):
            raise ValueError(f"answer must have shape [N,4], got {answer.shape}")
        if target_perm_idx.shape != (n,):
            raise ValueError(f"target_perm_idx must have shape [N], got {target_perm_idx.shape}")
        if pairwise_labels.shape != (n, 6):
            raise ValueError(f"pairwise_labels must have shape [N,6], got {pairwise_labels.shape}")
        for key in LABEL_KEYS:
            _check_finite(key, np.asarray(data[key]))

