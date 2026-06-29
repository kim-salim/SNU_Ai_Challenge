from pathlib import Path

import numpy as np
import pytest

from snu_order.features.cache import load_feature_cache, save_feature_cache, validate_feature_cache


def make_cache(with_labels: bool = True):
    data = {
        "ids": np.asarray(["a", "b"]),
        "text_emb": np.ones((2, 8), dtype=np.float32),
        "frame_emb": np.ones((2, 4, 8), dtype=np.float32),
        "quality": np.zeros((2, 4, 9), dtype=np.float32),
    }
    if with_labels:
        data.update(
            {
                "answer": np.asarray([[1, 4, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
                "target_perm_idx": np.asarray([0, 1], dtype=np.int64),
                "pairwise_labels": np.ones((2, 6), dtype=np.float32),
            }
        )
    return data


def test_validate_feature_cache_with_labels():
    validate_feature_cache(make_cache(True), require_labels=True)


def test_validate_feature_cache_without_labels():
    validate_feature_cache(make_cache(False), require_labels=False)


def test_save_and_load_feature_cache(tmp_path: Path):
    path = tmp_path / "cache.npz"
    save_feature_cache(path, make_cache(True))
    loaded = load_feature_cache(path)
    validate_feature_cache(loaded, require_labels=True)
    assert loaded["frame_emb"].shape == (2, 4, 8)


def test_reject_nan():
    data = make_cache(False)
    data["quality"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_feature_cache(data, require_labels=False)

