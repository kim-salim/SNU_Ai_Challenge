import numpy as np
from PIL import Image

from snu_order.features.quality import (
    compute_basic_image_quality,
    compute_embedding_quality,
    compute_quality_features,
)


def test_basic_image_quality_shape():
    frames = [[Image.fromarray(np.full((8, 8, 3), fill_value=i * 40, dtype=np.uint8)) for i in range(4)]]
    quality = compute_basic_image_quality(frames)
    assert quality.shape == (1, 4, 6)
    assert np.isfinite(quality).all()


def test_embedding_quality_shape():
    text = np.ones((2, 4), dtype=np.float32)
    frames = np.ones((2, 4, 4), dtype=np.float32)
    quality = compute_embedding_quality(text, frames)
    assert quality.shape == (2, 4, 3)
    assert np.isfinite(quality).all()


def test_full_quality_shape():
    images = [[Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(4)]]
    text = np.ones((1, 4), dtype=np.float32)
    frames = np.ones((1, 4, 4), dtype=np.float32)
    quality = compute_quality_features(images, text, frames)
    assert quality.shape == (1, 4, 9)

