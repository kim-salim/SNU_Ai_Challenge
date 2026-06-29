from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

QUALITY_DIM = 9


def _image_to_gray_array(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.max(initial=0.0) > 2.0:
            arr = arr / 255.0
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            gray = arr[..., 0]
        else:
            gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        raise ValueError(f"Expected image rank 2 or 3, got shape {arr.shape}")
    return np.clip(gray.astype(np.float32), 0.0, 1.0)


def _as_nested_images(images: Any) -> list[list[Any]]:
    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    if isinstance(images, np.ndarray):
        if images.ndim == 5:
            return [[images[b, i] for i in range(images.shape[1])] for b in range(images.shape[0])]
        if images.ndim == 4 and images.shape[0] == 4:
            return [[images[i] for i in range(4)]]
        raise ValueError(f"Expected image array [B,4,...] or [4,...], got {images.shape}")
    if not isinstance(images, Sequence):
        raise ValueError("images must be a nested sequence, PIL list, numpy array, or torch tensor")
    if len(images) == 0:
        return []
    first = images[0]
    if isinstance(first, Sequence) and not isinstance(first, (Image.Image, np.ndarray, str, bytes)):
        nested = [list(item) for item in images]
    else:
        if len(images) != 4:
            raise ValueError("Flat image list must contain exactly 4 frames")
        nested = [list(images)]
    for sample in nested:
        if len(sample) != 4:
            raise ValueError(f"Each sample must contain 4 frames, got {len(sample)}")
    return nested


def _blur_and_edge(gray: np.ndarray) -> tuple[float, float]:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0, 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    gx = gray[1:-1, 2:] - gray[1:-1, :-2]
    gy = gray[2:, 1:-1] - gray[:-2, 1:-1]
    edge_density = float((np.abs(gx) + np.abs(gy) > 0.08).mean())
    return float(lap.var()), edge_density


def compute_basic_image_quality(images: Any) -> np.ndarray:
    """Return [B, 4, 6] image-only quality features."""
    nested = _as_nested_images(images)
    quality = np.zeros((len(nested), 4, 6), dtype=np.float32)
    for b, sample in enumerate(nested):
        for i, image in enumerate(sample):
            gray = _image_to_gray_array(image)
            blur_score, edge_density = _blur_and_edge(gray)
            quality[b, i] = np.asarray(
                [
                    float((gray < 0.05).mean()),
                    float((gray > 0.95).mean()),
                    float(gray.mean()),
                    float(gray.std()),
                    blur_score,
                    edge_density,
                ],
                dtype=np.float32,
            )
    return quality


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch")


def compute_embedding_quality(text_emb: Any, frame_emb: Any) -> Any:
    """Return [B, 4, 3] embedding quality features."""
    if _is_torch_tensor(text_emb) or _is_torch_tensor(frame_emb):
        import torch
        import torch.nn.functional as F

        text = F.normalize(text_emb.float(), dim=-1)
        frames = F.normalize(frame_emb.float(), dim=-1)
        sim = torch.matmul(frames, frames.transpose(1, 2))
        eye = torch.eye(4, device=sim.device, dtype=torch.bool).unsqueeze(0)
        max_frame_similarity = sim.masked_fill(eye, -float("inf")).max(dim=-1).values
        mean_frame_similarity = sim.masked_fill(eye, 0.0).sum(dim=-1) / 3.0
        text_image_cosine = torch.sum(frames * text.unsqueeze(1), dim=-1)
        return torch.stack(
            [max_frame_similarity, mean_frame_similarity, text_image_cosine],
            dim=-1,
        )

    text_np = np.asarray(text_emb, dtype=np.float32)
    frame_np = np.asarray(frame_emb, dtype=np.float32)
    text_norm = text_np / np.maximum(np.linalg.norm(text_np, axis=-1, keepdims=True), 1e-12)
    frame_norm = frame_np / np.maximum(np.linalg.norm(frame_np, axis=-1, keepdims=True), 1e-12)
    sim = np.matmul(frame_norm, np.swapaxes(frame_norm, 1, 2))
    mask = np.eye(4, dtype=bool)[None, :, :]
    off_diag = np.where(mask, np.nan, sim)
    max_frame_similarity = np.nanmax(off_diag, axis=-1)
    mean_frame_similarity = np.nanmean(off_diag, axis=-1)
    text_image_cosine = np.sum(frame_norm * text_norm[:, None, :], axis=-1)
    return np.stack([max_frame_similarity, mean_frame_similarity, text_image_cosine], axis=-1).astype(np.float32)


def compute_quality_features(images: Any, text_emb: Any, frame_emb: Any) -> Any:
    basic = compute_basic_image_quality(images)
    embedding_quality = compute_embedding_quality(text_emb, frame_emb)
    if _is_torch_tensor(embedding_quality):
        import torch

        basic_tensor = torch.as_tensor(basic, dtype=embedding_quality.dtype, device=embedding_quality.device)
        return torch.cat([basic_tensor, embedding_quality], dim=-1)
    return np.concatenate([basic, np.asarray(embedding_quality, dtype=np.float32)], axis=-1)
