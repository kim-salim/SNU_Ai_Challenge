from __future__ import annotations

from typing import Any

import torch


VALID_FRAME_CHUNK_SIZES = (1, 2, 4)


def normalize_frame_chunk_size(value: int | None) -> int | None:
    if value is None:
        return None
    size = int(value)
    if size not in VALID_FRAME_CHUNK_SIZES:
        raise RuntimeError(
            f"inference.frame_chunk_size must be null or one of {VALID_FRAME_CHUNK_SIZES}, got {value}"
        )
    return size


def _image_patch_offsets(image_grid_thw: torch.Tensor, total_frames: int) -> list[int]:
    if image_grid_thw.ndim != 2 or tuple(image_grid_thw.shape) != (total_frames, 3):
        raise RuntimeError(
            f"image_grid_thw must have shape [{total_frames},3], got {tuple(image_grid_thw.shape)}"
        )
    counts = image_grid_thw.detach().to(device="cpu", dtype=torch.long).prod(dim=1)
    if bool((counts <= 0).any()):
        raise RuntimeError(f"image_grid_thw contains a non-positive patch count: {counts.tolist()}")
    offsets = [0]
    for count in counts.tolist():
        offsets.append(offsets[-1] + int(count))
    return offsets


def slice_frame_multimodal_inputs(
    inputs: dict[str, Any],
    *,
    start: int,
    end: int,
    total_frames: int,
) -> dict[str, Any]:
    if not (0 <= int(start) < int(end) <= int(total_frames)):
        raise RuntimeError(f"Invalid frame slice [{start},{end}) for total_frames={total_frames}")
    grid = inputs.get("image_grid_thw")
    pixels = inputs.get("pixel_values")
    if not torch.is_tensor(grid) or not torch.is_tensor(pixels):
        raise RuntimeError("Frame chunking requires tensor image_grid_thw and pixel_values")
    offsets = _image_patch_offsets(grid, total_frames)
    if pixels.ndim < 1 or int(pixels.shape[0]) != offsets[-1]:
        raise RuntimeError(
            "pixel_values row count does not match image_grid_thw patch counts: "
            f"pixels={tuple(pixels.shape)}, expected_rows={offsets[-1]}"
        )

    sliced: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            sliced[key] = value
            continue
        if key == "pixel_values":
            sliced[key] = value[offsets[start] : offsets[end]]
        elif key == "image_grid_thw":
            sliced[key] = value[start:end]
        elif value.ndim >= 1 and int(value.shape[0]) == total_frames:
            sliced[key] = value[start:end]
        elif key == "position_ids" and value.ndim >= 2 and int(value.shape[1]) == total_frames:
            sliced[key] = value[:, start:end]
        else:
            sliced[key] = value

    expected_rows = end - start
    for key in ("input_ids", "attention_mask", "image_grid_thw"):
        value = sliced.get(key)
        if not torch.is_tensor(value) or value.ndim < 1 or int(value.shape[0]) != expected_rows:
            raise RuntimeError(
                f"Chunked {key} must have leading dimension {expected_rows}, got "
                f"{None if not torch.is_tensor(value) else tuple(value.shape)}"
            )
    expected_patches = offsets[end] - offsets[start]
    if int(sliced["pixel_values"].shape[0]) != expected_patches:
        raise RuntimeError("Chunked pixel_values lost or reordered image patches")
    return sliced
