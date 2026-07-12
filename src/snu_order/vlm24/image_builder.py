from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_LABELS = ["F1", "F2", "F3", "F4"]


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def detect_frame_columns(columns: list[str]) -> list[str]:
    normalized = {_normalize_name(col): col for col in columns}
    patterns = [
        [f"frame{i}" for i in range(4)],
        [f"frame{i}" for i in range(1, 5)],
        [f"image{i}" for i in range(4)],
        [f"image{i}" for i in range(1, 5)],
        [f"path{i}" for i in range(4)],
        [f"path{i}" for i in range(1, 5)],
    ]
    for pattern in patterns:
        if all(key in normalized for key in pattern):
            return [normalized[key] for key in pattern]

    indexed: list[tuple[int, str]] = []
    for col in columns:
        key = _normalize_name(col)
        match = re.fullmatch(r"(?:frame|image|img|path)([0-4])", key)
        if match is None:
            continue
        raw_idx = int(match.group(1))
        idx = raw_idx if raw_idx in {0, 1, 2, 3} else raw_idx - 1
        if 0 <= idx <= 3:
            indexed.append((idx, col))
    if len(indexed) >= 4:
        by_idx = {idx: col for idx, col in indexed}
        if set(by_idx) == {0, 1, 2, 3}:
            return [by_idx[idx] for idx in range(4)]
    raise ValueError(f"Could not detect 4 frame path columns. Available columns: {columns}")


def load_sample_frames(row: dict[str, Any], image_root: str | Path) -> list[Image.Image]:
    frame_cols = detect_frame_columns(list(row.keys()))
    root = Path(image_root)
    frames: list[Image.Image] = []
    for col in frame_cols:
        raw_path = str(row[col]).strip()
        if not raw_path:
            raise ValueError(f"Empty frame path in column {col}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Frame image not found: {path}")
        with Image.open(path) as image:
            frames.append(ImageOps.exif_transpose(image).convert("RGB"))
    return frames


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_labeled_grid_2x2(
    frames: list[Image.Image],
    labels: list[str] = DEFAULT_LABELS,
    grid_size: int = 1024,
    label_height: int = 40,
) -> Image.Image:
    if len(frames) != 4:
        raise ValueError(f"make_labeled_grid_2x2 requires 4 frames, got {len(frames)}")
    if len(labels) != 4:
        raise ValueError(f"labels must have length 4, got {len(labels)}")
    if grid_size <= 0 or label_height < 0:
        raise ValueError("grid_size must be positive and label_height must be non-negative")

    grid = Image.new("RGB", (grid_size, grid_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = _load_font(max(14, int(label_height * 0.6)))
    cell_w = grid_size // 2
    cell_h = grid_size // 2
    image_h = max(1, cell_h - label_height)

    for idx, frame in enumerate(frames):
        row = idx // 2
        col = idx % 2
        x0 = col * cell_w
        y0 = row * cell_h
        x1 = grid_size if col == 1 else x0 + cell_w
        y1 = grid_size if row == 1 else y0 + cell_h
        actual_cell_w = x1 - x0
        actual_cell_h = y1 - y0
        actual_image_h = max(1, actual_cell_h - label_height)

        draw.rectangle([x0, y0, x1 - 1, y0 + label_height - 1], fill=(20, 20, 20))
        label_text = labels[idx]
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (x0 + (actual_cell_w - text_w) / 2, y0 + (label_height - text_h) / 2 - 1),
            label_text,
            fill=(255, 255, 255),
            font=font,
        )

        image = frame.convert("RGB")
        scale = min(actual_cell_w / image.width, actual_image_h / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        paste_x = x0 + (actual_cell_w - resized.width) // 2
        paste_y = y0 + label_height + (image_h - resized.height) // 2
        grid.paste(resized, (paste_x, paste_y))
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 0, 0), width=2)
    return grid
