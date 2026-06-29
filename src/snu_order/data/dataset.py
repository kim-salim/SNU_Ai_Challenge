from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from snu_order.data.validate_submission import find_column, parse_answer
from snu_order.utils.io import read_csv_rows

TEXT_COLUMNS = ("Sentence", "sentence", "text", "Text", "caption", "Caption", "description", "Description")


def find_id_column(row: dict[str, Any]) -> str:
    col = find_column(row.keys(), ["Id", "ID", "id"])
    if col is None:
        raise ValueError("CSV row does not contain an Id column")
    return col


def find_text_column(row: dict[str, Any]) -> str:
    col = find_column(row.keys(), TEXT_COLUMNS)
    if col is None:
        raise ValueError(f"CSV row does not contain a sentence/text column. Available: {list(row.keys())}")
    return col


def _frame_candidates(index: int) -> tuple[str, ...]:
    return (
        f"frame{index}",
        f"frame_{index}",
        f"Frame{index}",
        f"Frame_{index}",
        f"image{index}",
        f"image_{index}",
        f"Image{index}",
        f"Image_{index}",
        f"path{index}",
        f"path_{index}",
        f"frame{index}_path",
        f"frame_{index}_path",
        f"image{index}_path",
        f"image_{index}_path",
    )


def find_frame_columns(row: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for idx in range(4):
        col = find_column(row.keys(), _frame_candidates(idx))
        if col is None:
            raise ValueError(
                f"CSV row does not contain a frame path column for frame {idx}. "
                f"Available: {list(row.keys())}"
            )
        columns.append(col)
    return columns


def read_competition_csv(path: str | Path) -> list[dict[str, str]]:
    return read_csv_rows(path)


def row_id(row: dict[str, Any]) -> str:
    return str(row[find_id_column(row)])


def row_sentence(row: dict[str, Any]) -> str:
    return str(row[find_text_column(row)])


def row_answer(row: dict[str, Any]) -> list[int]:
    answer_col = find_column(row.keys(), ["answer", "Answer"])
    if answer_col is not None and str(row[answer_col]).strip():
        return parse_answer(row[answer_col])
    cols = [find_column(row.keys(), [f"answer{i}", f"answer_{i}", f"Answer{i}", f"Answer_{i}"]) for i in range(4)]
    if all(col is not None for col in cols):
        return parse_answer([row[col] for col in cols if col is not None])
    raise ValueError("Row does not contain answer labels")


def resolve_path(base_csv: str | Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    csv_dir = Path(base_csv).parent
    repo_relative = Path(value)
    if repo_relative.exists():
        return repo_relative
    return csv_dir / path


def load_frames_for_row(row: dict[str, Any], csv_path: str | Path) -> list[Image.Image]:
    frame_cols = find_frame_columns(row)
    frames: list[Image.Image] = []
    for col in frame_cols:
        path = resolve_path(csv_path, str(row[col]))
        if not path.exists():
            raise FileNotFoundError(f"Frame image not found: {path}")
        frames.append(Image.open(path).convert("RGB"))
    return frames


class FeatureCacheDataset:
    def __init__(self, cache: dict[str, np.ndarray], require_labels: bool = True):
        from snu_order.features.cache import validate_feature_cache

        validate_feature_cache(cache, require_labels=require_labels)
        self.cache = cache
        self.require_labels = require_labels

    def __len__(self) -> int:
        return int(self.cache["ids"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("FeatureCacheDataset requires torch") from exc

        item: dict[str, Any] = {
            "id": str(self.cache["ids"][index]),
            "text_emb": torch.as_tensor(self.cache["text_emb"][index], dtype=torch.float32),
            "frame_emb": torch.as_tensor(self.cache["frame_emb"][index], dtype=torch.float32),
            "quality": torch.as_tensor(self.cache["quality"][index], dtype=torch.float32),
        }
        if self.require_labels:
            item["answer"] = torch.as_tensor(self.cache["answer"][index], dtype=torch.long)
            item["target_perm_idx"] = torch.as_tensor(self.cache["target_perm_idx"][index], dtype=torch.long)
            item["pairwise_labels"] = torch.as_tensor(self.cache["pairwise_labels"][index], dtype=torch.float32)
        return item

