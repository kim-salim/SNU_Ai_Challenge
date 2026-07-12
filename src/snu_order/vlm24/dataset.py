from __future__ import annotations

from pathlib import Path
from typing import Any

from snu_order.data.validate_submission import find_column, parse_answer
from snu_order.utils.io import read_csv_rows
from snu_order.vlm24.image_builder import detect_frame_columns, load_sample_frames


ID_CANDIDATES = ["Id", "ID", "id", "sample_id", "sampleId"]
SENTENCE_CANDIDATES = ["Sentence", "sentence", "text", "caption", "prompt"]
ANSWER_CANDIDATES = ["Answer", "answer", "label", "true_answer"]


class VLM24MetadataDataset:
    def __init__(
        self,
        metadata_csv: str | Path,
        image_root: str | Path,
        has_answer: bool | None = None,
        max_samples: int | None = None,
    ):
        self.metadata_csv = Path(metadata_csv)
        self.image_root = Path(image_root)
        self.rows = read_csv_rows(self.metadata_csv)
        if not self.rows:
            raise ValueError(f"metadata CSV has no rows: {self.metadata_csv}")
        first = self.rows[0]
        columns = list(first.keys())
        self.id_col = find_column(columns, ID_CANDIDATES)
        self.sentence_col = find_column(columns, SENTENCE_CANDIDATES)
        self.answer_col = find_column(columns, ANSWER_CANDIDATES)
        self.frame_cols = detect_frame_columns(columns)
        if self.id_col is None:
            raise ValueError(f"Could not detect id column. Available columns: {columns}")
        if self.sentence_col is None:
            raise ValueError(f"Could not detect sentence column. Available columns: {columns}")
        if has_answer is None:
            has_answer = self.answer_col is not None and str(first.get(self.answer_col, "")).strip() != ""
        self.has_answer = bool(has_answer)
        if self.has_answer and self.answer_col is None:
            raise ValueError(f"Could not detect answer column. Available columns: {columns}")
        if max_samples is not None and max_samples >= 0:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item: dict[str, Any] = {
            "id": str(row[self.id_col]),
            "sentence": str(row[self.sentence_col]),
            "frames": load_sample_frames(row, self.image_root),
        }
        if self.has_answer and self.answer_col is not None:
            item["answer"] = parse_answer(row[self.answer_col])
        else:
            item["answer"] = None
        return item
