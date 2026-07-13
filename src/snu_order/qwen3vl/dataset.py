from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from snu_order.data.validate_submission import find_column, parse_answer
from snu_order.utils.io import read_csv_rows
from snu_order.vlm24.image_builder import load_sample_frames

from .permutations import (
    PERMS,
    apply_frame_permutation,
    answer_to_perm_index,
    pairwise_labels_from_answer,
    position_labels_from_answer,
    uniform_shuffle_for_sample,
)
from .prompts import build_classifier_prompt

ID_CANDIDATES = ["Id", "ID", "id", "sample_id", "sampleId"]
SENTENCE_CANDIDATES = ["Sentence", "sentence", "text", "caption", "prompt"]
ANSWER_CANDIDATES = ["Answer", "answer", "label", "true_answer"]


class Qwen3VLFrameOrderDataset:
    def __init__(
        self,
        metadata_csv: str | Path,
        image_root: str | Path,
        *,
        training: bool = False,
        augment_permutation: bool = False,
        permutation_probability: float = 1.0,
        seed: int = 42,
        max_samples: int | None = None,
        sample_indices: list[int] | None = None,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.image_root = Path(image_root)
        self.rows = read_csv_rows(self.metadata_csv)
        if not self.rows:
            raise ValueError(f"metadata CSV has no rows: {self.metadata_csv}")
        columns = list(self.rows[0].keys())
        self.id_col = find_column(columns, ID_CANDIDATES)
        self.sentence_col = find_column(columns, SENTENCE_CANDIDATES)
        self.answer_col = find_column(columns, ANSWER_CANDIDATES)
        if self.id_col is None:
            raise ValueError(f"Could not detect id column. Available columns: {columns}")
        if self.sentence_col is None:
            raise ValueError(f"Could not detect sentence column. Available columns: {columns}")
        if self.answer_col is None:
            raise ValueError(f"Could not detect answer column. Available columns: {columns}")
        if sample_indices is not None:
            self.rows = [self.rows[int(i)] for i in sample_indices]
        if max_samples is not None and max_samples >= 0:
            self.rows = self.rows[: int(max_samples)]
        self.training = bool(training)
        self.augment_permutation = bool(augment_permutation)
        self.permutation_probability = float(permutation_probability)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _maybe_augment(self, index: int, frames: list[Any], answer: list[int]) -> tuple[list[Any], list[int], tuple[int, int, int, int] | None]:
        if not self.training or not self.augment_permutation:
            return frames, answer, None
        rng = random.Random(self.seed + index * 9176 + self.epoch * 1_000_003)
        if rng.random() > self.permutation_probability:
            return frames, answer, None
        shuffle_idx = uniform_shuffle_for_sample(self.seed, index, self.epoch)
        new_frames, new_answer = apply_frame_permutation(frames, answer, shuffle_idx)
        return new_frames, new_answer, shuffle_idx

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        sample_id = str(row[self.id_col])
        try:
            frames = load_sample_frames(row, self.image_root)
        except Exception as exc:
            raise RuntimeError(f"Failed to load frames for sample id={sample_id}") from exc
        answer = parse_answer(row[self.answer_col])
        frames, answer, shuffle_idx = self._maybe_augment(int(index), frames, answer)
        target_perm_idx = answer_to_perm_index(answer)
        return {
            "id": sample_id,
            "prompt": build_classifier_prompt(str(row[self.sentence_col])),
            "images": frames,
            "answer": answer,
            "target_perm_idx": target_perm_idx,
            "position_labels": position_labels_from_answer(answer),
            "pairwise_labels": pairwise_labels_from_answer(answer),
            "shuffle_idx": None if shuffle_idx is None else list(shuffle_idx),
        }


def fixed_subset_indices(dataset_len: int, size: int, seed: int) -> list[int]:
    n = min(int(size), int(dataset_len))
    rng = random.Random(int(seed))
    indices = list(range(int(dataset_len)))
    rng.shuffle(indices)
    return indices[:n]
