from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from snu_order.data.validate_submission import find_column, parse_answer
from snu_order.utils.io import read_csv_rows
from snu_order.vlm24.image_builder import load_sample_frames

from .permutations import (
    apply_frame_permutation,
    answer_to_perm_index,
    pairwise_labels_from_answer,
    position_labels_from_answer,
    uniform_shuffle_for_sample,
)
from .stage_pair_prompt import (
    LEGACY_POOLING_MODE,
    StagePairPromptSpec,
    build_stage_pair_message,
    build_stage_pair_prompt,
    prepare_stage_pair_multimodal_inputs,
)

ID_CANDIDATES = ["Id", "ID", "id", "sample_id", "sampleId"]
SENTENCE_CANDIDATES = ["Sentence", "sentence", "text", "caption", "prompt"]
ANSWER_CANDIDATES = ["Answer", "answer", "label", "true_answer"]


def build_single_frame_prompt(
    sentence: str,
    prompt_spec: StagePairPromptSpec | None = None,
) -> str:
    return build_stage_pair_prompt(
        sentence,
        prompt_spec or StagePairPromptSpec(
            pooling_mode=LEGACY_POOLING_MODE,
            anchor_text=None,
            anchor_prefix="\n",
            add_generation_prompt=True,
            enable_thinking=None,
            strict_template=False,
        ),
    )


def build_single_frame_message(prompt: str, image: Any) -> list[dict[str, Any]]:
    return build_stage_pair_message(prompt, image)


class Qwen3VLSingleFrameDataset:
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
        prompt_spec: StagePairPromptSpec | None = None,
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
        if max_samples is not None and int(max_samples) >= 0:
            self.rows = self.rows[: int(max_samples)]
        self.training = bool(training)
        self.augment_permutation = bool(augment_permutation)
        self.permutation_probability = float(permutation_probability)
        self.seed = int(seed)
        self.epoch = 0
        self.prompt_spec = prompt_spec

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
        return {
            "id": sample_id,
            "prompt": build_single_frame_prompt(str(row[self.sentence_col]), self.prompt_spec),
            "images": frames,
            "answer": answer,
            "target_perm_idx": answer_to_perm_index(answer),
            "stage_targets": position_labels_from_answer(answer),
            "pairwise_labels": pairwise_labels_from_answer(answer),
            "shuffle_idx": None if shuffle_idx is None else list(shuffle_idx),
        }


class Qwen3VLSingleFrameCollator:
    def __init__(
        self,
        processor: Any | None = None,
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        prompt_spec: StagePairPromptSpec | None = None,
        model_revision: str | None = None,
    ) -> None:
        self.processor = processor
        self.prompt_spec = prompt_spec or StagePairPromptSpec(
            pooling_mode=LEGACY_POOLING_MODE,
            anchor_text=None,
            anchor_prefix="\n",
            add_generation_prompt=bool(add_generation_prompt),
            enable_thinking=enable_thinking,
            strict_template=False,
        )
        self.model_revision = model_revision

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        batch = {
            "id": [str(sample["id"]) for sample in samples],
            "answer": torch.tensor([sample["answer"] for sample in samples], dtype=torch.long),
            "target_perm_idx": torch.tensor([sample["target_perm_idx"] for sample in samples], dtype=torch.long),
            "stage_targets": torch.tensor([sample["stage_targets"] for sample in samples], dtype=torch.long),
            "pairwise_labels": torch.tensor([sample["pairwise_labels"] for sample in samples], dtype=torch.float32),
            "batch_size": len(samples),
        }
        if self.processor is None:
            return batch
        conversations = []
        for sample in samples:
            images = list(sample["images"])
            if len(images) != 4:
                raise ValueError(f"Expected 4 images per sample, got {len(images)}")
            for image in images:
                conversations.append(build_single_frame_message(str(sample["prompt"]), image))
        prepared = prepare_stage_pair_multimodal_inputs(
            self.processor,
            conversations,
            self.prompt_spec,
            model_revision=self.model_revision,
        )
        batch["inputs"] = prepared.inputs
        if prepared.anchor_mask is not None:
            batch["anchor_mask"] = prepared.anchor_mask
            batch["anchor_spans"] = prepared.anchor_spans
        return batch


def move_stage_pair_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "inputs":
            moved[key] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in value.items()}
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class CachedStagePairDataset:
    def __init__(self, cache_path: str | Path, *, max_samples: int | None = None) -> None:
        payload = torch.load(cache_path, map_location="cpu")
        self.ids = [str(v) for v in payload["ids"]]
        self.frame_hidden = payload["frame_hidden"].float()
        self.answer = payload["answer"].long()
        self.target_perm_idx = payload["target_perm_idx"].long()
        if max_samples is not None and int(max_samples) >= 0:
            n = int(max_samples)
            self.ids = self.ids[:n]
            self.frame_hidden = self.frame_hidden[:n]
            self.answer = self.answer[:n]
            self.target_perm_idx = self.target_perm_idx[:n]
        if self.frame_hidden.ndim != 3 or self.frame_hidden.shape[1] != 4:
            raise ValueError(f"cached frame_hidden must have shape [N,4,H], got {tuple(self.frame_hidden.shape)}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        idx = int(index)
        return {
            "id": self.ids[idx],
            "frame_hidden": self.frame_hidden[idx],
            "answer": self.answer[idx],
            "target_perm_idx": self.target_perm_idx[idx],
        }


def collate_cached_stage_pair(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": [str(s["id"]) for s in samples],
        "frame_hidden": torch.stack([s["frame_hidden"] for s in samples], dim=0),
        "answer": torch.stack([s["answer"] for s in samples], dim=0),
        "target_perm_idx": torch.stack([s["target_perm_idx"] for s in samples], dim=0).long(),
        "batch_size": len(samples),
    }
