from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from snu_order.data.dataset import (
    load_frames_for_row,
    read_competition_csv,
    row_answer,
    row_id,
    row_sentence,
)
from snu_order.data.label import answer_to_pairwise_labels, answer_to_perm_index
from snu_order.encoders.siglip2_encoder import SigLIP2Encoder
from snu_order.features.cache import save_feature_cache
from snu_order.features.quality import compute_quality_features
from snu_order.utils.config import get_by_path, load_config


def _split_csv_path(cfg: dict[str, Any], split: str) -> str:
    if split == "train":
        return str(get_by_path(cfg, "data.split_train") or get_by_path(cfg, "data.train_csv"))
    if split == "valid":
        return str(get_by_path(cfg, "data.split_valid") or get_by_path(cfg, "data.valid_csv"))
    if split == "test":
        return str(get_by_path(cfg, "data.test_csv"))
    raise ValueError(f"Unsupported split: {split}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_split(cfg: dict[str, Any], split: str, csv_path: str | None = None, output: str | None = None) -> Path:
    selected_csv = csv_path or _split_csv_path(cfg, split)
    rows = read_competition_csv(selected_csv)
    if not rows:
        raise ValueError(f"No rows found in {selected_csv}")

    feature_dir = Path(str(get_by_path(cfg, "data.feature_dir", "data/features/siglip2_base_224")))
    output_path = Path(output) if output else feature_dir / f"{split}.npz"
    batch_size = int(get_by_path(cfg, "features.batch_size", get_by_path(cfg, "inference.batch_size", 32)))

    encoder = SigLIP2Encoder(
        model_dir=str(get_by_path(cfg, "encoder.local_dir", "weights/pretrained/siglip2_base_224")),
        dtype=str(get_by_path(cfg, "encoder.dtype", "fp16")),
    )

    ids: list[str] = []
    text_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []
    quality_chunks: list[np.ndarray] = []
    answers: list[list[int]] = []
    target_indices: list[int] = []
    pairwise: list[list[float]] = []
    require_labels = split in {"train", "valid"}

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_ids = [row_id(row) for row in batch_rows]
        sentences = [row_sentence(row) for row in batch_rows]
        frames = [load_frames_for_row(row, selected_csv) for row in batch_rows]
        text_emb, frame_emb = encoder.encode_batch(sentences, frames)
        quality = compute_quality_features(frames, text_emb, frame_emb)

        ids.extend(batch_ids)
        text_chunks.append(_to_numpy(text_emb).astype(np.float32))
        frame_chunks.append(_to_numpy(frame_emb).astype(np.float32))
        quality_chunks.append(_to_numpy(quality).astype(np.float32))

        if require_labels:
            for row in batch_rows:
                answer = row_answer(row)
                answers.append(answer)
                target_indices.append(answer_to_perm_index(answer))
                pairwise.append(answer_to_pairwise_labels(answer))

    data: dict[str, Any] = {
        "ids": np.asarray(ids, dtype=str),
        "text_emb": np.concatenate(text_chunks, axis=0),
        "frame_emb": np.concatenate(frame_chunks, axis=0),
        "quality": np.concatenate(quality_chunks, axis=0),
    }
    if require_labels:
        data.update(
            {
                "answer": np.asarray(answers, dtype=np.int64),
                "target_perm_idx": np.asarray(target_indices, dtype=np.int64),
                "pairwise_labels": np.asarray(pairwise, dtype=np.float32),
            }
        )
    save_feature_cache(output_path, data)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_path = extract_split(cfg, args.split, args.csv, args.output)
    print(f"saved feature cache: {output_path}")


if __name__ == "__main__":
    main()

