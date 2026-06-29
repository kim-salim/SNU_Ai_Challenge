from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from snu_order.data.dataset import load_frames_for_row, read_competition_csv, row_id, row_sentence
from snu_order.data.validate_submission import validate_submission
from snu_order.encoders.siglip2_encoder import SigLIP2Encoder
from snu_order.features.quality import compute_quality_features
from snu_order.order.answer_convert import perm_to_answer
from snu_order.order.permutation24 import index_to_perm
from snu_order.pipeline.make_submission import save_submission
from snu_order.pipeline.model_io import load_modules_from_checkpoint
from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.device import get_device
from snu_order.utils.seed import seed_everything


def run_inference(cfg: dict[str, Any]) -> Path:
    import torch

    seed_everything(int(get_by_path(cfg, "experiment.seed", 42)))
    device = get_device()
    test_csv = str(get_by_path(cfg, "data.test_csv"))
    if not test_csv:
        raise ValueError("Config must set data.test_csv")
    rows = read_competition_csv(test_csv)
    if not rows:
        raise ValueError(f"No rows found in {test_csv}")

    checkpoint_path = str(get_by_path(cfg, "output.checkpoint"))
    if not checkpoint_path:
        raise ValueError("Config must set output.checkpoint")
    projector, ranker, _pairwise, _checkpoint = load_modules_from_checkpoint(checkpoint_path, cfg, device)
    projector.eval()
    ranker.eval()

    encoder = SigLIP2Encoder(
        model_dir=str(get_by_path(cfg, "encoder.local_dir", "weights/pretrained/siglip2_base_224")),
        device=device,
        dtype=str(get_by_path(cfg, "encoder.dtype", "fp16")),
    )

    batch_size = int(get_by_path(cfg, "inference.batch_size", 32))
    ids: list[str] = []
    answers: list[list[int]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_ids = [row_id(row) for row in batch_rows]
            sentences = [row_sentence(row) for row in batch_rows]
            frames = [load_frames_for_row(row, test_csv) for row in batch_rows]
            text_emb, frame_emb = encoder.encode_batch(sentences, frames)
            quality = compute_quality_features(frames, text_emb, frame_emb)
            frame_tokens = projector(text_emb.to(device), frame_emb.to(device), quality.to(device))
            scores = ranker(frame_tokens).detach().cpu().numpy()
            pred_idx = np.argmax(scores, axis=1)
            ids.extend(batch_ids)
            answers.extend([perm_to_answer(index_to_perm(int(idx))) for idx in pred_idx])

    output = Path(str(get_by_path(cfg, "output.submission", "outputs/submissions/final_submission.csv")))
    save_submission(ids, answers, output)
    reference = get_by_path(cfg, "data.sample_submission_csv")
    if reference and Path(str(reference)).exists():
        validate_submission(output, str(reference))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = run_inference(cfg)
    print(f"saved submission: {output}")


if __name__ == "__main__":
    main()

