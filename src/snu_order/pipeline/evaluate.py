from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from snu_order.data.metric import exact_match_accuracy, pairwise_accuracy, top1_top2_margin
from snu_order.features.cache import load_feature_cache, validate_feature_cache
from snu_order.order.answer_convert import perm_to_answer
from snu_order.order.permutation24 import index_to_perm
from snu_order.pipeline.model_io import load_modules_from_checkpoint
from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.device import get_device
from snu_order.utils.io import write_csv_rows, write_json


def _to_tensor(array: np.ndarray, device: str) -> Any:
    import torch

    return torch.as_tensor(array, dtype=torch.float32, device=device)


def predict_from_cache(
    cache: dict[str, np.ndarray],
    projector: Any,
    ranker: Any,
    device: str,
    batch_size: int = 256,
) -> tuple[list[list[int]], np.ndarray]:
    import torch

    validate_feature_cache(cache, require_labels="answer" in cache)
    projector.eval()
    ranker.eval()
    scores_list: list[np.ndarray] = []
    pred_answers: list[list[int]] = []
    with torch.no_grad():
        n = int(cache["ids"].shape[0])
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            text_emb = _to_tensor(cache["text_emb"][start:end], device)
            frame_emb = _to_tensor(cache["frame_emb"][start:end], device)
            quality = None if getattr(projector, "quality_dim", None) == 0 else _to_tensor(cache["quality"][start:end], device)
            frame_tokens = projector(text_emb, frame_emb, quality)
            scores = ranker(frame_tokens)
            scores_np = scores.detach().cpu().numpy()
            scores_list.append(scores_np)
            for idx in np.argmax(scores_np, axis=1):
                pred_answers.append(perm_to_answer(index_to_perm(int(idx))))
    return pred_answers, np.concatenate(scores_list, axis=0)


def evaluate_cache(
    cache: dict[str, np.ndarray],
    projector: Any,
    ranker: Any,
    device: str,
    batch_size: int = 256,
) -> dict[str, Any]:
    pred_answers, scores = predict_from_cache(cache, projector, ranker, device, batch_size)
    if "answer" not in cache:
        return {"pred_answers": pred_answers, "scores": scores, "metrics": {}}
    true_answers = cache["answer"].astype(int).tolist()
    metrics = {
        "exact_match": exact_match_accuracy(pred_answers, true_answers),
        "pairwise_acc": pairwise_accuracy(pred_answers, true_answers),
        "top1_top2_margin": top1_top2_margin(scores),
    }
    return {"pred_answers": pred_answers, "scores": scores, "metrics": metrics}


def write_eval_outputs(
    output_dir: str | Path,
    ids: list[str],
    pred_answers: list[list[int]],
    true_answers: list[list[int]],
    metrics: dict[str, float],
) -> None:
    rows = []
    errors = []
    for sample_id, pred, true in zip(ids, pred_answers, true_answers, strict=True):
        correct = list(pred) == list(true)
        row = {
            "Id": sample_id,
            "pred_answer": json.dumps(pred, separators=(",", ":")),
            "true_answer": json.dumps(true, separators=(",", ":")),
            "correct": int(correct),
        }
        rows.append(row)
        if not correct:
            errors.append(row)
    out = Path(output_dir)
    write_csv_rows(out / "valid_predictions.csv", rows, ["Id", "pred_answer", "true_answer", "correct"])
    write_csv_rows(out / "valid_errors.csv", errors, ["Id", "pred_answer", "true_answer", "correct"])
    write_json(out / "eval_metrics.json", metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    feature_dir = Path(str(get_by_path(cfg, "data.feature_dir", "data/features/siglip2_base_224")))
    cache_path = Path(args.cache) if args.cache else feature_dir / "valid.npz"
    checkpoint = args.checkpoint or str(get_by_path(cfg, "output.checkpoint"))
    if checkpoint is None:
        raise ValueError("Config must set output.checkpoint")

    cache = load_feature_cache(cache_path)
    projector, ranker, _pairwise, _checkpoint = load_modules_from_checkpoint(checkpoint, cfg, device)
    result = evaluate_cache(
        cache,
        projector,
        ranker,
        device,
        batch_size=int(get_by_path(cfg, "train.batch_size", 128)),
    )
    ids = [str(v) for v in cache["ids"].tolist()]
    true_answers = cache["answer"].astype(int).tolist()
    output_dir = str(get_by_path(cfg, "output.dir", "outputs/experiments/eval"))
    write_eval_outputs(output_dir, ids, result["pred_answers"], true_answers, result["metrics"])
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
