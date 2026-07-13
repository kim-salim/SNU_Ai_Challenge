from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from snu_order.data.validate_submission import find_column, validate_submission
from snu_order.pipeline.make_submission import save_submission
from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import read_csv_rows
from snu_order.vlm24.image_builder import load_sample_frames

from .dataset_single_frame import build_single_frame_message, build_single_frame_prompt
from .modeling_stage_pair import build_stage_pair_model_from_config, load_stage_pair_checkpoint
from .permutations import perm_index_to_answer, validate_perm_index
from .train_lora24 import _model_device


ID_CANDIDATES = ["Id", "ID", "id", "sample_id", "sampleId"]
SENTENCE_CANDIDATES = ["Sentence", "sentence", "text", "caption", "prompt"]


def _detect_columns(rows: list[dict[str, str]], metadata_csv: str | Path) -> tuple[str, str]:
    if not rows:
        raise ValueError(f"metadata CSV has no rows: {metadata_csv}")
    columns = list(rows[0].keys())
    id_col = find_column(columns, ID_CANDIDATES)
    sentence_col = find_column(columns, SENTENCE_CANDIDATES)
    if id_col is None:
        raise ValueError(f"Could not detect id column in {metadata_csv}. Available columns: {columns}")
    if sentence_col is None:
        raise ValueError(f"Could not detect sentence column in {metadata_csv}. Available columns: {columns}")
    return id_col, sentence_col


def _tensor_inputs_to_device(inputs: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _prepare_inputs(
    processor: Any,
    prompt: str,
    frames: list[Any],
    device: torch.device | str,
    *,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    if len(frames) != 4:
        raise ValueError(f"Expected 4 frames, got {len(frames)}")
    conversations = [build_single_frame_message(prompt, frame) for frame in frames]
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "padding": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = bool(enable_thinking)
    try:
        inputs = processor.apply_chat_template(conversations, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        inputs = processor.apply_chat_template(conversations, **template_kwargs)
    return _tensor_inputs_to_device(dict(inputs), device)


def predict_rows(
    *,
    cfg: dict[str, Any],
    checkpoint: str | Path,
    metadata_csv: str | Path,
    image_root: str | Path,
    max_samples: int = -1,
    device_index: int = 0,
) -> tuple[list[str], list[list[int]], list[dict[str, Any]]]:
    rows = read_csv_rows(metadata_csv)
    id_col, sentence_col = _detect_columns(rows, metadata_csv)
    if max_samples is not None and int(max_samples) >= 0:
        rows = rows[: int(max_samples)]

    cfg.setdefault("backbone", {})["frozen"] = True
    cfg.setdefault("backbone", {})["device_map"] = {"": int(device_index)}
    cfg.setdefault("lora", {})["enabled"] = False
    model, processor = build_stage_pair_model_from_config(cfg, live_backbone=True)
    model, _ = load_stage_pair_checkpoint(checkpoint, model, strict=False, is_trainable=False)
    device = _model_device(model)
    model.set_encoder.to(device)
    model.stage_head.to(device)
    if model.pair_head is not None:
        model.pair_head.to(device)
    model.eval()

    ids: list[str] = []
    answers: list[list[int]] = []
    debug_rows: list[dict[str, Any]] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        enable_thinking = get_by_path(cfg, "prompt.enable_thinking", None)
        for idx, row in enumerate(rows):
            sample_id = str(row[id_col])
            try:
                frames = load_sample_frames(row, image_root)
            except Exception as exc:
                raise RuntimeError(f"Failed to load frames for sample id={sample_id}") from exc

            prompt = build_single_frame_prompt(str(row[sentence_col]))
            inputs = _prepare_inputs(processor, prompt, frames, device, enable_thinking=enable_thinking)
            start = time.perf_counter()
            outputs = model(inputs=inputs, batch_size=1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start

            scores = outputs["final_logits"].detach().float().cpu()[0]
            pred_perm_idx = validate_perm_index(int(scores.argmax().item()))
            answer = perm_index_to_answer(pred_perm_idx)
            top2 = torch.topk(scores, k=2)

            ids.append(sample_id)
            answers.append(answer)
            debug_rows.append(
                {
                    "Id": sample_id,
                    "pred_perm_idx": pred_perm_idx,
                    "pred_answer": json.dumps(answer),
                    "top1_score": float(top2.values[0].item()),
                    "top2_score": float(top2.values[1].item()),
                    "margin": float((top2.values[0] - top2.values[1]).item()),
                    "latency_sec": latency,
                }
            )
            if (idx + 1) % 100 == 0:
                print(json.dumps({"processed": idx + 1, "last_latency_sec": latency}), flush=True)

    return ids, answers, debug_rows


def write_debug_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Id", "pred_perm_idx", "pred_answer", "top1_score", "top2_score", "margin", "latency_sec"]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-csv", default="data/raw/test.csv")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--sample-submission", default="data/raw/sample_submission.csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--debug-csv", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_root = args.image_root or str(get_by_path(cfg, "data.image_root", "data/raw"))
    ids, answers, debug_rows = predict_rows(
        cfg=cfg,
        checkpoint=args.checkpoint,
        metadata_csv=args.metadata_csv,
        image_root=image_root,
        max_samples=args.max_samples,
        device_index=args.device_index,
    )
    save_submission(ids, answers, args.output_csv)
    validate_submission(args.output_csv, args.sample_submission)
    if args.debug_csv:
        write_debug_csv(args.debug_csv, debug_rows)
    print(f"saved submission: {args.output_csv}")


if __name__ == "__main__":
    main()
