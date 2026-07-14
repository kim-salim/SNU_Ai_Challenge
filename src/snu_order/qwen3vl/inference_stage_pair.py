from __future__ import annotations

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from snu_order.data.validate_submission import find_column, validate_submission
from snu_order.pipeline.make_submission import save_submission
from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import read_csv_rows
from snu_order.utils.seed import seed_everything
from snu_order.vlm24.image_builder import load_sample_frames

from .calibration_stage_pair import (
    calibrated_structured_logits,
    checkpoint_calibration_bindings,
    load_calibration,
)
from .dataset_single_frame import build_single_frame_message, build_single_frame_prompt
from .frame_chunking import normalize_frame_chunk_size
from .modeling_stage_pair import build_stage_pair_model_from_config, load_stage_pair_checkpoint
from .permutations import perm_index_to_answer, validate_perm_index
from .stage_pair_prompt import (
    PreparedStagePairInputs,
    StagePairPromptSpec,
    checkpoint_processor_path,
    prepare_stage_pair_multimodal_inputs,
)
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
    *,
    spec: StagePairPromptSpec,
    model_revision: str | None,
) -> PreparedStagePairInputs:
    if len(frames) != 4:
        raise ValueError(f"Expected 4 frames, got {len(frames)}")
    conversations = [build_single_frame_message(prompt, frame) for frame in frames]
    return prepare_stage_pair_multimodal_inputs(
        processor,
        conversations,
        spec,
        model_revision=model_revision,
    )


def predict_rows(
    *,
    cfg: dict[str, Any],
    checkpoint: str | Path,
    metadata_csv: str | Path,
    image_root: str | Path,
    max_samples: int = -1,
    device_index: int = 0,
    calibration_path: str | Path | None = None,
    frame_chunk_size: int | None = None,
) -> tuple[list[str], list[list[int]], list[dict[str, Any]]]:
    seed_everything(int(get_by_path(cfg, "experiment.seed", 42)))
    rows = read_csv_rows(metadata_csv)
    id_col, sentence_col = _detect_columns(rows, metadata_csv)
    if max_samples is not None and int(max_samples) >= 0:
        rows = rows[: int(max_samples)]

    runtime_cfg = deepcopy(cfg)
    runtime_cfg.setdefault("backbone", {})["frozen"] = True
    runtime_cfg.setdefault("backbone", {})["device_map"] = {"": int(device_index)}
    processor_path = checkpoint_processor_path(checkpoint)
    model, processor = build_stage_pair_model_from_config(
        runtime_cfg,
        live_backbone=True,
        processor_path=processor_path,
    )
    model, _ = load_stage_pair_checkpoint(
        checkpoint,
        model,
        strict=True,
        is_trainable=False,
        cfg=runtime_cfg,
        processor=processor,
    )
    device = _model_device(model)
    model.set_encoder.to(device)
    model.stage_head.to(device)
    if model.pair_head is not None:
        model.pair_head.to(device)
    model.eval()

    ids: list[str] = []
    answers: list[list[int]] = []
    debug_rows: list[dict[str, Any]] = []
    calibration = (
        load_calibration(
            calibration_path,
            expected_bindings=checkpoint_calibration_bindings(checkpoint, runtime_cfg),
        )
        if calibration_path is not None
        else None
    )
    spec = StagePairPromptSpec.from_config(runtime_cfg)
    model_revision = str(get_by_path(runtime_cfg, "backbone.revision", "")) or None
    chunk_size = normalize_frame_chunk_size(
        frame_chunk_size
        if frame_chunk_size is not None
        else get_by_path(runtime_cfg, "inference.frame_chunk_size", None)
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for idx, row in enumerate(rows):
            sample_id = str(row[id_col])
            try:
                frames = load_sample_frames(row, image_root)
            except Exception as exc:
                raise RuntimeError(f"Failed to load frames for sample id={sample_id}") from exc

            prompt = build_single_frame_prompt(str(row[sentence_col]), spec)
            prepared = _prepare_inputs(
                processor,
                prompt,
                frames,
                spec=spec,
                model_revision=model_revision,
            )
            inputs = _tensor_inputs_to_device(prepared.inputs, device)
            anchor_mask = None if prepared.anchor_mask is None else prepared.anchor_mask.to(device)
            start = time.perf_counter()
            outputs = model(
                inputs=inputs,
                batch_size=1,
                anchor_mask=anchor_mask,
                frame_chunk_size=chunk_size,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start

            if calibration is None:
                scores = outputs["final_logits"].detach().float().cpu()[0]
            else:
                scores = calibrated_structured_logits(
                    outputs["stage_logits"], outputs["pair_logits"], calibration
                ).detach().float().cpu()[0]
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
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--profile-json", default=None)
    parser.add_argument("--frame-chunk-size", type=int, choices=[1, 2, 4], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_root = args.image_root or str(get_by_path(cfg, "data.image_root", "data/raw"))
    wall_start = time.perf_counter()
    ids, answers, debug_rows = predict_rows(
        cfg=cfg,
        checkpoint=args.checkpoint,
        metadata_csv=args.metadata_csv,
        image_root=image_root,
        max_samples=args.max_samples,
        device_index=args.device_index,
        calibration_path=args.calibration,
        frame_chunk_size=args.frame_chunk_size,
    )
    full_run = args.max_samples is None or args.max_samples < 0
    save_submission(ids, answers, args.output_csv, reference=args.sample_submission if full_run else None)
    if full_run:
        validate_submission(args.output_csv, args.sample_submission)
    if args.debug_csv:
        write_debug_csv(args.debug_csv, debug_rows)
    wall_time = time.perf_counter() - wall_start
    latencies = [float(row["latency_sec"]) for row in debug_rows]
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)) if ordered else 0
    forward_time = sum(latencies)
    profile = {
        "sample_count": len(debug_rows),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        "frame_chunk_size": normalize_frame_chunk_size(
            args.frame_chunk_size
            if args.frame_chunk_size is not None
            else get_by_path(cfg, "inference.frame_chunk_size", None)
        ),
        "model_forward_time_sec": forward_time,
        "end_to_end_wall_clock_sec": wall_time,
        "mean_sample_latency_sec": forward_time / max(len(latencies), 1),
        "p95_sample_latency_sec": ordered[p95_index] if ordered else 0.0,
        "estimated_full_test_runtime_sec": wall_time,
    }
    if args.profile_json:
        from snu_order.utils.io import write_json

        write_json(args.profile_json, profile)
    print(json.dumps({"profile": profile}, ensure_ascii=False))
    print(f"saved submission: {args.output_csv}")


if __name__ == "__main__":
    main()
