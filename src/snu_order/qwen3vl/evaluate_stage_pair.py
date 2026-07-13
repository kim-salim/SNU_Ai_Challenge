from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import write_json

from .calibration_stage_pair import (
    calibrated_structured_logits,
    load_calibration,
    load_raw_stage_pair_logits,
)
from .dataset_single_frame import Qwen3VLSingleFrameCollator, Qwen3VLSingleFrameDataset
from .metrics_stage_pair import compute_stage_pair_metrics, write_stage_pair_artifacts
from .modeling_stage_pair import build_stage_pair_model_from_config, load_stage_pair_checkpoint
from .stage_pair_prompt import StagePairPromptSpec
from .train_lora24 import _model_device
from .train_stage_pair import evaluate_model


def evaluate_checkpoint(
    *,
    cfg: dict[str, Any],
    checkpoint: str | Path,
    metadata_csv: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    max_samples: int,
    split_name: str,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    normalized_split = split_name.lower().replace("-", "_")
    if normalized_split not in {"valid_a", "valid_b"}:
        raise RuntimeError(f"Evaluation split must be valid_a or valid_b, got {split_name!r}")
    if normalized_split == "valid_b" and calibration_path is None:
        raise RuntimeError("valid-B evaluation requires fixed valid-A calibration parameters")
    runtime_cfg = deepcopy(cfg)
    runtime_cfg.setdefault("backbone", {})["frozen"] = True
    model, processor = build_stage_pair_model_from_config(runtime_cfg, live_backbone=True)
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
    dataset = Qwen3VLSingleFrameDataset(
        metadata_csv,
        image_root,
        training=False,
        max_samples=max_samples if max_samples >= 0 else None,
    )
    spec = StagePairPromptSpec.from_config(runtime_cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=Qwen3VLSingleFrameCollator(
            processor,
            prompt_spec=spec,
            model_revision=str(get_by_path(runtime_cfg, "backbone.revision")),
        ),
    )
    output = Path(output_dir)
    raw_path = output / "raw_stage_pair_logits.pt"
    raw_metrics = evaluate_model(
        model,
        loader,
        device=device,
        output_dir=output,
        raw_logits_path=raw_path,
    )
    result: dict[str, Any] = {"split": normalized_split, "raw_metrics": raw_metrics, "raw_logits": str(raw_path)}
    if calibration_path is not None:
        payload = load_raw_stage_pair_logits(raw_path)
        parameters = load_calibration(calibration_path)
        logits = calibrated_structured_logits(payload["stage_logits"], payload["pair_logits"], parameters)
        metrics = compute_stage_pair_metrics(
            logits,
            payload["target_perm_idx"],
            stage_logits=payload["stage_logits"],
            pair_logits=payload["pair_logits"],
            answer=payload["answer"],
        )
        calibrated_dir = output / "calibrated"
        write_stage_pair_artifacts(
            calibrated_dir,
            [str(value) for value in payload["ids"]],
            logits,
            payload["target_perm_idx"],
            metrics,
            stage_logits=payload["stage_logits"],
            pair_logits=payload["pair_logits"],
        )
        result["calibrated_metrics"] = metrics
        result["calibration"] = parameters.as_dict()
    write_json(output / "evaluation_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--split-name", choices=["valid_a", "valid_b"], default="valid_a")
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = evaluate_checkpoint(
        cfg=cfg,
        checkpoint=args.checkpoint,
        metadata_csv=args.metadata_csv,
        image_root=args.image_root or str(get_by_path(cfg, "data.image_root", "data/raw")),
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        split_name=args.split_name,
        calibration_path=args.calibration,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
