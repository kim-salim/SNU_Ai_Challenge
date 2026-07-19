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

from .dataset_single_frame import (
    Qwen3VLSingleFrameCollator,
    Qwen3VLSingleFrameDataset,
    move_stage_pair_batch_to_device,
)
from .modeling_stage_pair import build_stage_pair_model_from_config, load_stage_pair_checkpoint
from .permutations import perm_index_to_answer
from .stage_pair_checkpoint import verify_stage_pair_checkpoint_files
from .stage_pair_prompt import StagePairPromptSpec, checkpoint_processor_path
from .train_lora24 import _model_device
from .train_stage_pair import _move_stage_pair_modules


def verify_checkpoint_forward(
    *,
    cfg: dict[str, Any],
    checkpoint: str | Path,
    metadata_csv: str | Path,
    image_root: str | Path,
    max_samples: int = 8,
) -> dict[str, Any]:
    runtime_cfg = deepcopy(cfg)
    runtime_cfg.setdefault("backbone", {})["frozen"] = True
    processor_path = checkpoint_processor_path(checkpoint)
    model, processor = build_stage_pair_model_from_config(
        runtime_cfg,
        live_backbone=True,
        processor_path=processor_path,
    )
    manifest = verify_stage_pair_checkpoint_files(
        checkpoint,
        runtime_cfg=runtime_cfg,
        processor=processor,
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
    _move_stage_pair_modules(model, device)
    prompt_spec = StagePairPromptSpec.from_config(runtime_cfg)
    dataset = Qwen3VLSingleFrameDataset(
        metadata_csv,
        image_root,
        training=False,
        max_samples=max_samples,
        prompt_spec=prompt_spec,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=Qwen3VLSingleFrameCollator(
            processor,
            prompt_spec=prompt_spec,
            model_revision=str(get_by_path(runtime_cfg, "backbone.revision")),
        ),
    )
    predictions: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_stage_pair_batch_to_device(batch, device)
            outputs = model(
                inputs=batch["inputs"],
                batch_size=int(batch["batch_size"]),
                anchor_mask=batch.get("anchor_mask"),
            )
            logits = outputs["final_logits"]
            if logits.shape != (1, 24) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Checkpoint verification produced invalid logits: {tuple(logits.shape)}")
            prediction = int(logits.argmax(dim=1).item())
            if sorted(perm_index_to_answer(prediction)) != [1, 2, 3, 4]:
                raise RuntimeError(f"Checkpoint verification produced invalid prediction index: {prediction}")
            predictions.append(prediction)
    if not predictions:
        raise RuntimeError("Checkpoint verification did not process any validation rows")
    return {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "checkpoint_format_version": manifest["checkpoint_format_version"],
        "sample_count": len(predictions),
        "finite_logits": True,
        "prediction_indices": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--base-model-revision", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.base_model_path:
        cfg.setdefault("backbone", {})["base_model_path"] = str(args.base_model_path)
    if args.base_model_revision:
        cfg.setdefault("backbone", {})["revision"] = str(args.base_model_revision)
    result = verify_checkpoint_forward(
        cfg=cfg,
        checkpoint=args.checkpoint,
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        max_samples=args.max_samples,
    )
    if args.output_json:
        write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
