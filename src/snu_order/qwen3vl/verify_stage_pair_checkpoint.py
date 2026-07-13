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
from .stage_pair_prompt import StagePairPromptSpec
from .train_lora24 import _model_device


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
    model, processor = build_stage_pair_model_from_config(runtime_cfg, live_backbone=True)
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
    model.set_encoder.to(device)
    model.stage_head.to(device)
    if model.pair_head is not None:
        model.pair_head.to(device)
    dataset = Qwen3VLSingleFrameDataset(
        metadata_csv,
        image_root,
        training=False,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=Qwen3VLSingleFrameCollator(
            processor,
            prompt_spec=StagePairPromptSpec.from_config(runtime_cfg),
            model_revision=str(get_by_path(runtime_cfg, "backbone.revision")),
        ),
    )
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
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
    args = parser.parse_args()
    result = verify_checkpoint_forward(
        cfg=load_config(args.config),
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
