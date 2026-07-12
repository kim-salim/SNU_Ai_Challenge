from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from snu_order.utils.config import get_by_path, load_config

from .dataset_single_frame import CachedStagePairDataset, Qwen3VLSingleFrameCollator, Qwen3VLSingleFrameDataset, collate_cached_stage_pair
from .modeling_stage_pair import build_stage_pair_head_from_config, build_stage_pair_model_from_config, load_stage_pair_checkpoint
from .train_lora24 import _model_device
from .train_stage_pair import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["frozen_stage", "frozen_stage_set", "frozen_stage_pair", "qlora_stage_pair"], required=True)
    parser.add_argument("--source", choices=["base", "existing_lora"], default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg.setdefault("backbone", {})["source"] = args.source
    if args.mode.startswith("frozen"):
        source = str(get_by_path(cfg, "backbone.source", "base"))
        cache_path = Path(str(get_by_path(cfg, "cache.dir", "outputs/features/qwen3vl_stage_pair"))) / source / "valid_a.pt"
        dataset = CachedStagePairDataset(cache_path, max_samples=args.max_samples if args.max_samples >= 0 else None)
        model = build_stage_pair_head_from_config(cfg, hidden_size=int(dataset.frame_hidden.shape[-1]), backbone=None)
        processor = None
        loader = DataLoader(dataset, batch_size=int(get_by_path(cfg, "eval.batch_size", 128)), shuffle=False, collate_fn=collate_cached_stage_pair)
    else:
        cfg.setdefault("backbone", {})["frozen"] = True
        cfg.setdefault("backbone", {})["device_map"] = {"": int(args.device_index)}
        cfg.setdefault("lora", {})["enabled"] = False
        model, processor = build_stage_pair_model_from_config(cfg, live_backbone=True)
        dataset = Qwen3VLSingleFrameDataset(
            str(get_by_path(cfg, "data.valid_split")),
            str(get_by_path(cfg, "data.image_root", "data/raw")),
            training=False,
            max_samples=args.max_samples if args.max_samples >= 0 else None,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=Qwen3VLSingleFrameCollator(processor), num_workers=0)
    model, _ = load_stage_pair_checkpoint(args.checkpoint, model, strict=False, is_trainable=False)
    device = _model_device(model)
    if args.mode.startswith("frozen"):
        model.to(device)
    else:
        model.set_encoder.to(device)
        model.stage_head.to(device)
        if model.pair_head is not None:
            model.pair_head.to(device)
    out_dir = args.output_dir or str(Path(str(get_by_path(cfg, "output.dir", "outputs/experiments/qwen3vl_stage_pair"))) / "eval_valid_a")
    metrics = evaluate_model(model, loader, device=device, output_dir=out_dir, baseline_predictions=str(get_by_path(cfg, "baseline.valid_predictions", "")) or None)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
