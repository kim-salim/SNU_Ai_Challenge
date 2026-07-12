from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from snu_order.utils.config import get_by_path, load_config

from .checkpoint import load_lora24_checkpoint
from .collator import Qwen3VLCollator
from .dataset import Qwen3VLFrameOrderDataset
from .lockbox import guard_not_lockbox_path
from .modeling_lora24 import build_qwen3vl_lora24_model
from .train_lora24 import _model_device, evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    metadata_csv = args.metadata_csv or str(get_by_path(cfg, "data.valid_split"))
    guard_not_lockbox_path(metadata_csv, purpose="regular evaluation")
    output_dir = args.output_dir or str(Path(str(get_by_path(cfg, "output.dir"))) / "valid_a")
    model, processor = build_qwen3vl_lora24_model(cfg, frozen_probe=False)
    model, _ = load_lora24_checkpoint(args.checkpoint, model, strict=False, is_trainable=False)
    device = _model_device(model)
    model.classifier.to(device)
    dataset = Qwen3VLFrameOrderDataset(metadata_csv, str(get_by_path(cfg, "data.image_root", "data/raw")), training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=Qwen3VLCollator(processor), num_workers=0)
    metrics = evaluate_model(model, loader, device=device, output_dir=output_dir, profile=args.profile)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
