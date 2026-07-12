from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from torch.utils.data import DataLoader

from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import write_json

from .checkpoint import load_lora24_checkpoint
from .collator import Qwen3VLCollator
from .dataset import Qwen3VLFrameOrderDataset
from .lockbox import build_lockbox_gate, check_lockbox_unlock, check_valid_a_threshold
from .modeling_lora24 import build_qwen3vl_lora24_model
from .train_lora24 import _model_device, evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--unlock-valid-b", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    check_lockbox_unlock(args.unlock_valid_b)
    cfg = load_config(args.config)
    min_valid_a = float(get_by_path(cfg, "lockbox.min_valid_a_accuracy", 0.33))
    valid_a_best = check_valid_a_threshold(args.checkpoint, min_valid_a_accuracy=min_valid_a, force=args.force)
    output_dir = Path(args.output_dir or str(Path(str(get_by_path(cfg, "output.dir"))) / "valid_b_lockbox"))
    model, processor = build_qwen3vl_lora24_model(cfg, frozen_probe=False)
    model, _ = load_lora24_checkpoint(args.checkpoint, model, strict=False, is_trainable=False)
    device = _model_device(model)
    model.classifier.to(device)
    dataset = Qwen3VLFrameOrderDataset(
        str(get_by_path(cfg, "data.lockbox_split")),
        str(get_by_path(cfg, "data.image_root", "data/raw")),
        training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=Qwen3VLCollator(processor), num_workers=0)
    metrics = evaluate_model(model, loader, device=device, output_dir=output_dir, profile=True)
    gate = build_lockbox_gate(
        sample_count=int(metrics["sample_count"]),
        correct_count=int(metrics["correct_count"]),
        exact_match=float(metrics["exact_match"]),
        required_accuracy=float(get_by_path(cfg, "lockbox.min_valid_b_accuracy", 0.30)),
        valid_a_best_accuracy=valid_a_best,
        checkpoint_path=args.checkpoint,
        evaluated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    write_json(Path(str(get_by_path(cfg, "output.dir"))) / "lockbox_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
