from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from snu_order.utils.config import get_by_path, load_config

from .dataset_single_frame import Qwen3VLSingleFrameCollator, Qwen3VLSingleFrameDataset, move_stage_pair_batch_to_device
from .modeling_stage_pair import build_stage_pair_model_from_config
from .train_lora24 import _model_device


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_split(path: str | Path) -> None:
    text = str(path).lower()
    if "valid_b" in text or "test" in text:
        raise PermissionError(f"stage-pair feature cache is restricted to train_ab/valid_a, got: {path}")


def _cache_meta(cfg: dict[str, Any], source: str, split_path: str | Path, hidden: torch.Tensor) -> dict[str, Any]:
    adapter_path = Path(str(get_by_path(cfg, "backbone.existing_lora_path", "weights/qwen3vl_lora24/best"))) / "adapter" / "adapter_model.safetensors"
    return {
        "source": source,
        "base_model_path": str(get_by_path(cfg, "backbone.base_model_path")),
        "existing_lora_path": str(get_by_path(cfg, "backbone.existing_lora_path")),
        "adapter_sha256": _sha256(adapter_path) if adapter_path.exists() else None,
        "split_path": str(split_path),
        "split_sha256": _sha256(split_path),
        "hidden_dtype": str(hidden.dtype),
        "hidden_dim": int(hidden.shape[-1]),
        "sample_count": int(hidden.shape[0]),
    }


def build_cache(cfg: dict[str, Any], *, source: str, split: str, output_path: str | Path, max_samples: int = -1) -> dict[str, Any]:
    if split not in {"train_ab", "valid_a"}:
        raise ValueError(f"split must be train_ab or valid_a, got {split}")
    split_path = str(get_by_path(cfg, "data.train_split" if split == "train_ab" else "data.valid_split"))
    _safe_split(split_path)
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("backbone", {})["source"] = source
    cfg["backbone"]["frozen"] = True

    model, processor = build_stage_pair_model_from_config(cfg, live_backbone=True)
    device = _model_device(model)
    model.to(device)
    model.eval()
    dataset = Qwen3VLSingleFrameDataset(
        split_path,
        str(get_by_path(cfg, "data.image_root", "data/raw")),
        training=False,
        augment_permutation=False,
        max_samples=max_samples if int(max_samples) >= 0 else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(get_by_path(cfg, "cache.batch_size", 1)),
        shuffle=False,
        collate_fn=Qwen3VLSingleFrameCollator(processor),
        num_workers=int(get_by_path(cfg, "cache.num_workers", 0)),
    )
    ids: list[str] = []
    answers: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    start = time.perf_counter()
    first_100_time: float | None = None
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_stage_pair_batch_to_device(batch, device)
            frame_hidden = model.extract_frame_representations(batch["inputs"], batch_size=int(batch["batch_size"]))
            if not torch.isfinite(frame_hidden).all():
                raise FloatingPointError("NaN/Inf detected in cached frame hidden states")
            hidden_rows.append(frame_hidden.detach().to("cpu", dtype=torch.float32))
            answers.append(batch["answer"].detach().cpu())
            targets.append(batch["target_perm_idx"].detach().cpu())
            ids.extend([str(v) for v in batch["id"]])
            seen += len(batch["id"])
            if first_100_time is None and seen >= 100:
                first_100_time = time.perf_counter() - start
                total_est = first_100_time / max(seen, 1) * len(dataset)
                print(json.dumps({"processed": seen, "sec_first_block": first_100_time, "estimated_total_sec": total_est}), flush=True)
            elif seen % 500 == 0:
                print(json.dumps({"processed": seen}), flush=True)

    frame_hidden_tensor = torch.cat(hidden_rows, dim=0)
    answer_tensor = torch.cat(answers, dim=0).long()
    target_tensor = torch.cat(targets, dim=0).long()
    payload = {
        "ids": ids,
        "frame_hidden": frame_hidden_tensor,
        "answer": answer_tensor,
        "target_perm_idx": target_tensor,
        "meta": _cache_meta(cfg, source, split_path, frame_hidden_tensor),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    elapsed = time.perf_counter() - start
    summary = {
        "output_path": str(out),
        "sample_count": len(ids),
        "hidden_shape": list(frame_hidden_tensor.shape),
        "elapsed_sec": elapsed,
        "samples_per_sec": len(ids) / max(elapsed, 1e-9),
        "meta": payload["meta"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", choices=["base", "existing_lora"], required=True)
    parser.add_argument("--split", choices=["train_ab", "valid_a"], required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = args.output
    if output is None:
        output = Path(str(get_by_path(cfg, "cache.dir", "outputs/features/qwen3vl_stage_pair"))) / args.source / (
            "train_ab.pt" if args.split == "train_ab" else "valid_a.pt"
        )
    build_cache(cfg, source=args.source, split=args.split, output_path=output, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
