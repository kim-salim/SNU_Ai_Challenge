from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from snu_order.utils.config import get_by_path, load_config

from .dataset_single_frame import (
    Qwen3VLSingleFrameCollator,
    Qwen3VLSingleFrameDataset,
    move_stage_pair_batch_to_device,
)
from .modeling_stage_pair import build_stage_pair_model_from_config
from .qwen35_27b_port import quantization_contract, validate_qwen35_27b_architecture
from .stage_pair_prompt import StagePairPromptSpec, build_prompt_fingerprint
from .train_lora24 import _model_device


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def extract(
    cfg: dict[str, Any],
    *,
    metadata_csv: str,
    image_root: str,
    split_name: str,
    output: str,
    max_samples: int,
) -> dict[str, Any]:
    lowered = str(metadata_csv).lower()
    if "test" in lowered or "sample_submission" in lowered:
        raise RuntimeError("Official Test paths are forbidden during feature extraction")
    runtime = deepcopy(cfg)
    runtime["backbone"]["frozen"] = True
    runtime["lora"]["enabled"] = False
    model, processor = build_stage_pair_model_from_config(runtime, live_backbone=True)
    architecture = validate_qwen35_27b_architecture(model.backbone.config)
    model.eval()
    device = _model_device(model)
    model.frame_projector.to(device=device, dtype=torch.bfloat16)
    spec = StagePairPromptSpec.from_config(runtime)
    dataset = Qwen3VLSingleFrameDataset(
        metadata_csv,
        image_root,
        training=False,
        max_samples=max_samples if max_samples >= 0 else None,
        prompt_spec=spec,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=Qwen3VLSingleFrameCollator(
            processor,
            prompt_spec=spec,
            model_revision=str(get_by_path(runtime, "backbone.revision", "")) or None,
        ),
    )
    ids: list[str] = []
    frames: list[torch.Tensor] = []
    answers: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            moved = move_stage_pair_batch_to_device(batch, device)
            hidden = model.extract_frame_representations(
                moved["inputs"],
                batch_size=int(moved["batch_size"]),
                anchor_mask=moved.get("anchor_mask"),
                frame_chunk_size=int(get_by_path(runtime, "inference.frame_chunk_size", 1)),
            )
            ids.extend(str(value) for value in batch["id"])
            frames.append(hidden.detach().to(dtype=torch.bfloat16, device="cpu"))
            answers.append(batch["answer"].detach().long().cpu())
            targets.append(batch["target_perm_idx"].detach().long().cpu())
    fingerprint = build_prompt_fingerprint(runtime, processor)
    identity = {
        "architecture": str(get_by_path(runtime, "architecture.id")),
        "base_model_path": str(get_by_path(runtime, "backbone.base_model_path")),
        "base_model_revision": get_by_path(runtime, "backbone.revision", None),
        "backbone_config_sha256": str(architecture["config_sha256"]),
        "quantization": quantization_contract(runtime),
        "processor_tokenizer_fingerprint": fingerprint["tokenizer_config_sha256"],
        "prompt_fingerprint": fingerprint["rendered_prompt_sha256"],
        "anchor_token_ids": fingerprint["anchor_token_ids"],
        "image_policy": str(get_by_path(runtime, "data.image_policy")),
        "text_policy": str(get_by_path(runtime, "data.text_policy")),
        "hidden_width": 5120,
        "dtype": "torch.bfloat16",
        "split_name": split_name,
        "split_manifest_sha256": _sha(metadata_csv),
        "source_git_head": _git("rev-parse", "HEAD"),
        "source_git_tree": _git("rev-parse", "HEAD^{tree}"),
    }
    payload = {
        "format_version": 1,
        "ids": ids,
        "frame_hidden": torch.cat(frames, dim=0),
        "answer": torch.cat(answers, dim=0),
        "target_perm_idx": torch.cat(targets, dim=0),
        "cache_identity": identity,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {
        "status": "PASS",
        "output": str(destination.resolve()),
        "sample_count": len(ids),
        "shape": list(payload["frame_hidden"].shape),
        "dtype": str(payload["frame_hidden"].dtype),
        "cache_sha256": _sha(destination),
        "cache_identity": identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--split-name", choices=["train", "valid_a"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--base-model-revision", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.base_model_path:
        cfg.setdefault("backbone", {})["base_model_path"] = str(args.base_model_path)
    if args.base_model_revision:
        cfg.setdefault("backbone", {})["revision"] = str(args.base_model_revision)
    result = extract(
        cfg,
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        split_name=args.split_name,
        output=args.output,
        max_samples=args.max_samples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
