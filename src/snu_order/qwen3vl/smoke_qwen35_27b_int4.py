from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image

from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import write_json

from .modeling_stage_pair import build_stage_pair_model_from_config
from .qwen35_27b_port import load_migrated_champion_heads, validate_qwen35_27b_architecture
from .stage_pair_prompt import (
    StagePairPromptSpec,
    build_stage_pair_message,
    build_stage_pair_prompt,
    prepare_stage_pair_multimodal_inputs,
)
from .train_lora24 import _model_device


OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_DISABLED": "true",
}


def _sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()


def run_smoke(
    config_path: str,
    output: str,
    *,
    base_path: str | None,
    revision: str | None,
    migrated_heads: str | None,
) -> dict:
    for key, value in OFFLINE_ENV.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"Offline environment mismatch: {key}={os.environ.get(key)!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("HOLD_INT4_BACKEND_INCOMPATIBLE: CUDA is unavailable")
    cfg = deepcopy(load_config(config_path))
    if base_path:
        cfg["backbone"]["base_model_path"] = str(Path(base_path).resolve())
    if revision:
        cfg["backbone"]["revision"] = revision
    device_index = int(os.environ.get("QWEN35_27B_CUDA_DEVICE", "0"))
    cfg["backbone"]["device_map"] = {"": device_index}
    started = time.perf_counter()
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    model, processor = build_stage_pair_model_from_config(cfg, live_backbone=True)
    architecture = validate_qwen35_27b_architecture(model.backbone.config)
    if migrated_heads:
        migration = load_migrated_champion_heads(migrated_heads, model)
    else:
        migration = {"status": "FRESH_HEADS"}
    device = _model_device(model)
    for name in ("frame_projector", "set_encoder", "stage_head", "pair_head"):
        module = getattr(model, name, None)
        if module is not None:
            module.to(device=device, dtype=torch.bfloat16)
    model.eval()
    load_time = time.perf_counter() - started
    image = Image.new("RGB", (224, 224), color=(32, 96, 160))
    spec = StagePairPromptSpec.from_config(cfg)
    prompt = build_stage_pair_prompt("A person enters, performs an action, and then leaves.", spec)
    conversations = [build_stage_pair_message(prompt, image) for _ in range(4)]
    prepared = prepare_stage_pair_multimodal_inputs(
        processor,
        conversations,
        spec,
        model_revision=str(get_by_path(cfg, "backbone.revision", "")) or None,
    )
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in prepared.inputs.items()}
    anchor_mask = prepared.anchor_mask.to(device) if prepared.anchor_mask is not None else None
    if anchor_mask is None:
        raise RuntimeError("HOLD_PROMPT_ANCHOR_FAILURE: synthetic smoke has no STATE anchor")
    hashes = []
    latencies = []
    pooled_shape = None
    with torch.inference_mode():
        for _ in range(2):
            start = time.perf_counter()
            outputs = model(inputs=inputs, batch_size=1, anchor_mask=anchor_mask, frame_chunk_size=1)
            torch.cuda.synchronize(device_index)
            latencies.append(time.perf_counter() - start)
            pooled_shape = list(outputs["frame_hidden"].shape)
            hashes.append(_sha_tensor(outputs["final_logits"]))
    if pooled_shape != [1, 4, 5120]:
        raise RuntimeError(f"HOLD_HIDDEN_STATE_CONTRACT_FAILURE: pooled shape={pooled_shape}")
    if len(set(hashes)) != 1:
        raise RuntimeError("HOLD_HIDDEN_STATE_CONTRACT_FAILURE: repeated prediction hash mismatch")
    report = {
        "status": "PASS_27B_INT4_SMOKE",
        "architecture": architecture,
        "migration": migration,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "load_time_sec": load_time,
        "sample_time_sec": latencies,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        "state_pooled_shape": pooled_shape,
        "prediction_hashes": hashes,
        "hidden_only_contract": model._last_hidden_only_contract,
        "official_test_accessed": False,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-path", default=os.environ.get("QWEN35_27B_BASE_PATH"))
    parser.add_argument("--revision", default=os.environ.get("QWEN35_27B_REVISION"))
    parser.add_argument("--migrated-heads", default=None)
    args = parser.parse_args()
    result = run_smoke(
        args.config,
        args.output,
        base_path=args.base_path,
        revision=args.revision,
        migrated_heads=args.migrated_heads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
