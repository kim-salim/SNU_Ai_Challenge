from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.config import load_config
from snu_order.utils.io import write_json

from .qwen35_27b_port import build_strict_nf4_config, validate_qwen35_27b_architecture


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def audit_local_model(
    config_path: str | Path,
    output: str | Path,
    base_path: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    from transformers import AutoConfig

    cfg = deepcopy(load_config(config_path))
    if base_path:
        cfg["backbone"]["base_model_path"] = str(Path(base_path).resolve())
    if revision:
        cfg["backbone"]["revision"] = revision
    model_path = Path(str(cfg["backbone"]["base_model_path"])).expanduser()
    if not model_path.exists():
        report = {
            "status": "HOLD_27B_MODEL_MISSING",
            "requested_path": str(model_path),
            "official_model_id": "Qwen/Qwen3.5-27B",
            "auto_download_attempted": False,
        }
        write_json(output, report)
        return report
    config = AutoConfig.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=bool(cfg["backbone"].get("trust_remote_code", True)),
    )
    architecture = validate_qwen35_27b_architecture(config)
    quant = build_strict_nf4_config(cfg)
    report = {
        "status": "PASS",
        "model_path": str(model_path.resolve()),
        "model_revision": cfg["backbone"].get("revision"),
        "architecture": architecture,
        "quantization": {
            "load_in_4bit": bool(quant.load_in_4bit),
            "bnb_4bit_quant_type": str(quant.bnb_4bit_quant_type),
            "bnb_4bit_use_double_quant": bool(quant.bnb_4bit_use_double_quant),
            "bnb_4bit_compute_dtype": str(quant.bnb_4bit_compute_dtype),
            "torch_dtype": str(torch.bfloat16),
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "packages": {
            name: _version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes", "safetensors")
        },
        "offline_environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
                "WANDB_DISABLED",
            )
        },
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-path", default=os.environ.get("QWEN35_27B_BASE_PATH"))
    parser.add_argument("--revision", default=os.environ.get("QWEN35_27B_REVISION"))
    args = parser.parse_args()
    result = audit_local_model(args.config, args.output, args.base_path, args.revision)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
