from __future__ import annotations

import argparse
import json
import platform
from typing import Any

import torch

from snu_order.utils.config import get_by_path, load_config


def collect_env(config_path: str, *, smoke_load: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "model_local_dir": str(get_by_path(cfg, "model.local_dir")),
        "local_files_only": bool(get_by_path(cfg, "model.local_files_only", True)),
    }
    for package in ("transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            mod = __import__(package)
            report[package] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            report[f"{package}_error"] = repr(exc)
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(
            str(get_by_path(cfg, "model.local_dir")),
            local_files_only=bool(get_by_path(cfg, "model.local_files_only", True)),
            trust_remote_code=bool(get_by_path(cfg, "model.trust_remote_code", True)),
        )
        report["local_config_check"] = "ok"
    except Exception as exc:
        report["local_config_check"] = "failed"
        report["local_config_error"] = repr(exc)
    if smoke_load:
        try:
            from .modeling_lora24 import build_qwen3vl_lora24_model

            model, _processor = build_qwen3vl_lora24_model(cfg, frozen_probe=True)
            report["smoke_load"] = "ok"
            report["smoke_hidden_size"] = int(model.hidden_size)
            del model
        except Exception as exc:
            report["smoke_load"] = "failed"
            report["smoke_error"] = repr(exc)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp/qwen3vl_8b_lora24.yaml")
    parser.add_argument("--smoke-load", action="store_true")
    args = parser.parse_args()
    report = collect_env(args.config, smoke_load=args.smoke_load)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing = [key for key in ("peft", "bitsandbytes", "accelerate") if key not in report]
    if missing or report.get("local_config_check") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
