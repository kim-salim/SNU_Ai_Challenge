from __future__ import annotations

import os
from pathlib import Path


def set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def require_local_model_dir(model_dir: str | Path) -> Path:
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Local pretrained model directory not found: {path}. "
            "Download google/siglip2-base-patch16-224 into this directory before running."
        )
    if not path.is_dir():
        raise FileNotFoundError(f"Local pretrained model path is not a directory: {path}")
    if not any(path.iterdir()):
        raise FileNotFoundError(
            f"Local pretrained model directory is empty: {path}. "
            "It must contain the Hugging Face config, processor, tokenizer, and weights files."
        )
    return path

