from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_json

from .permutations import PERMS


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_sha256(path: str | Path) -> str:
    root = Path(path)
    h = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(file.relative_to(root)).encode("utf-8"))
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def save_lora24_checkpoint(
    path: str | Path,
    model: Any,
    processor: Any | None,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    *,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
    minimal: bool = False,
) -> None:
    model = unwrap_model(model)
    ckpt_dir = Path(path)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier": model.classifier.state_dict(),
            "hidden_size": int(model.hidden_size),
            "num_classes": int(model.num_classes),
            "metrics": metrics,
        },
        ckpt_dir / "classifier.pt",
    )
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "save_pretrained"):
        try:
            backbone.save_pretrained(ckpt_dir / "adapter")
        except Exception:
            pass
    if processor is not None and hasattr(processor, "save_pretrained"):
        try:
            processor.save_pretrained(ckpt_dir / "processor")
        except Exception:
            pass
    write_json(ckpt_dir / "config.json", cfg)
    write_json(ckpt_dir / "metrics.json", metrics)
    write_json(ckpt_dir / "permutations.json", {"perms": [list(perm) for perm in PERMS]})
    if extra:
        write_json(ckpt_dir / "extra.json", extra)
    if not minimal:
        state: dict[str, Any] = {}
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if state:
            torch.save(state, ckpt_dir / "training_state.pt")


def load_classifier_weights(path: str | Path, model: Any, *, strict: bool = True) -> dict[str, Any]:
    model = unwrap_model(model)
    ckpt_path = Path(path) / "classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"classifier checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.classifier.load_state_dict(checkpoint["classifier"], strict=strict)
    return checkpoint


def load_lora24_checkpoint(
    path: str | Path,
    model: Any,
    *,
    strict: bool = True,
    is_trainable: bool = False,
) -> tuple[Any, dict[str, Any]]:
    model = unwrap_model(model)
    ckpt_dir = Path(path)
    adapter_dir = ckpt_dir / "adapter"
    if adapter_dir.exists():
        try:
            if hasattr(model.backbone, "load_adapter"):
                adapter_name = "loaded_lora24"
                model.backbone.load_adapter(str(adapter_dir), adapter_name=adapter_name, is_trainable=is_trainable)
                if hasattr(model.backbone, "set_adapter"):
                    model.backbone.set_adapter(adapter_name)
            else:
                from peft import PeftModel

                model.backbone = PeftModel.from_pretrained(model.backbone, str(adapter_dir), is_trainable=is_trainable)
        except Exception as exc:
            raise RuntimeError(f"Failed to load LoRA adapter from {adapter_dir}") from exc
    checkpoint = load_classifier_weights(ckpt_dir, model, strict=strict)
    return model, checkpoint


def load_checkpoint_metrics(path: str | Path) -> dict[str, Any]:
    metrics_path = Path(path) / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))
