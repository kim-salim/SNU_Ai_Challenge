from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from snu_order.qwen3vl.modeling_stage_pair import build_stage_pair_head_from_config
from snu_order.utils.config import load_config
from snu_order.utils.io import write_json


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_exact(module: torch.nn.Module, state: dict[str, Any], *, prefix: str) -> list[str]:
    expected = set(module.state_dict())
    observed = set(state)
    if expected != observed:
        raise RuntimeError(
            f"{prefix} state mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    module.load_state_dict(state, strict=True)
    return [f"{prefix}.{key}" for key in sorted(expected)]


def migrate_champion_heads(
    source_checkpoint: str | Path,
    config: str | Path,
    output: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    source_root = Path(source_checkpoint)
    heads_path = source_root / "heads.pt"
    if not heads_path.is_file():
        raise FileNotFoundError(f"Champion heads are missing: {heads_path}")
    payload = torch.load(heads_path, map_location="cpu", weights_only=False)
    if int(payload.get("hidden_size", -1)) != 4096:
        raise RuntimeError("Champion source heads must have hidden_size=4096")
    cfg = load_config(config)
    architecture_id = str(cfg.get("architecture", {}).get("id", ""))
    from snu_order.qwen3vl.qwen35_27b_port import SUPPORTED_ARCHITECTURE_IDS

    if architecture_id not in SUPPORTED_ARCHITECTURE_IDS:
        raise RuntimeError(f"Unsupported 27B migration architecture: {architecture_id!r}")
    torch.manual_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    model = build_stage_pair_head_from_config(cfg, hidden_size=5120, backbone=None)
    loaded: list[str] = []
    skipped: list[str] = []
    fresh = [f"frame_projector.{key}" for key in sorted(model.frame_projector.state_dict())]
    rejected: list[str] = []

    source_set = payload.get("set_encoder")
    if not isinstance(source_set, dict):
        raise RuntimeError("Champion heads have no set_encoder state")
    transferable = {
        key: value
        for key, value in source_set.items()
        if key.startswith("encoder.") or key.startswith("output_norm.")
    }
    skipped.extend(
        f"set_encoder.{key}"
        for key in sorted(set(source_set) - set(transferable))
    )
    loaded.extend(_load_exact(model.set_encoder, transferable, prefix="set_encoder"))
    loaded.extend(_load_exact(model.stage_head, payload["stage_head"], prefix="stage_head"))
    if model.pair_head is None or payload.get("pair_head") is None:
        raise RuntimeError("Champion/port pair head is missing")
    loaded.extend(_load_exact(model.pair_head, payload["pair_head"], prefix="pair_head"))

    adapter_manifest = source_root / "lora_target_manifest.json"
    if adapter_manifest.is_file():
        old_targets = json.loads(adapter_manifest.read_text(encoding="utf-8"))
        rejected.extend(f"9b_lora:{entry['module_name']}" for entry in old_targets)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture_id,
            "hidden_size": 5120,
            "model_dim": int(model.model_dim),
            "pooling_mode": str(model.pooling_mode),
            "frame_projector": model.frame_projector.state_dict(),
            "set_encoder": model.set_encoder.state_dict(),
            "stage_head": model.stage_head.state_dict(),
            "pair_head": model.pair_head.state_dict(),
            "migration_source_sha256": file_sha256(heads_path),
        },
        destination,
    )
    report = {
        "status": "PASS",
        "source_checkpoint": str(source_root.resolve()),
        "source_checkpoint_sha256": file_sha256(heads_path),
        "destination": str(destination.resolve()),
        "destination_schema": {
            "architecture": architecture_id,
            "hidden_size": 5120,
            "model_dim": int(model.model_dim),
        },
        "loaded_keys": loaded,
        "skipped_keys": skipped,
        "fresh_keys": fresh,
        "rejected_keys": rejected,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = migrate_champion_heads(args.source_checkpoint, args.config, args.output, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
