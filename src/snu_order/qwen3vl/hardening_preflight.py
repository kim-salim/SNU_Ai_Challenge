from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_json

from .calibration_stage_pair import file_sha256


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _directory_hash(root: Path) -> tuple[str, list[dict[str, Any]]]:
    import hashlib

    files = [
        {
            "relative_path": str(path.relative_to(root)),
            "byte_size": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), files


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def run_preflight(
    repo_root: str | Path,
    output_dir: str | Path,
    artifacts: list[str],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly '0', got {visible!r}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Exactly one CUDA device must be visible, got {torch.cuda.device_count()}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(f"Distributed world size must be 1, got {world_size}")

    git_state = {
        "repo_root": str(root),
        "head": _run_git(root, "rev-parse", "HEAD"),
        "branch": _run_git(root, "branch", "--show-current"),
        "status_short": _run_git(root, "status", "--short").splitlines(),
    }
    properties = torch.cuda.get_device_properties(0)
    environment = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "transformers": _version("transformers"),
        "peft": _version("peft"),
        "bitsandbytes": _version("bitsandbytes"),
        "pillow": _version("Pillow"),
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": visible,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_total_memory": int(properties.total_memory),
        "world_size": world_size,
    }
    sources: dict[str, Any] = {}
    for value in artifacts:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise RuntimeError(f"Artifact must use name=path syntax: {value!r}")
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Preflight source artifact does not exist: {path}")
        if path.is_dir():
            digest, files = _directory_hash(path)
            sources[name] = {
                "path": str(path),
                "type": "directory",
                "sha256": digest,
                "file_count": len(files),
                "files": files,
            }
        else:
            sources[name] = {
                "path": str(path),
                "type": "file",
                "byte_size": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
    write_json(output / "git_state.json", git_state)
    write_json(output / "environment.json", environment)
    write_json(output / "source_artifacts.json", sources)
    return {"git_state": git_state, "environment": environment, "source_artifact_count": len(sources)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            run_preflight(args.repo_root, args.output_dir, args.artifact),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
