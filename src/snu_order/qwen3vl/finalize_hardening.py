from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from snu_order.data.validate_submission import validate_submission
from snu_order.utils.config import load_config
from snu_order.utils.io import write_json

from .calibration_stage_pair import (
    checkpoint_calibration_bindings,
    file_sha256,
    load_calibration,
)


FINAL_ROW_COUNT = 819


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _package_versions() -> dict[str, str]:
    result = {"python": os.sys.version.split()[0], "torch": torch.__version__}
    for package in ("transformers", "peft", "bitsandbytes", "Pillow", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _write_checksums(output: Path) -> None:
    checksum_path = output / "checksums.sha256"
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != checksum_path.name
    )
    lines = [f"{file_sha256(path)}  {path.name}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_hardening(
    *,
    output_dir: str | Path,
    submission_path: str | Path,
    sample_submission_path: str | Path,
    decision_path: str | Path,
    selected_candidate: str,
    checkpoint_path: str | Path,
    config_path: str | Path,
    calibration_path: str | Path,
    inference_profile_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    submission = Path(submission_path)
    reference = Path(sample_submission_path)
    checkpoint = Path(checkpoint_path)
    config_source = Path(config_path)
    calibration_source = Path(calibration_path)
    profile_source = Path(inference_profile_path)
    required = [
        submission,
        reference,
        Path(decision_path),
        checkpoint / "checkpoint_manifest.json",
        checkpoint / "prompt_fingerprint.json",
        config_source,
        calibration_source,
        profile_source,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final hardening inputs are missing: {missing}")

    if submission.resolve().parent != output.resolve():
        raise RuntimeError("Final submission must be written directly into the final output directory")
    if submission.read_bytes().splitlines()[0] != b"Id,Answer":
        raise RuntimeError("Final submission byte header is not exactly Id,Answer")
    validator = validate_submission(submission, reference)
    if validator["row_count"] != FINAL_ROW_COUNT:
        raise RuntimeError(
            f"Final submission must contain {FINAL_ROW_COUNT} data rows, got {validator['row_count']}"
        )

    cfg = load_config(config_source)
    expected_bindings = checkpoint_calibration_bindings(checkpoint, cfg)
    calibration = load_calibration(calibration_source, expected_bindings=expected_bindings)
    prompt_fingerprint = _read_json(checkpoint / "prompt_fingerprint.json")
    decision = _read_json(decision_path)
    if decision.get("selected_candidate") != selected_candidate:
        raise RuntimeError(
            "Selected candidate differs between the final command and comparison decision: "
            f"{selected_candidate!r} != {decision.get('selected_candidate')!r}"
        )

    generated = [
        "validator_report.json",
        "selected_model.json",
        "selected_calibration.json",
        "inference_environment.json",
        "inference_memory.json",
        "submission_manifest.json",
        "final_report.md",
        "checksums.sha256",
    ]
    collisions = [str(output / name) for name in generated if (output / name).exists()]
    if collisions:
        raise RuntimeError(f"Refusing to overwrite final hardening artifacts: {collisions}")

    write_json(output / "validator_report.json", validator)
    selected_model = {
        "candidate": selected_candidate,
        "checkpoint": str(checkpoint.resolve()),
        "config": str(config_source.resolve()),
        "checkpoint_manifest_sha256": file_sha256(checkpoint / "checkpoint_manifest.json"),
        "prompt_fingerprint_sha256": file_sha256(checkpoint / "prompt_fingerprint.json"),
        "prompt_anchor_text": prompt_fingerprint.get("anchor_text"),
        "prompt_anchor_token_ids": prompt_fingerprint.get("anchor_token_ids"),
        "pooling_mode": prompt_fingerprint.get("pooling_mode"),
        "frame_chunk_size": 4,
        "selection_decision": str(Path(decision_path).resolve()),
        "selection_decision_sha256": file_sha256(decision_path),
    }
    write_json(output / "selected_model.json", selected_model)
    shutil.copyfile(calibration_source, output / "selected_calibration.json")

    environment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "packages": _package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_total_memory": (
            int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0
        ),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    if environment["cuda_visible_devices"] != "0" or environment["cuda_device_count"] != 1:
        raise RuntimeError(f"Final inference environment is not single-GPU 0: {environment}")
    if environment["world_size"] != 1:
        raise RuntimeError(f"Final inference WORLD_SIZE must be 1: {environment['world_size']}")
    write_json(output / "inference_environment.json", environment)

    profile = _read_json(profile_source)
    if int(profile.get("frame_chunk_size", -1)) != 4:
        raise RuntimeError(f"Final inference did not use the certified unchunked path: {profile}")
    write_json(output / "inference_memory.json", profile)

    submission_hash = file_sha256(submission)
    manifest = {
        "schema": ["Id", "Answer"],
        "row_count": int(validator["row_count"]),
        "id_order_sha256": validator["id_order_sha256"],
        "sample_submission_sha256": file_sha256(reference),
        "submission_sha256": submission_hash,
        "checkpoint_manifest_sha256": selected_model["checkpoint_manifest_sha256"],
        "calibration_sha256": file_sha256(calibration_source),
        "calibration_bindings": expected_bindings,
        "scorer_code_sha256": expected_bindings["scorer_code_sha256"],
        "git_commit": environment["git_commit"],
        "generated_at": environment["generated_at"],
        "validator_version": validator["validator_version"],
        "selected_candidate": selected_candidate,
        "calibration_parameters": calibration.as_dict(),
        "frame_chunk_size": 4,
    }
    write_json(output / "submission_manifest.json", manifest)

    line_count = len(submission.read_bytes().splitlines())
    report = [
        "# Final hardening report",
        "",
        f"- Champion: {selected_candidate}",
        f"- Checkpoint: {checkpoint.resolve()}",
        f"- Calibration: {calibration_source.resolve()}",
        "- Inference: single GPU, unchunked frame_chunk_size=4",
        "- Identity prior: disabled",
        "- valid-B tuning: not used",
        "",
        "## Submission checks",
        "",
        "```text",
        "head -n 1 submission.csv",
        "Id,Answer",
        "wc -l submission.csv",
        f"{line_count} submission.csv",
        "sha256sum submission.csv",
        f"{submission_hash}  submission.csv",
        "```",
        "",
        f"CSV parser data-row count: {validator['row_count']}",
        f"Strict validator: {validator['status']}",
    ]
    (output / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_checksums(output)
    return {"selected_model": selected_model, "manifest": manifest, "validator": validator}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--selected-candidate", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--inference-profile", required=True)
    args = parser.parse_args()
    result = finalize_hardening(
        output_dir=args.output_dir,
        submission_path=args.submission,
        sample_submission_path=args.sample_submission,
        decision_path=args.decision,
        selected_candidate=args.selected_candidate,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        calibration_path=args.calibration,
        inference_profile_path=args.inference_profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
