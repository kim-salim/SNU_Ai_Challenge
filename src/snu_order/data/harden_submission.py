from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from snu_order.data.submission_schema import SUBMISSION_COLUMNS, SUBMISSION_SCHEMA_VERSION
from snu_order.data.validate_submission import validate_submission


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing hardening artifact: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repair_submission_header_only(
    source: str | Path,
    sample_submission: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_manifest: str | Path,
    calibration_artifact: str | Path,
    scorer_code: str | Path,
    repo_root: str | Path,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    sample_path = Path(sample_submission).resolve()
    checkpoint_path = Path(checkpoint_manifest).resolve()
    calibration_path = Path(calibration_artifact).resolve()
    scorer_path = Path(scorer_code).resolve()
    root = Path(repo_root).resolve()
    for required in (source_path, sample_path, checkpoint_path, calibration_path, scorer_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required source artifact does not exist: {required}")

    original = source_path.read_bytes()
    if original.startswith(b"\xef\xbb\xbf"):
        raise RuntimeError("Source submission unexpectedly contains a UTF-8 BOM")
    lines = original.splitlines(keepends=True)
    if not lines:
        raise RuntimeError("Source submission is empty")
    header = lines[0]
    if header == b"Id,answer\r\n":
        corrected_header = b"Id,Answer\r\n"
    elif header == b"Id,answer\n":
        corrected_header = b"Id,Answer\n"
    else:
        raise RuntimeError(f"Header-only repair requires exact source header Id,answer, got {header!r}")
    corrected = corrected_header + b"".join(lines[1:])
    if b"".join(lines[1:]) != corrected[len(corrected_header) :]:
        raise AssertionError("Submission data rows changed during header-only repair")

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    preserved_path = out / "submission.pre_schema_fix.csv"
    corrected_path = out / "submission.csv"
    _atomic_write_bytes(preserved_path, original)
    _atomic_write_bytes(corrected_path, corrected)
    if preserved_path.read_bytes() != original:
        raise AssertionError("Preserved submission bytes differ from the source")
    if corrected_path.read_bytes().splitlines(keepends=True)[1:] != lines[1:]:
        raise AssertionError("Corrected submission data rows differ from the source")

    report = validate_submission(corrected_path, sample_path)
    if expected_rows is not None and int(report["row_count"]) != int(expected_rows):
        raise RuntimeError(
            f"Corrected submission row count is {report['row_count']}, expected {int(expected_rows)}"
        )
    report_path = out / "validator_report.json"
    _atomic_write_json(report_path, report)
    manifest = {
        "schema": list(SUBMISSION_COLUMNS),
        "row_count": int(report["row_count"]),
        "id_order_sha256": report["id_order_sha256"],
        "sample_submission_sha256": sha256_file(sample_path),
        "original_submission_sha256": sha256_file(preserved_path),
        "corrected_submission_sha256": sha256_file(corrected_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
        "calibration_artifact_sha256": sha256_file(calibration_path),
        "scorer_code_sha256": sha256_file(scorer_path),
        "git_commit": _git_commit(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validator_version": SUBMISSION_SCHEMA_VERSION,
        "data_rows_byte_identical": True,
        "source_submission": str(source_path),
    }
    manifest_path = out / "submission_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    checksum_paths = [preserved_path, corrected_path, manifest_path, report_path]
    checksums = "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths)
    _atomic_write_bytes(out / "checksums.sha256", checksums.encode("ascii"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--calibration-artifact", required=True)
    parser.add_argument("--scorer-code", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-rows", type=int, default=None)
    args = parser.parse_args()
    result = repair_submission_header_only(
        args.source,
        args.sample_submission,
        args.output_dir,
        checkpoint_manifest=args.checkpoint_manifest,
        calibration_artifact=args.calibration_artifact,
        scorer_code=args.scorer_code,
        repo_root=args.repo_root,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
