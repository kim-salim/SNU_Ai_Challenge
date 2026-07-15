from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from snu_order.data.submission_schema import SUBMISSION_COLUMNS, SUBMISSION_SCHEMA_VERSION
from snu_order.order.answer_convert import answer_to_perm, perm_to_answer, validate_answer


def find_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        found = lower_to_original.get(candidate.lower())
        if found is not None:
            return found
    return None


def parse_answer(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        return validate_answer(value)
    text = str(value).strip()
    if not text:
        raise ValueError("answer is empty")
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return validate_answer(parsed)
    except (SyntaxError, ValueError):
        pass
    normalized = text.replace("[", " ").replace("]", " ").replace(",", " ")
    parts = [part for part in normalized.split() if part]
    return validate_answer([int(part) for part in parts])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_strict_submission_csv(path: str | Path, *, role: str) -> tuple[list[str], list[list[str]]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{role} CSV not found: {csv_path}")
    raw = csv_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{role} CSV must not contain a UTF-8 BOM: {csv_path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{role} CSV must be UTF-8: {csv_path}") from exc
    reader = csv.reader(text.splitlines())
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError(f"{role} CSV is empty: {csv_path}") from exc
    expected = list(SUBMISSION_COLUMNS)
    if header != expected:
        raise ValueError(f"{role} CSV header must be exactly {expected}, got {header}")
    rows: list[list[str]] = []
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(expected):
            raise ValueError(
                f"{role} CSV row {row_number} must contain exactly {len(expected)} fields, got {len(row)}"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"{role} CSV has no data rows: {csv_path}")
    return header, rows


def read_reference_ids(path: str | Path) -> list[str]:
    _, rows = _read_strict_submission_csv(path, role="Reference")
    ids = [row[0] for row in rows]
    if any(not sample_id for sample_id in ids):
        raise ValueError("Reference Id values must not be empty or null")
    if len(ids) != len(set(ids)):
        raise ValueError("Reference Id values must be unique")
    return ids


def validate_submission(file: str | Path, reference: str | Path | None = None) -> dict[str, Any]:
    sub_path = Path(file)
    header, rows = _read_strict_submission_csv(sub_path, role="Submission")
    ids = [row[0] for row in rows]
    if any(not sample_id for sample_id in ids):
        raise ValueError("Submission Id values must not be empty or null")
    if len(ids) != len(set(ids)):
        raise ValueError("Submission Id values must be unique")
    parsed_answers: list[list[int]] = []
    for row_idx, row in enumerate(rows, start=2):
        value = row[1]
        if not value:
            raise ValueError(f"Answer is empty at CSV row {row_idx}")
        if value != value.strip():
            raise ValueError(f"Answer has leading or trailing whitespace at CSV row {row_idx}: {value!r}")
        try:
            parsed = parse_answer(value)
        except ValueError as exc:
            raise ValueError(f"Invalid Answer at CSV row {row_idx}: {value!r}") from exc
        canonical = json.dumps(parsed)
        if value != canonical:
            raise ValueError(
                f"Answer must use canonical serialization at CSV row {row_idx}: "
                f"expected {canonical!r}, got {value!r}"
            )
        if perm_to_answer(answer_to_perm(parsed)) != parsed:
            raise ValueError(f"Answer does not round-trip through the official mapping at CSV row {row_idx}")
        parsed_answers.append(parsed)

    reference_path: Path | None = None
    if reference is not None:
        reference_path = Path(reference)
        reference_ids = read_reference_ids(reference)
        if len(rows) != len(reference_ids):
            raise ValueError(
                f"Submission row count {len(rows)} does not match reference {len(reference_ids)}"
            )
        if ids != reference_ids:
            mismatch = next(
                (idx for idx, (actual, expected) in enumerate(zip(ids, reference_ids, strict=True)) if actual != expected),
                None,
            )
            detail = "" if mismatch is None else (
                f"; first mismatch at row {mismatch}: {ids[mismatch]!r} != {reference_ids[mismatch]!r}"
            )
            raise ValueError(f"Submission Id values/order do not exactly match the reference{detail}")
    return {
        "status": "PASS",
        "validator_version": SUBMISSION_SCHEMA_VERSION,
        "schema": header,
        "row_count": len(rows),
        "id_order_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest(),
        "submission_sha256": _sha256(sub_path),
        "reference_sha256": None if reference_path is None else _sha256(reference_path),
        "answer_count": len(parsed_answers),
        "bom": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    import json

    print(json.dumps(validate_submission(args.file, args.reference), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
