from __future__ import annotations

import argparse
import ast
from pathlib import Path
from collections.abc import Sequence

from snu_order.order.answer_convert import validate_answer
from snu_order.utils.io import read_csv_rows


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


def read_reference_ids(path: str | Path) -> list[str]:
    rows = read_csv_rows(path)
    if not rows:
        return []
    id_col = find_column(rows[0].keys(), ["Id", "ID", "id"])
    if id_col is None:
        raise ValueError(f"Reference CSV must contain an Id column: {path}")
    return [str(row[id_col]) for row in rows]


def validate_submission(file: str | Path, reference: str | Path | None = None) -> None:
    sub_path = Path(file)
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_path}")
    rows = read_csv_rows(sub_path)
    if not rows:
        raise ValueError("Submission CSV has no rows")

    columns = list(rows[0].keys())
    id_col = find_column(columns, ["Id", "ID", "id"])
    answer_col = find_column(columns, ["answer", "Answer"])
    if id_col is None:
        raise ValueError("Submission CSV must contain an Id column")
    if answer_col is None:
        raise ValueError("Submission CSV must contain an answer column")

    ids = [str(row[id_col]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Submission Id values must be unique")
    for row_idx, row in enumerate(rows):
        try:
            parse_answer(row[answer_col])
        except ValueError as exc:
            raise ValueError(f"Invalid answer at row {row_idx}: {row[answer_col]!r}") from exc

    if reference is not None:
        reference_ids = read_reference_ids(reference)
        if len(rows) != len(reference_ids):
            raise ValueError(
                f"Submission row count {len(rows)} does not match reference {len(reference_ids)}"
            )
        missing = set(reference_ids) - set(ids)
        extra = set(ids) - set(reference_ids)
        if missing:
            raise ValueError(f"Submission is missing {len(missing)} reference Id values")
        if extra:
            raise ValueError(f"Submission has {len(extra)} unknown Id values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    validate_submission(args.file, args.reference)
    print(f"valid submission: {args.file}")


if __name__ == "__main__":
    main()

