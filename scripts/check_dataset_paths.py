from __future__ import annotations

import argparse
import csv
from pathlib import Path


TEXT_COLUMNS = (
    "Sentence", "sentence", "text", "Text",
    "caption", "Caption", "description", "Description",
)


def find_column(columns: list[str], candidates: list[str] | tuple[str, ...]) -> str | None:
    lower_to_real = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_real:
            return lower_to_real[candidate.lower()]
    return None


def frame_candidates(index: int) -> tuple[str, ...]:
    return (
        f"frame{index}", f"frame_{index}", f"Frame{index}", f"Frame_{index}",
        f"image{index}", f"image_{index}", f"Image{index}", f"Image_{index}",
        f"path{index}", f"path_{index}",
        f"frame{index}_path", f"frame_{index}_path",
        f"image{index}_path", f"image_{index}_path",
    )


def resolve_path(csv_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    repo_relative = Path(value)
    if repo_relative.exists():
        return repo_relative

    return csv_path.parent / path


def check_csv(csv_path: Path, max_rows: int) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames is None:
        raise RuntimeError(f"Could not read CSV header: {csv_path}")
    if not rows:
        raise RuntimeError(f"No rows found: {csv_path}")

    print(f"\n== {csv_path} ==")
    print("columns:", fieldnames)

    id_col = find_column(fieldnames, ("Id", "ID", "id"))
    text_col = find_column(fieldnames, TEXT_COLUMNS)
    frame_cols = [find_column(fieldnames, frame_candidates(i)) for i in range(4)]

    print("id_col    :", id_col)
    print("text_col  :", text_col)
    print("frame_cols:", frame_cols)

    if id_col is None:
        raise RuntimeError("Id column not found")
    if text_col is None:
        raise RuntimeError("Text/Sentence column not found")
    if any(col is None for col in frame_cols):
        raise RuntimeError("One or more frame columns not found")

    checked = min(max_rows, len(rows))
    missing = 0

    for row_idx, row in enumerate(rows[:checked]):
        for col in frame_cols:
            assert col is not None
            path = resolve_path(csv_path, str(row[col]))
            if not path.exists():
                missing += 1
                print(f"[MISSING] row={row_idx} col={col} value={row[col]} -> {path}")

    if missing:
        raise FileNotFoundError(f"Missing image files: {missing}")

    print(f"[OK] checked {checked} rows, all frame paths exist.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--test", default="data/raw/test.csv")
    parser.add_argument("--max-rows", type=int, default=20)
    args = parser.parse_args()

    check_csv(Path(args.train), args.max_rows)
    check_csv(Path(args.test), args.max_rows)


if __name__ == "__main__":
    main()
