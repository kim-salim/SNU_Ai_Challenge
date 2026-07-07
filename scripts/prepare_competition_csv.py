from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames is None:
        raise RuntimeError(f"Could not read CSV header: {path}")

    return rows, fieldnames


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(
    input_csv: Path,
    output_csv: Path,
    image_split_dir: str,
    has_answer: bool,
) -> None:
    rows, _ = read_rows(input_csv)

    prepared_rows: list[dict[str, str]] = []

    for row in rows:
        sample_id = row["Id"]

        new_row: dict[str, str] = {
            "Id": sample_id,
            "frame0": f"data/raw/{image_split_dir}/{sample_id}/{row['Input_1']}",
            "frame1": f"data/raw/{image_split_dir}/{sample_id}/{row['Input_2']}",
            "frame2": f"data/raw/{image_split_dir}/{sample_id}/{row['Input_3']}",
            "frame3": f"data/raw/{image_split_dir}/{sample_id}/{row['Input_4']}",
            "Sentence": row["Sentence"],
        }

        if has_answer:
            new_row["Answer"] = row["Answer"]
            if "No_ordering" in row:
                new_row["No_ordering"] = row["No_ordering"]

        prepared_rows.append(new_row)

    fieldnames = ["Id", "frame0", "frame1", "frame2", "frame3", "Sentence"]
    if has_answer:
        fieldnames += ["Answer"]
        if prepared_rows and "No_ordering" in prepared_rows[0]:
            fieldnames += ["No_ordering"]

    write_rows(output_csv, prepared_rows, fieldnames)

    print(f"[OK] {input_csv} -> {output_csv}")
    print(f"[OK] rows: {len(prepared_rows)}")


def backup_once(path: Path) -> Path:
    backup_path = path.with_suffix(path.suffix + ".original")

    if backup_path.exists():
        print(f"[SKIP] backup already exists: {backup_path}")
        return backup_path

    shutil.copy2(path, backup_path)
    print(f"[OK] backup: {path} -> {backup_path}")
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    train_csv = raw_dir / "train.csv"
    test_csv = raw_dir / "test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not test_csv.exists():
        raise FileNotFoundError(test_csv)

    train_out = raw_dir / "train_prepared.csv"
    test_out = raw_dir / "test_prepared.csv"

    prepare_split(
        input_csv=train_csv,
        output_csv=train_out,
        image_split_dir="train",
        has_answer=True,
    )
    prepare_split(
        input_csv=test_csv,
        output_csv=test_out,
        image_split_dir="test",
        has_answer=False,
    )

    if args.in_place:
        backup_once(train_csv)
        backup_once(test_csv)

        shutil.copy2(train_out, train_csv)
        shutil.copy2(test_out, test_csv)

        print("[OK] replaced data/raw/train.csv with prepared CSV")
        print("[OK] replaced data/raw/test.csv with prepared CSV")
        print("[OK] original files are saved as:")
        print(f"     {train_csv}.original")
        print(f"     {test_csv}.original")


if __name__ == "__main__":
    main()
