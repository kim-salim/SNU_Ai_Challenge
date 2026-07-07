from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--train-out", default="data/splits/train_v1.csv")
    parser.add_argument("--valid-out", default="data/splits/valid_v1.csv")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    train_out = Path(args.train_out)
    valid_out = Path(args.valid_out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    random.seed(args.seed)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames is None:
        raise RuntimeError(f"Could not read CSV header from {input_path}")
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    random.shuffle(rows)

    n_valid = max(1, int(len(rows) * args.valid_ratio))
    valid_rows = rows[:n_valid]
    train_rows = rows[n_valid:]

    train_out.parent.mkdir(parents=True, exist_ok=True)
    valid_out.parent.mkdir(parents=True, exist_ok=True)

    for path, part in [(train_out, train_rows), (valid_out, valid_rows)]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(part)

    print(f"[OK] input rows : {len(rows)}")
    print(f"[OK] train rows : {len(train_rows)} -> {train_out}")
    print(f"[OK] valid rows : {len(valid_rows)} -> {valid_out}")


if __name__ == "__main__":
    main()
