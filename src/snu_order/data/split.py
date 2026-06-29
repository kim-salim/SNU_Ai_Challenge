from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise SystemExit(
        "Split generation is dataset-specific. Create data/splits/train_v1.csv and "
        f"data/splits/valid_v1.csv for config {args.config}."
    )


if __name__ == "__main__":
    main()

