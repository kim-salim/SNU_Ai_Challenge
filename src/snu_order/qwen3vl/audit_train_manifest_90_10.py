from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from snu_order.utils.io import write_json


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Id" not in rows[0]:
        raise RuntimeError(f"Split has no Id column: {path}")
    return [str(row["Id"]) for row in rows]


def audit(manifest_path: str | Path, output: str | Path) -> dict:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    train = Path(str(manifest["train_csv"]))
    valid = Path(str(manifest["valid_csv"]))
    train_ids = _ids(train)
    valid_ids = _ids(valid)
    failures = []
    if int(manifest["source_count"]) != 9535:
        failures.append("source_count")
    if len(train_ids) != int(manifest["train_count"]) or len(valid_ids) != int(manifest["valid_count"]):
        failures.append("row_count")
    if set(train_ids) & set(valid_ids):
        failures.append("id_overlap")
    if _sha(train) != manifest["train_csv_sha256"] or _sha(valid) != manifest["valid_csv_sha256"]:
        failures.append("csv_sha256")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "split_id": manifest.get("split_id"),
        "source_count": int(manifest["source_count"]),
        "train_count": len(train_ids),
        "valid_count": len(valid_ids),
        "train_ratio": len(train_ids) / (len(train_ids) + len(valid_ids)),
        "valid_ratio": len(valid_ids) / (len(train_ids) + len(valid_ids)),
        "train_csv": str(train),
        "valid_csv": str(valid),
        "train_sha256": _sha(train),
        "valid_sha256": _sha(valid),
        "manifest_sha256": _sha(path),
        "id_overlap_count": len(set(train_ids) & set(valid_ids)),
        "failures": failures,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

