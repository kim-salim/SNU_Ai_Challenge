from __future__ import annotations

from pathlib import Path
from typing import Any

from snu_order.data.validate_submission import find_column
from snu_order.utils.io import read_csv_rows


def _read_ids(path: str | Path) -> tuple[set[str], set[str] | None]:
    rows = read_csv_rows(path)
    if not rows:
        return set(), None
    columns = list(rows[0].keys())
    id_col = find_column(columns, ["Id", "ID", "id", "sample_id", "sampleId"])
    if id_col is None:
        raise ValueError(f"Could not detect id column in {path}")
    group_col = find_column(columns, ["source_video_id", "video_id", "group_id", "source_id"])
    ids = {str(row[id_col]) for row in rows}
    groups = {str(row[group_col]) for row in rows} if group_col is not None else None
    return ids, groups


def check_split_overlap(
    train_ab: str | Path,
    valid_a: str | Path,
    valid_b: str | Path,
) -> dict[str, Any]:
    ids_train, groups_train = _read_ids(train_ab)
    ids_a, groups_a = _read_ids(valid_a)
    ids_b, groups_b = _read_ids(valid_b)
    id_overlaps = {
        "train_ab_valid_a": len(ids_train & ids_a),
        "train_ab_valid_b": len(ids_train & ids_b),
        "valid_a_valid_b": len(ids_a & ids_b),
    }
    group_overlaps = None
    if groups_train is not None and groups_a is not None and groups_b is not None:
        group_overlaps = {
            "train_ab_valid_a": len(groups_train & groups_a),
            "train_ab_valid_b": len(groups_train & groups_b),
            "valid_a_valid_b": len(groups_a & groups_b),
        }
    result: dict[str, Any] = {
        "rows": {
            "train_ab": len(ids_train),
            "valid_a": len(ids_a),
            "valid_b": len(ids_b),
        },
        "id_overlaps": id_overlaps,
        "group_overlaps": group_overlaps,
    }
    if any(value for value in id_overlaps.values()):
        raise ValueError(f"Split ID overlap detected: {id_overlaps}")
    if group_overlaps is not None and any(value for value in group_overlaps.values()):
        raise ValueError(f"Split group overlap detected: {group_overlaps}")
    return result
