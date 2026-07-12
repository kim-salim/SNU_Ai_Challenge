from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from snu_order.data.validate_submission import find_column, parse_answer
from snu_order.utils.io import read_csv_rows, write_json

from .metrics24 import compute_metrics_from_logits, write_eval_artifacts
from .permutations import answer_to_perm_index, order_to_perm_index


def _guard_valid_b(path: str | Path, unlock: bool) -> None:
    if "valid_b" in str(path).lower() and not unlock:
        raise PermissionError("valid_b zero-shot score analysis requires --unlock-valid-b-analysis")


def _read_targets(metadata_csv: str | Path) -> dict[str, int]:
    rows = read_csv_rows(metadata_csv)
    if not rows:
        raise ValueError(f"metadata CSV has no rows: {metadata_csv}")
    cols = list(rows[0].keys())
    id_col = find_column(cols, ["Id", "ID", "id", "sample_id", "sampleId"])
    ans_col = find_column(cols, ["Answer", "answer", "label", "true_answer"])
    if id_col is None or ans_col is None:
        raise ValueError(f"Could not detect id/answer columns in {metadata_csv}")
    return {str(row[id_col]): answer_to_perm_index(parse_answer(row[ans_col])) for row in rows}


def _scores_to_lexicographic(entry: dict[str, Any]) -> list[float]:
    scores = entry.get("scores", entry.get("logits"))
    if scores is None:
        raise ValueError("raw score entry missing scores/logits")
    values = [float(v) for v in scores]
    if len(values) != 24:
        raise ValueError(f"raw score entry must contain 24 scores, got {len(values)}")
    orders = entry.get("orders")
    if not orders:
        return values
    out = [0.0] * 24
    for order, score in zip(orders, values, strict=True):
        out[order_to_perm_index(order)] = float(score)
    return out


def analyze(raw_scores: str | Path, metadata_csv: str | Path, output_dir: str | Path) -> dict[str, Any]:
    targets_by_id = _read_targets(metadata_csv)
    ids: list[str] = []
    scores: list[list[float]] = []
    targets: list[int] = []
    with Path(raw_scores).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sample_id = str(entry.get("id", entry.get("Id")))
            if sample_id not in targets_by_id:
                continue
            ids.append(sample_id)
            scores.append(_scores_to_lexicographic(entry))
            targets.append(targets_by_id[sample_id])
    logits = torch.tensor(scores, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    metrics = compute_metrics_from_logits(logits, target_tensor)
    out = Path(output_dir)
    write_eval_artifacts(out, ids, logits, target_tensor, metrics)
    write_json(out / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-scores", default="outputs/predictions/qwen3vl_8b_24candidate/valid_a_full/raw_scores.jsonl")
    parser.add_argument("--metadata-csv", default="data/splits/ab_v1/valid_a_v1.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/qwen3vl_zero_shot_valid_a")
    parser.add_argument("--unlock-valid-b-analysis", action="store_true")
    args = parser.parse_args()
    _guard_valid_b(args.raw_scores, args.unlock_valid_b_analysis)
    _guard_valid_b(args.metadata_csv, args.unlock_valid_b_analysis)
    metrics = analyze(args.raw_scores, args.metadata_csv, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
