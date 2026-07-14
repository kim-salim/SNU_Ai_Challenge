from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from snu_order.utils.io import write_json

from .calibration_stage_pair import save_raw_stage_pair_logits
from .permutations import perm_index_to_answer
from .stage_pair_scorer import structured_permutation_logits


def import_legacy_raw_scores(
    source: str | Path,
    destination: str | Path,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    source_path = Path(source)
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line]
    if expected_count is not None and len(rows) != int(expected_count):
        raise RuntimeError(f"Legacy raw score row count is {len(rows)}, expected {expected_count}")
    ids = [str(row["Id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Legacy raw scores contain duplicate sample IDs")
    stage = torch.tensor([row["stage_logits"] for row in rows], dtype=torch.float32)
    pair = torch.tensor([row["pair_logits"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([int(row["true_perm_idx"]) for row in rows], dtype=torch.long)
    answers = torch.tensor([perm_index_to_answer(int(value)) for value in targets.tolist()], dtype=torch.long)
    recorded = torch.tensor([row["logits"] for row in rows], dtype=torch.float32)
    reconstructed = structured_permutation_logits(stage, pair, stage_weight=1.0, pair_weight=0.3).float()
    max_abs_diff = float((recorded - reconstructed).abs().max().item())
    prediction_mismatches = int(recorded.argmax(dim=1).ne(reconstructed.argmax(dim=1)).sum().item())
    recorded_prediction_mismatches = sum(
        int(int(row["pred_perm_idx"]) != int(recorded[index].argmax().item()))
        for index, row in enumerate(rows)
    )
    if max_abs_diff > 1e-5 or prediction_mismatches or recorded_prediction_mismatches:
        raise RuntimeError(
            "Legacy raw scores do not reconstruct under the fixed scorer: "
            f"max_abs_diff={max_abs_diff}, prediction_mismatches={prediction_mismatches}, "
            f"recorded_prediction_mismatches={recorded_prediction_mismatches}"
        )
    correct_count = int(reconstructed.argmax(dim=1).eq(targets).sum().item())
    save_raw_stage_pair_logits(
        destination,
        ids=ids,
        stage_logits=stage,
        pair_logits=pair,
        target_perm_idx=targets,
        answer=answers,
        metadata={
            "source_format": "legacy_raw_scores_jsonl",
            "source_path": str(source_path.resolve()),
            "reconstruction_max_abs_diff": max_abs_diff,
        },
    )
    report = {
        "status": "PASS",
        "sample_count": len(rows),
        "correct_count": correct_count,
        "max_abs_diff": max_abs_diff,
        "prediction_mismatch_count": prediction_mismatches,
        "recorded_prediction_mismatch_count": recorded_prediction_mismatches,
        "destination": str(Path(destination).resolve()),
    }
    write_json(Path(destination).with_suffix(".import_report.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    report = import_legacy_raw_scores(
        args.source,
        args.destination,
        expected_count=args.expected_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
