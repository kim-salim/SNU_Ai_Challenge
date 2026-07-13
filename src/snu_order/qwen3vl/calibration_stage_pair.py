from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import write_csv_rows, write_json

from .metrics24 import rows_from_logits
from .metrics_stage_pair import compute_stage_pair_metrics
from .permutations import PERMS
from .stage_pair_scorer import structured_permutation_logits


DEFAULT_PAIR_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
DEFAULT_TEMPERATURES = (0.8, 1.0, 1.2, 1.5)


@dataclass(frozen=True)
class CalibrationParameters:
    pair_weight: float
    stage_temperature: float
    pair_temperature: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pair_weight": self.pair_weight,
            "stage_temperature": self.stage_temperature,
            "pair_temperature": self.pair_temperature,
        }


def permutation_table_fingerprint() -> str:
    payload = json.dumps([list(perm) for perm in PERMS], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_raw_stage_pair_logits(
    path: str | Path,
    *,
    ids: list[str],
    stage_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    target_perm_idx: torch.Tensor,
    answer: torch.Tensor,
) -> None:
    payload = {
        "format_version": 1,
        "ids": [str(value) for value in ids],
        "stage_logits": stage_logits.detach().float().cpu(),
        "pair_logits": pair_logits.detach().float().cpu(),
        "target_perm_idx": target_perm_idx.detach().long().cpu(),
        "answer": answer.detach().long().cpu(),
        "permutation_table_fingerprint": permutation_table_fingerprint(),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_raw_stage_pair_logits(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format_version",
        "ids",
        "stage_logits",
        "pair_logits",
        "target_perm_idx",
        "answer",
        "permutation_table_fingerprint",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError("Raw stage/pair artifact does not match the expected schema")
    if payload["permutation_table_fingerprint"] != permutation_table_fingerprint():
        raise RuntimeError("Permutation table fingerprint mismatch in raw stage/pair artifact")
    count = len(payload["ids"])
    for key in ("stage_logits", "pair_logits", "target_perm_idx", "answer"):
        if int(payload[key].shape[0]) != count:
            raise RuntimeError(f"Raw artifact row count mismatch for {key}")
    return payload


def calibrated_structured_logits(
    stage_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    parameters: CalibrationParameters,
) -> torch.Tensor:
    if parameters.stage_temperature <= 0 or parameters.pair_temperature <= 0:
        raise RuntimeError("Calibration temperatures must be positive")
    return structured_permutation_logits(
        stage_logits.float() / parameters.stage_temperature,
        pair_logits.float() / parameters.pair_temperature,
        stage_weight=1.0,
        pair_weight=parameters.pair_weight,
    )


def _metrics(payload: dict[str, Any], logits: torch.Tensor) -> dict[str, Any]:
    return compute_stage_pair_metrics(
        logits,
        payload["target_perm_idx"],
        stage_logits=payload["stage_logits"],
        pair_logits=payload["pair_logits"],
        answer=payload["answer"],
    )


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    distance = (
        abs(float(row["pair_weight"]) - 0.3)
        + abs(float(row["stage_temperature"]) - 1.0)
        + abs(float(row["pair_temperature"]) - 1.0)
    )
    return (
        -float(row["exact_match"]),
        -float(row["MRR"]),
        -float(row["top3_accuracy"]),
        distance,
        float(row["pair_weight"]),
        float(row["stage_temperature"]),
        float(row["pair_temperature"]),
    )


def search_calibration_grid(
    payload: dict[str, Any],
    *,
    pair_weights: Iterable[float] = DEFAULT_PAIR_WEIGHTS,
    stage_temperatures: Iterable[float] = DEFAULT_TEMPERATURES,
    pair_temperatures: Iterable[float] = DEFAULT_TEMPERATURES,
) -> tuple[CalibrationParameters, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for pair_weight in pair_weights:
        for stage_temperature in stage_temperatures:
            for pair_temperature in pair_temperatures:
                parameters = CalibrationParameters(
                    float(pair_weight), float(stage_temperature), float(pair_temperature)
                )
                metrics = _metrics(
                    payload,
                    calibrated_structured_logits(payload["stage_logits"], payload["pair_logits"], parameters),
                )
                rows.append({**parameters.as_dict(), **metrics})
    if not rows:
        raise RuntimeError("Calibration grid is empty")
    selected = min(rows, key=_selection_key)
    return (
        CalibrationParameters(
            float(selected["pair_weight"]),
            float(selected["stage_temperature"]),
            float(selected["pair_temperature"]),
        ),
        rows,
    )


def _comparison(raw_logits: torch.Tensor, calibrated_logits: torch.Tensor, targets: torch.Tensor) -> dict[str, int]:
    raw_correct = raw_logits.argmax(dim=1).cpu().eq(targets.cpu())
    calibrated_correct = calibrated_logits.argmax(dim=1).cpu().eq(targets.cpu())
    return {
        "broken": int((raw_correct & ~calibrated_correct).sum().item()),
        "fixed": int((~raw_correct & calibrated_correct).sum().item()),
        "unchanged_correct": int((raw_correct & calibrated_correct).sum().item()),
        "unchanged_wrong": int((~raw_correct & ~calibrated_correct).sum().item()),
    }


def _write_grid_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_weight",
        "stage_temperature",
        "pair_temperature",
        "exact_match",
        "MRR",
        "top3_accuracy",
        "stage_accuracy",
        "pairwise_head_accuracy",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_calibration(
    payload: dict[str, Any],
    output_dir: str | Path,
    *,
    tune_split: str,
    pair_weights: Iterable[float] = DEFAULT_PAIR_WEIGHTS,
    stage_temperatures: Iterable[float] = DEFAULT_TEMPERATURES,
    pair_temperatures: Iterable[float] = DEFAULT_TEMPERATURES,
) -> dict[str, Any]:
    normalized_split = tune_split.lower().replace("-", "_")
    if normalized_split not in {"valid_a", "val_a"}:
        raise RuntimeError(f"Calibration tuning is restricted to valid-A, got {tune_split!r}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected, grid = search_calibration_grid(
        payload,
        pair_weights=pair_weights,
        stage_temperatures=stage_temperatures,
        pair_temperatures=pair_temperatures,
    )
    raw_parameters = CalibrationParameters(0.3, 1.0, 1.0)
    raw_logits = calibrated_structured_logits(payload["stage_logits"], payload["pair_logits"], raw_parameters)
    calibrated_logits = calibrated_structured_logits(payload["stage_logits"], payload["pair_logits"], selected)
    raw_metrics = _metrics(payload, raw_logits)
    calibrated_metrics = _metrics(payload, calibrated_logits)
    comparison = {
        "raw_exact": raw_metrics["exact_match"],
        "calibrated_exact": calibrated_metrics["exact_match"],
        "raw_MRR": raw_metrics["MRR"],
        "calibrated_MRR": calibrated_metrics["MRR"],
        "raw_top3": raw_metrics["top3_accuracy"],
        "calibrated_top3": calibrated_metrics["top3_accuracy"],
        **_comparison(raw_logits, calibrated_logits, payload["target_perm_idx"]),
        **selected.as_dict(),
        "grid_result_count": len(grid),
    }
    calibration = {
        "format_version": 1,
        "tune_split": "valid_a",
        "stage_weight": 1.0,
        **selected.as_dict(),
        "permutation_table_fingerprint": permutation_table_fingerprint(),
    }
    prediction_rows = rows_from_logits(
        [str(value) for value in payload["ids"]],
        calibrated_logits,
        payload["target_perm_idx"],
    )
    prediction_fields = [
        "Id",
        "true_answer",
        "pred_answer",
        "true_perm_idx",
        "pred_perm_idx",
        "gt_rank",
        "top1_margin",
        "correct",
    ]
    _write_grid_csv(out / "calibration_grid.csv", grid)
    write_json(out / "calibration.json", calibration)
    write_json(out / "calibrated_metrics.json", calibrated_metrics)
    write_csv_rows(out / "calibrated_valid_predictions.csv", prediction_rows, prediction_fields)
    write_csv_rows(
        out / "calibrated_wrong_cases.csv",
        [row for row in prediction_rows if not int(row["correct"])],
        prediction_fields,
    )
    write_json(out / "raw_vs_calibrated_comparison.json", comparison)
    return {"calibration": calibration, "calibrated_metrics": calibrated_metrics, "comparison": comparison}


def load_calibration(path: str | Path) -> CalibrationParameters:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("tune_split") != "valid_a":
        raise RuntimeError("Calibration file was not selected on valid-A")
    if payload.get("permutation_table_fingerprint") != permutation_table_fingerprint():
        raise RuntimeError("Calibration permutation table fingerprint mismatch")
    return CalibrationParameters(
        float(payload["pair_weight"]),
        float(payload["stage_temperature"]),
        float(payload["pair_temperature"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-logits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tune-split", default="valid_a")
    args = parser.parse_args()
    cfg = load_config(args.config)
    calibration_cfg = cfg.get("calibration", {})
    if bool(calibration_cfg.get("lockbox_tuning", False)):
        raise RuntimeError("calibration.lockbox_tuning must remain false")
    result = run_calibration(
        load_raw_stage_pair_logits(args.raw_logits),
        args.output_dir,
        tune_split=args.tune_split,
        pair_weights=calibration_cfg.get("pair_weight_grid", DEFAULT_PAIR_WEIGHTS),
        stage_temperatures=calibration_cfg.get("stage_temperature_grid", DEFAULT_TEMPERATURES),
        pair_temperatures=calibration_cfg.get("pair_temperature_grid", DEFAULT_TEMPERATURES),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
