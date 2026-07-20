from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.io import write_csv_rows, write_json

from .metrics24 import rows_from_logits
from .metrics_stage_pair import compute_stage_pair_metrics
from .permutations import PERMS
from .stage_pair_scorer import pair_scores_from_logits, stage_scores_from_logits, structured_permutation_logits
from .canonical_stage_pair_evaluation import (
    canonical_cpu_float32_scores,
    semantic_prediction_sha256,
    stable_ranking,
)


DEFAULT_PAIR_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
DEFAULT_TEMPERATURES = (0.8, 1.0, 1.2, 1.5)
DEFAULT_FOLD_COUNT = 5
FOLD_ASSIGNMENT_METHOD = "sha256_hex8_mod_n"
REQUIRED_BINDING_KEYS = {
    "checkpoint_manifest_sha256",
    "adapter_sha256",
    "heads_sha256",
    "prompt_fingerprint_sha256",
    "processor_fingerprint_sha256",
    "permutation_mapping_sha256",
    "validation_split_sha256",
    "scorer_code_sha256",
}


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_raw_stage_pair_logits(
    path: str | Path,
    *,
    ids: list[str],
    stage_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    target_perm_idx: torch.Tensor,
    answer: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> None:
    stage = stage_logits.detach().float().cpu()
    pair = pair_logits.detach().float().cpu()
    targets = target_perm_idx.detach().long().cpu()
    raw_fused = canonical_cpu_float32_scores(stage, pair, stage_weight=1.0, pair_weight=0.3)
    ranking = stable_ranking(raw_fused, targets)
    assert ranking.gt_rank is not None
    payload = {
        "format_version": 2,
        "ids": [str(value) for value in ids],
        "stage_logits": stage,
        "pair_logits": pair,
        "target_perm_idx": targets,
        "answer": answer.detach().long().cpu(),
        "permutation_table_fingerprint": permutation_table_fingerprint(),
        "stage_component_scores": stage_scores_from_logits(stage).float().cpu(),
        "set_component_scores": None,
        "set_component_available": False,
        "pair_component_scores": pair_scores_from_logits(pair).float().cpu(),
        "raw_fused_scores": raw_fused.cpu(),
        "raw_prediction": ranking.prediction,
        "true_class_rank": ranking.gt_rank,
        "top1_margin": ranking.top1_margin,
        "metadata": {
            **dict(metadata or {}),
            "scorer_mode": "canonical_cpu_float32",
            "semantic_prediction_sha256": semantic_prediction_sha256(ids, ranking),
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_raw_stage_pair_logits(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required_v1 = {
        "format_version",
        "ids",
        "stage_logits",
        "pair_logits",
        "target_perm_idx",
        "answer",
        "permutation_table_fingerprint",
    }
    required_v2 = required_v1 | {
        "stage_component_scores",
        "set_component_scores",
        "set_component_available",
        "pair_component_scores",
        "raw_fused_scores",
        "raw_prediction",
        "true_class_rank",
        "top1_margin",
        "metadata",
    }
    if not isinstance(payload, dict):
        raise RuntimeError("Raw stage/pair artifact does not match the expected schema")
    version = int(payload.get("format_version", -1))
    expected = required_v1 if version == 1 else required_v2 if version == 2 else set()
    if not expected or set(payload) != expected:
        raise RuntimeError("Raw stage/pair artifact does not match the expected schema")
    if payload["permutation_table_fingerprint"] != permutation_table_fingerprint():
        raise RuntimeError("Permutation table fingerprint mismatch in raw stage/pair artifact")
    count = len(payload["ids"])
    for key in ("stage_logits", "pair_logits", "target_perm_idx", "answer"):
        if int(payload[key].shape[0]) != count:
            raise RuntimeError(f"Raw artifact row count mismatch for {key}")
    if version == 2:
        for key in (
            "stage_component_scores",
            "pair_component_scores",
            "raw_fused_scores",
            "raw_prediction",
            "true_class_rank",
            "top1_margin",
        ):
            if int(payload[key].shape[0]) != count:
                raise RuntimeError(f"Raw artifact row count mismatch for {key}")
        if payload["set_component_scores"] is not None or payload["set_component_available"] is not False:
            raise RuntimeError("Stage/Set/Pair architecture has no standalone Set scoring head")
        if payload["metadata"].get("scorer_mode") == "canonical_cpu_float32":
            canonical = canonical_cpu_float32_scores(payload["stage_logits"], payload["pair_logits"])
            ranking = stable_ranking(canonical, payload["target_perm_idx"])
            assert ranking.gt_rank is not None
            if not torch.equal(ranking.prediction, payload["raw_prediction"].long().cpu()):
                raise RuntimeError("Raw artifact prediction is not canonical CPU float32 scorer output")
            if not torch.equal(ranking.gt_rank, payload["true_class_rank"].long().cpu()):
                raise RuntimeError("Raw artifact GT rank is not canonical stable-ranking output")
    return payload


def calibrated_structured_logits(
    stage_logits: torch.Tensor,
    pair_logits: torch.Tensor,
    parameters: CalibrationParameters,
) -> torch.Tensor:
    if parameters.stage_temperature <= 0 or parameters.pair_temperature <= 0:
        raise RuntimeError("Calibration temperatures must be positive")
    return canonical_cpu_float32_scores(
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


def assign_id_hash_folds(ids: Iterable[str], *, fold_count: int = DEFAULT_FOLD_COUNT) -> list[int]:
    if int(fold_count) < 2:
        raise RuntimeError("Calibration fold_count must be at least 2")
    return [
        int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:8], 16) % int(fold_count)
        for sample_id in ids
    ]


def _subset_payload(payload: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    index_values = indices.detach().long().cpu()
    return {
        "ids": [str(payload["ids"][int(index)]) for index in index_values.tolist()],
        "stage_logits": payload["stage_logits"][index_values],
        "pair_logits": payload["pair_logits"][index_values],
        "target_perm_idx": payload["target_perm_idx"][index_values],
        "answer": payload["answer"][index_values],
    }


def run_oof_calibration(
    payload: dict[str, Any],
    *,
    pair_weights: Iterable[float] = DEFAULT_PAIR_WEIGHTS,
    stage_temperatures: Iterable[float] = DEFAULT_TEMPERATURES,
    pair_temperatures: Iterable[float] = DEFAULT_TEMPERATURES,
    fold_count: int = DEFAULT_FOLD_COUNT,
) -> dict[str, Any]:
    pair_values = tuple(float(value) for value in pair_weights)
    stage_values = tuple(float(value) for value in stage_temperatures)
    pair_temperature_values = tuple(float(value) for value in pair_temperatures)
    folds = torch.tensor(assign_id_hash_folds(payload["ids"], fold_count=fold_count), dtype=torch.long)
    if folds.shape[0] != len(payload["ids"]):
        raise AssertionError("Fold assignment count mismatch")
    oof_logits = torch.empty((len(payload["ids"]), len(PERMS)), dtype=torch.float32)
    fold_results: list[dict[str, Any]] = []
    for fold_index in range(int(fold_count)):
        held_out = folds.eq(fold_index).nonzero(as_tuple=False).flatten()
        train = folds.ne(fold_index).nonzero(as_tuple=False).flatten()
        if not held_out.numel() or not train.numel():
            raise RuntimeError(f"ID-hash fold {fold_index} is empty")
        train_payload = _subset_payload(payload, train)
        selected, _ = search_calibration_grid(
            train_payload,
            pair_weights=pair_values,
            stage_temperatures=stage_values,
            pair_temperatures=pair_temperature_values,
        )
        held_payload = _subset_payload(payload, held_out)
        held_logits = calibrated_structured_logits(
            held_payload["stage_logits"], held_payload["pair_logits"], selected
        ).detach().float().cpu()
        oof_logits[held_out] = held_logits
        held_metrics = _metrics(held_payload, held_logits)
        fold_results.append(
            {
                "fold": fold_index,
                "train_count": int(train.numel()),
                "held_out_count": int(held_out.numel()),
                **selected.as_dict(),
                "correct_count": int(held_metrics["correct_count"]),
                "exact_match": float(held_metrics["exact_match"]),
                "MRR": float(held_metrics["MRR"]),
                "top3_accuracy": float(held_metrics["top3_accuracy"]),
                "top5_accuracy": float(held_metrics["top5_accuracy"]),
            }
        )
    assignments = [
        {"Id": str(sample_id), "fold": int(fold)}
        for sample_id, fold in zip(payload["ids"], folds.tolist(), strict=True)
    ]
    return {
        "fold_assignment_method": FOLD_ASSIGNMENT_METHOD,
        "fold_count": int(fold_count),
        "fold_assignment_sha256": _canonical_sha256(assignments),
        "assignments": assignments,
        "folds": fold_results,
        "logits": oof_logits,
        "metrics": _metrics(payload, oof_logits),
    }


def mcnemar_exact_p_value(broken: int, fixed: int) -> float:
    discordant = int(broken) + int(fixed)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(int(broken), int(fixed)) + 1))
    return min(1.0, 2.0 * tail / (2.0**discordant))


def _grid_fingerprint(
    pair_weights: Iterable[float],
    stage_temperatures: Iterable[float],
    pair_temperatures: Iterable[float],
) -> str:
    values = [
        [float(pair_weight), float(stage_temperature), float(pair_temperature)]
        for pair_weight in pair_weights
        for stage_temperature in stage_temperatures
        for pair_temperature in pair_temperatures
    ]
    return _canonical_sha256(values)


def _plateau_diagnostic(selected: CalibrationParameters, grid: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("pair_weight", "stage_temperature", "pair_temperature")
    selected_values = selected.as_dict()
    selected_row = next(
        row
        for row in grid
        if all(float(row[key]) == float(value) for key, value in selected_values.items())
    )
    exact_count = int(selected_row["correct_count"])
    value_sets = {key: sorted({float(row[key]) for row in grid}) for key in keys}
    neighbor_values: dict[str, set[float]] = {}
    for key in keys:
        values = value_sets[key]
        index = values.index(float(selected_values[key]))
        neighbor_values[key] = {
            values[neighbor_index]
            for neighbor_index in (index - 1, index + 1)
            if 0 <= neighbor_index < len(values)
        }
    adjacent = []
    for row in grid:
        changed = [key for key in keys if float(row[key]) != float(selected_values[key])]
        if len(changed) == 1 and float(row[changed[0]]) in neighbor_values[changed[0]]:
            adjacent.append(row)
    near = [row for row in adjacent if int(row["correct_count"]) >= exact_count - 1]
    return {
        "selected_correct_count": exact_count,
        "adjacent_point_count": len(adjacent),
        "adjacent_within_one_correct_count": len(near),
        "has_plateau": bool(near),
        "selected_on_grid_boundary": any(
            float(selected_values[key]) in {min(value_sets[key]), max(value_sets[key])}
            for key in keys
        ),
    }


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
        "correct_count",
        "exact_match",
        "MRR",
        "top3_accuracy",
        "top5_accuracy",
        "mean_gt_rank",
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
    artifact_bindings: dict[str, str] | None = None,
    fixed_diagnostic_parameters: CalibrationParameters | None = None,
    fold_count: int = DEFAULT_FOLD_COUNT,
) -> dict[str, Any]:
    normalized_split = tune_split.lower().replace("-", "_")
    if normalized_split not in {"valid_a", "val_a"}:
        raise RuntimeError(f"Calibration tuning is restricted to valid-A, got {tune_split!r}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair_values = tuple(float(value) for value in pair_weights)
    stage_values = tuple(float(value) for value in stage_temperatures)
    pair_temperature_values = tuple(float(value) for value in pair_temperatures)
    bindings = dict(artifact_bindings or {})
    if bindings and set(bindings) != REQUIRED_BINDING_KEYS:
        raise RuntimeError(
            f"Calibration artifact bindings differ from the required schema: "
            f"missing={sorted(REQUIRED_BINDING_KEYS - set(bindings))}, "
            f"unexpected={sorted(set(bindings) - REQUIRED_BINDING_KEYS)}"
        )
    selected, grid = search_calibration_grid(
        payload,
        pair_weights=pair_values,
        stage_temperatures=stage_values,
        pair_temperatures=pair_temperature_values,
    )
    raw_parameters = CalibrationParameters(0.3, 1.0, 1.0)
    raw_logits = calibrated_structured_logits(payload["stage_logits"], payload["pair_logits"], raw_parameters)
    calibrated_logits = calibrated_structured_logits(payload["stage_logits"], payload["pair_logits"], selected)
    raw_metrics = _metrics(payload, raw_logits)
    calibrated_metrics = _metrics(payload, calibrated_logits)
    oof = run_oof_calibration(
        payload,
        pair_weights=pair_values,
        stage_temperatures=stage_values,
        pair_temperatures=pair_temperature_values,
        fold_count=fold_count,
    )
    oof_logits = oof.pop("logits")
    fixed_parameters = fixed_diagnostic_parameters or selected
    fixed_logits = calibrated_structured_logits(
        payload["stage_logits"], payload["pair_logits"], fixed_parameters
    )
    fixed_metrics = _metrics(payload, fixed_logits)
    raw_vs_calibrated = _comparison(raw_logits, calibrated_logits, payload["target_perm_idx"])
    comparison = {
        "raw_exact": raw_metrics["exact_match"],
        "calibrated_exact": calibrated_metrics["exact_match"],
        "raw_MRR": raw_metrics["MRR"],
        "calibrated_MRR": calibrated_metrics["MRR"],
        "raw_top3": raw_metrics["top3_accuracy"],
        "calibrated_top3": calibrated_metrics["top3_accuracy"],
        "raw_top5": raw_metrics["top5_accuracy"],
        "calibrated_top5": calibrated_metrics["top5_accuracy"],
        **raw_vs_calibrated,
        **selected.as_dict(),
        "grid_result_count": len(grid),
        "mcnemar_exact_p_value": mcnemar_exact_p_value(
            raw_vs_calibrated["broken"], raw_vs_calibrated["fixed"]
        ),
        "plateau": _plateau_diagnostic(selected, grid),
    }
    calibration_grid_sha256 = _grid_fingerprint(
        pair_values, stage_values, pair_temperature_values
    )
    calibration = {
        "format_version": 2 if bindings else 1,
        "tune_split": "valid_a",
        "stage_weight": 1.0,
        **selected.as_dict(),
        "permutation_table_fingerprint": permutation_table_fingerprint(),
        "selected_metric": "exact_match,mrr,top3,default_distance,lexicographic",
        "tie_break_rule": "exact>Mrr>top3>distance_to_(0.3,1.0,1.0)>tuple_lexicographic",
        "calibration_grid_sha256": calibration_grid_sha256,
        "fold_assignment_sha256": oof["fold_assignment_sha256"],
        "fold_assignment_method": FOLD_ASSIGNMENT_METHOD,
        "fold_count": int(fold_count),
        "artifact_bindings": bindings,
        "raw_metrics": raw_metrics,
        "oof_metrics": oof["metrics"],
        "full_fit_metrics": calibrated_metrics,
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
    oof_prediction_rows = rows_from_logits(
        [str(value) for value in payload["ids"]],
        oof_logits,
        payload["target_perm_idx"],
    )
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
    write_json(out / "oof_metrics.json", oof["metrics"])
    write_json(out / "fold_calibration.json", {"folds": oof["folds"]})
    write_json(
        out / "fold_assignments.json",
        {
            "method": FOLD_ASSIGNMENT_METHOD,
            "sha256": oof["fold_assignment_sha256"],
            "assignments": oof["assignments"],
        },
    )
    write_csv_rows(out / "oof_valid_predictions.csv", oof_prediction_rows, prediction_fields)
    write_json(
        out / "fixed_calibration_diagnostic.json",
        {"parameters": fixed_parameters.as_dict(), "metrics": fixed_metrics},
    )
    return {
        "calibration": calibration,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "oof_metrics": oof["metrics"],
        "folds": oof["folds"],
        "fixed_diagnostic": {"parameters": fixed_parameters.as_dict(), "metrics": fixed_metrics},
        "comparison": comparison,
    }


def load_calibration(
    path: str | Path,
    *,
    expected_bindings: dict[str, str] | None = None,
) -> CalibrationParameters:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("tune_split") != "valid_a":
        raise RuntimeError("Calibration file was not selected on valid-A")
    if payload.get("permutation_table_fingerprint") != permutation_table_fingerprint():
        raise RuntimeError("Calibration permutation table fingerprint mismatch")
    if expected_bindings is not None:
        if set(expected_bindings) != REQUIRED_BINDING_KEYS:
            raise RuntimeError("Runtime calibration bindings do not match the required schema")
        saved_bindings = payload.get("artifact_bindings")
        if not isinstance(saved_bindings, dict) or set(saved_bindings) != REQUIRED_BINDING_KEYS:
            raise RuntimeError("Calibration is not bound to the required checkpoint artifacts")
        mismatches = {
            key: {"calibration": saved_bindings.get(key), "runtime": expected_bindings.get(key)}
            for key in REQUIRED_BINDING_KEYS
            if saved_bindings.get(key) != expected_bindings.get(key)
        }
        if mismatches:
            raise RuntimeError(f"Calibration/checkpoint artifact mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return CalibrationParameters(
        float(payload["pair_weight"]),
        float(payload["stage_temperature"]),
        float(payload["pair_temperature"]),
    )


def artifact_bindings_from_paths(values: Iterable[str]) -> dict[str, str]:
    paths: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = str(value).partition("=")
        if not separator or not key or not raw_path:
            raise RuntimeError(f"Calibration binding must use key=path syntax: {value!r}")
        if key in paths:
            raise RuntimeError(f"Duplicate calibration binding: {key}")
        paths[key] = Path(raw_path)
    if not paths:
        return {}
    if set(paths) != REQUIRED_BINDING_KEYS:
        raise RuntimeError(
            f"Calibration binding paths differ from the required schema: "
            f"missing={sorted(REQUIRED_BINDING_KEYS - set(paths))}, "
            f"unexpected={sorted(set(paths) - REQUIRED_BINDING_KEYS)}"
        )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Calibration binding source files are missing: {missing}")
    return {key: file_sha256(path) for key, path in paths.items()}


def checkpoint_calibration_bindings(
    checkpoint: str | Path,
    cfg: dict[str, Any],
) -> dict[str, str]:
    """Recompute every checkpoint/calibration binding used by inference."""
    root = Path(checkpoint)
    adapter_candidates = [
        root / "adapter" / "adapter_model.safetensors",
        root / "adapter" / "adapter_model.bin",
    ]
    adapters = [path for path in adapter_candidates if path.is_file()]
    if len(adapters) != 1:
        raise RuntimeError(
            "Expected exactly one adapter weight file while binding calibration, "
            f"found {[str(path) for path in adapters]}"
        )
    paths = {
        "checkpoint_manifest_sha256": root / "checkpoint_manifest.json",
        "adapter_sha256": adapters[0],
        "heads_sha256": root / "heads.pt",
        "prompt_fingerprint_sha256": root / "prompt_fingerprint.json",
        "processor_fingerprint_sha256": root / "processor" / "tokenizer_config.json",
        "permutation_mapping_sha256": root / "permutations.json",
        "validation_split_sha256": Path(str(get_by_path(cfg, "data.valid_split"))),
        "scorer_code_sha256": Path(__file__).with_name("stage_pair_scorer.py"),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot verify calibration/checkpoint binding; required files are missing: {missing}"
        )
    if set(paths) != REQUIRED_BINDING_KEYS:
        raise AssertionError("Runtime calibration binding schema drifted")
    return {key: file_sha256(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-logits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tune-split", default="valid_a")
    parser.add_argument("--binding", action="append", default=[])
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    parser.add_argument("--fixed-pair-weight", type=float, default=None)
    parser.add_argument("--fixed-stage-temperature", type=float, default=None)
    parser.add_argument("--fixed-pair-temperature", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    calibration_cfg = cfg.get("calibration", {})
    if bool(calibration_cfg.get("lockbox_tuning", False)):
        raise RuntimeError("calibration.lockbox_tuning must remain false")
    fixed_values = (
        args.fixed_pair_weight,
        args.fixed_stage_temperature,
        args.fixed_pair_temperature,
    )
    if any(value is not None for value in fixed_values) and not all(value is not None for value in fixed_values):
        raise RuntimeError("All three fixed calibration diagnostic parameters must be supplied together")
    fixed = None if fixed_values[0] is None else CalibrationParameters(*(float(value) for value in fixed_values))
    result = run_calibration(
        load_raw_stage_pair_logits(args.raw_logits),
        args.output_dir,
        tune_split=args.tune_split,
        pair_weights=calibration_cfg.get("pair_weight_grid", DEFAULT_PAIR_WEIGHTS),
        stage_temperatures=calibration_cfg.get("stage_temperature_grid", DEFAULT_TEMPERATURES),
        pair_temperatures=calibration_cfg.get("pair_temperature_grid", DEFAULT_TEMPERATURES),
        artifact_bindings=artifact_bindings_from_paths(args.binding),
        fixed_diagnostic_parameters=fixed,
        fold_count=args.fold_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
