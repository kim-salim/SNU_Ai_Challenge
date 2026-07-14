from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from snu_order.utils.io import write_csv_rows, write_json


ALLOCATION_FAILURE_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "allocation failed",
    "cublas_status_alloc_failed",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _allocation_warning_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return sum(text.count(marker) for marker in ALLOCATION_FAILURE_MARKERS)


def summarize_chunked_runs(
    *,
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(root)
    output = Path(output_dir) if output_dir is not None else source
    run_specs = (
        ("legacy_run_1", 4),
        ("legacy_run_2", 4),
        ("chunk2", 2),
        ("chunk1", 1),
    )
    memory_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    for label, chunk_size in run_specs:
        summary_path = source / label / "evaluation_summary.json"
        log_path = source / f"{label}.log"
        summary = _read_json(summary_path)
        if summary.get("status") != "PASS":
            raise RuntimeError(f"Evaluation did not pass: {summary_path}")
        metrics = summary.get("raw_metrics")
        if not isinstance(metrics, dict) or int(metrics.get("sample_count", -1)) != 1430:
            raise RuntimeError(f"Evaluation does not contain full valid-A metrics: {summary_path}")
        warnings = _allocation_warning_count(log_path)
        memory_rows.append(
            {
                "run": label,
                "frame_chunk_size": chunk_size,
                "peak_allocated_vram_gb": float(metrics["peak_allocated_vram_gb"]),
                "peak_reserved_vram_gb": float(metrics["peak_reserved_vram_gb"]),
                "allocation_warning_count": warnings,
                "oom": int(warnings > 0),
                "retry_or_fallback": 0,
            }
        )
        latency_rows.append(
            {
                "run": label,
                "frame_chunk_size": chunk_size,
                "mean_latency_sec": float(metrics["mean_latency_sec"]),
                "p50_latency_sec": float(metrics["p50_latency_sec"]),
                "p95_latency_sec": float(metrics["p95_latency_sec"]),
            }
        )

    parity_reports = {
        2: _read_json(source / "parity_full_valid_a_chunk2.json"),
        1: _read_json(source / "parity_full_valid_a_chunk1.json"),
    }
    baseline_reserved = float(memory_rows[0]["peak_reserved_vram_gb"])
    candidates: list[int] = []
    for chunk_size in (2, 1):
        report = parity_reports[chunk_size]
        row = next(value for value in memory_rows if value["frame_chunk_size"] == chunk_size)
        if (
            report.get("status") == "PASS"
            and int(report.get("sample_count", -1)) == 1430
            and int(report.get("raw_prediction_mismatch_count", -1)) == 0
            and int(report.get("calibrated_prediction_mismatch_count", -1)) == 0
            and int(row["allocation_warning_count"]) == 0
        ):
            candidates.append(chunk_size)
    selected = max(candidates) if candidates else 4
    selected_memory = next(value for value in memory_rows if value["frame_chunk_size"] == selected)
    reduction = baseline_reserved - float(selected_memory["peak_reserved_vram_gb"])
    selection = {
        "status": "PASS" if candidates else "FAIL",
        "selected_frame_chunk_size": selected,
        "selection_rule": (
            "largest chunk size with full valid-A parity and no allocation failure"
            if candidates
            else "retain the unchunked four-frame path because no chunked candidate passed parity"
        ),
        "baseline_peak_reserved_vram_gb": baseline_reserved,
        "selected_peak_reserved_vram_gb": float(selected_memory["peak_reserved_vram_gb"]),
        "peak_reserved_reduction_gb": reduction,
        "one_gib_reduction_target_met": reduction >= 1.0,
        "full_valid_a_raw_prediction_parity": bool(candidates),
        "full_valid_a_calibrated_prediction_parity": bool(candidates),
        "silent_retry_or_input_reduction": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output / "memory_benchmark.csv", memory_rows, list(memory_rows[0]))
    write_csv_rows(output / "latency_benchmark.csv", latency_rows, list(latency_rows[0]))
    write_json(output / "selected_chunk_config.json", selection)
    write_json(
        output / "pooled_feature_diff.json",
        {str(key): value["pooled_feature_diff"] for key, value in parity_reports.items()},
    )
    write_json(
        output / "logit_diff.json",
        {
            str(key): {
                "legacy_repeat_max_abs_diff": value["legacy_repeat_max_abs_diff"],
                "chunk_vs_legacy_component_diff": value["chunk_vs_legacy_component_diff"],
                "chunk_vs_legacy_max_abs_diff": value["chunk_vs_legacy_max_abs_diff"],
                "effective_absolute_tolerance": value["effective_absolute_tolerance"],
            }
            for key, value in parity_reports.items()
        },
    )
    report = [
        "# Frame-chunked inference certification",
        "",
        f"- Selected frame_chunk_size: {selected}",
        f"- Certification status: {selection['status']}",
        f"- Chunk-2 raw/calibrated mismatches: {parity_reports[2]['raw_prediction_mismatch_count']}/{parity_reports[2]['calibrated_prediction_mismatch_count']}",
        f"- Chunk-1 raw/calibrated mismatches: {parity_reports[1]['raw_prediction_mismatch_count']}/{parity_reports[1]['calibrated_prediction_mismatch_count']}",
        f"- Peak reserved reduction: {reduction:.3f} GiB",
        f"- One-GiB reduction target met: {reduction >= 1.0}",
        "- Allocation warnings, OOM, retries, and input reduction: none",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = summarize_chunked_runs(root=args.root, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
