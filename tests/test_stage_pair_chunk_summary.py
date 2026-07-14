import json

from snu_order.qwen3vl.summarize_chunked_inference import summarize_chunked_runs


def test_chunk_summary_selects_largest_fully_certified_chunk(tmp_path):
    for label, chunk_size, reserved in (
        ("legacy_run_1", 4, 28.0),
        ("legacy_run_2", 4, 28.0),
        ("chunk2", 2, 23.0),
        ("chunk1", 1, 19.0),
    ):
        run = tmp_path / label
        run.mkdir()
        summary = {
            "status": "PASS",
            "raw_metrics": {
                "sample_count": 1430,
                "peak_allocated_vram_gb": reserved - 1.0,
                "peak_reserved_vram_gb": reserved,
                "mean_latency_sec": 0.1 * chunk_size,
                "p50_latency_sec": 0.1 * chunk_size,
                "p95_latency_sec": 0.2 * chunk_size,
            },
        }
        (run / "evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (tmp_path / f"{label}.log").write_text("clean run\n", encoding="utf-8")
    parity = {
        "status": "PASS",
        "sample_count": 1430,
        "raw_prediction_mismatch_count": 0,
        "calibrated_prediction_mismatch_count": 0,
        "pooled_feature_diff": {"max": 0.0},
        "legacy_repeat_max_abs_diff": 0.0,
        "chunk_vs_legacy_component_diff": {},
        "chunk_vs_legacy_max_abs_diff": 0.0,
        "effective_absolute_tolerance": 0.125,
    }
    for chunk_size in (1, 2):
        (tmp_path / f"parity_full_valid_a_chunk{chunk_size}.json").write_text(
            json.dumps(parity), encoding="utf-8"
        )
    result = summarize_chunked_runs(root=tmp_path)
    assert result["selected_frame_chunk_size"] == 2
    assert result["peak_reserved_reduction_gb"] == 5.0
