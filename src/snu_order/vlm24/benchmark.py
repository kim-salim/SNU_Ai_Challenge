from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from snu_order.utils.config import load_config
from snu_order.utils.io import read_csv_rows
from snu_order.vlm24.candidates import build_24_candidates
from snu_order.vlm24.dataset import VLM24MetadataDataset
from snu_order.vlm24.prompt_builder import build_prompt
from snu_order.vlm24.qwen25_adapter import Qwen25VLAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--image-mode", choices=["multi_image", "grid_2x2"], default=None)
    parser.add_argument("--scoring-mode", choices=["option_label_logprob", "direct_generation"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_mode = args.image_mode or str(cfg.get("input", {}).get("image_mode", "multi_image"))
    scoring_mode = args.scoring_mode or str(cfg.get("scoring", {}).get("mode", "option_label_logprob"))
    load_start = time.perf_counter()
    adapter = Qwen25VLAdapter(cfg, image_mode=image_mode, scoring_mode=scoring_mode)
    adapter.load_model_and_processor()
    load_time = time.perf_counter() - load_start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dataset = VLM24MetadataDataset(
        args.metadata_csv,
        args.image_root,
        has_answer=True,
        max_samples=args.max_samples if args.max_samples >= 0 else None,
    )
    candidates = build_24_candidates(str(cfg.get("candidate", {}).get("option_labels", "ABCDEFGHIJKLMNOPQRSTUVWX")))
    frame_labels = list(cfg.get("input", {}).get("frame_labels", ["F1", "F2", "F3", "F4"]))
    latencies: list[float] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        prompt = build_prompt(sample["sentence"], candidates, frame_labels=frame_labels)
        start = time.perf_counter()
        adapter.predict_one(prompt, sample["frames"], candidates)
        latency = time.perf_counter() - start
        latencies.append(latency)
        print(f"[{index + 1}/{len(dataset)}] latency={latency:.2f}s")

    test_rows = read_csv_rows("data/raw/test.csv")
    mean_latency = statistics.mean(latencies) if latencies else 0.0
    report = {
        "model_load_time_sec": load_time,
        "num_samples": len(dataset),
        "average_sec_per_sample": mean_latency,
        "p50_latency_sec": statistics.median(latencies) if latencies else 0.0,
        "p90_latency_sec": sorted(latencies)[int(0.9 * (len(latencies) - 1))] if latencies else 0.0,
        "max_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        "estimated_full_test_time_hours": mean_latency * len(test_rows) / 3600.0 if test_rows else None,
        "under_24h": (mean_latency * len(test_rows) / 3600.0) <= 24.0 if test_rows else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
