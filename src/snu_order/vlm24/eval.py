from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from snu_order.data.metric import exact_match_accuracy
from snu_order.utils.config import load_config
from snu_order.utils.io import ensure_parent, read_csv_rows, write_csv_rows, write_json
from snu_order.vlm24.candidates import (
    answer_to_order,
    build_24_candidates,
    deterministic_shuffle_candidates,
)
from snu_order.vlm24.dataset import VLM24MetadataDataset
from snu_order.vlm24.prompt_builder import build_prompt
from snu_order.vlm24.qwen25_adapter import Qwen25VLAdapter


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = deepcopy(cfg)
    if args.model_name:
        out.setdefault("model", {})["name"] = args.model_name
    if args.local_files_only:
        out.setdefault("model", {})["local_files_only"] = True
    if args.image_mode:
        out.setdefault("input", {})["image_mode"] = args.image_mode
    if args.scoring_mode:
        out.setdefault("scoring", {})["mode"] = args.scoring_mode
    if args.deterministic_option_shuffle:
        out.setdefault("candidate", {})["deterministic_option_shuffle"] = True
    if args.max_new_tokens is not None:
        out.setdefault("scoring", {})["max_new_tokens"] = int(args.max_new_tokens)
    return out


def _top_scores(scores: list[float] | None) -> tuple[float | None, float | None, float | None]:
    if not scores:
        return None, None, None
    ordered = sorted(scores, reverse=True)
    top1 = float(ordered[0])
    top2 = float(ordered[1]) if len(ordered) > 1 else None
    margin = None if top2 is None else top1 - top2
    return top1, top2, margin


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _test_size() -> int | None:
    path = Path("data/raw/test.csv")
    if not path.exists():
        return None
    return len(read_csv_rows(path))


def run_eval(
    cfg: dict[str, Any],
    metadata_csv: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    max_samples: int,
    image_mode: str,
    scoring_mode: str,
    benchmark: bool,
) -> dict[str, Any]:
    del benchmark
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = VLM24MetadataDataset(
        metadata_csv,
        image_root,
        has_answer=True,
        max_samples=max_samples if max_samples >= 0 else None,
    )
    adapter = Qwen25VLAdapter(cfg, image_mode=image_mode, scoring_mode=scoring_mode)
    adapter.load_model_and_processor()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    candidate_cfg = cfg.get("candidate", {})
    prompt_cfg = cfg.get("prompt", {})
    input_cfg = cfg.get("input", {})
    option_labels = str(candidate_cfg.get("option_labels", "ABCDEFGHIJKLMNOPQRSTUVWX"))
    do_shuffle = bool(candidate_cfg.get("deterministic_option_shuffle", False))
    frame_labels = list(input_cfg.get("frame_labels", ["F1", "F2", "F3", "F4"]))
    use_cot = bool(prompt_cfg.get("use_cot", False))
    base_candidates = build_24_candidates(option_labels=option_labels)

    pred_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    raw_scores_path = out / "raw_scores.jsonl"
    if raw_scores_path.exists():
        raw_scores_path.unlink()

    pred_answers: list[list[int]] = []
    true_answers: list[list[int]] = []
    latencies: list[float] = []
    margins: list[float] = []
    parse_failures = 0
    option_counts: Counter[str] = Counter()

    for index in range(len(dataset)):
        sample = dataset[index]
        candidates = (
            deterministic_shuffle_candidates(base_candidates, str(sample["id"]), option_labels=option_labels)
            if do_shuffle
            else base_candidates
        )
        prompt = build_prompt(
            sentence=str(sample["sentence"]),
            candidates=candidates,
            frame_labels=frame_labels,
            use_cot=use_cot,
        )
        start = time.perf_counter()
        result = adapter.predict_one(prompt, sample["frames"], candidates)
        latency = time.perf_counter() - start
        latencies.append(latency)

        gt_answer = sample["answer"]
        gt_order = list(answer_to_order(gt_answer))
        pred_answer = result["pred_answer"]
        pred_order = result["pred_order"]
        exact = int(pred_answer == gt_answer) if pred_answer is not None else 0
        if pred_answer is None:
            parse_failures += 1
            pred_answer_for_metric = [1, 2, 3, 4]
        else:
            pred_answer_for_metric = pred_answer
        pred_answers.append(pred_answer_for_metric)
        true_answers.append(gt_answer)

        scores = result.get("scores")
        top1, top2, margin = _top_scores(scores)
        if margin is not None:
            margins.append(float(margin))
        pred_option = result.get("pred_option")
        if pred_option is not None:
            option_counts[str(pred_option)] += 1

        row = {
            "id": sample["id"],
            "gt_answer": _json_cell(gt_answer),
            "pred_answer": _json_cell(pred_answer),
            "gt_order": _json_cell(gt_order),
            "pred_order": _json_cell(pred_order),
            "pred_option": pred_option or "",
            "exact_match": exact,
            "margin": "" if margin is None else margin,
            "top1_score": "" if top1 is None else top1,
            "top2_score": "" if top2 is None else top2,
            "parse_status": result.get("parse_status", ""),
            "latency_sec": latency,
            "raw_output": result.get("raw_output", ""),
        }
        pred_rows.append(row)
        if not exact:
            wrong_rows.append(row)
        if scores is not None:
            ensure_parent(raw_scores_path)
            with raw_scores_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "id": sample["id"],
                            "labels": [candidate["label"] for candidate in candidates],
                            "orders": [list(candidate["order"]) for candidate in candidates],
                            "scores": scores,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(
            f"[{index + 1}/{len(dataset)}] id={sample['id']} option={pred_option} "
            f"exact={exact} latency={latency:.2f}s"
        )

    fieldnames = [
        "id",
        "gt_answer",
        "pred_answer",
        "gt_order",
        "pred_order",
        "pred_option",
        "exact_match",
        "margin",
        "top1_score",
        "top2_score",
        "parse_status",
        "latency_sec",
        "raw_output",
    ]
    write_csv_rows(out / "valid_predictions.csv", pred_rows, fieldnames)
    write_csv_rows(out / "wrong_cases.csv", wrong_rows, fieldnames)

    mean_latency = statistics.mean(latencies) if latencies else 0.0
    test_n = _test_size()
    metrics = {
        "num_samples": len(dataset),
        "exact_match_accuracy": exact_match_accuracy(pred_answers, true_answers),
        "parsing_failure_rate": parse_failures / max(len(dataset), 1),
        "option_distribution": dict(sorted(option_counts.items())),
        "average_margin": statistics.mean(margins) if margins else 0.0,
        "mean_latency_sec": mean_latency,
        "p50_latency_sec": statistics.median(latencies) if latencies else 0.0,
        "max_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        "estimated_test_time_hours": None if test_n is None else mean_latency * test_n / 3600.0,
    }
    write_json(out / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-mode", choices=["multi_image", "grid_2x2"], default=None)
    parser.add_argument("--scoring-mode", choices=["option_label_logprob", "direct_generation"], default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--deterministic-option-shuffle", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = _apply_overrides(load_config(args.config), args)
    output_dir = args.output_dir or str(cfg.get("paths", {}).get("prediction_dir", "outputs/predictions/qwen25vl_7b_24candidate"))
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = int(cfg.get("runtime", {}).get("max_samples", -1))
    image_mode = args.image_mode or str(cfg.get("input", {}).get("image_mode", "multi_image"))
    scoring_mode = args.scoring_mode or str(cfg.get("scoring", {}).get("mode", "option_label_logprob"))
    run_eval(
        cfg=cfg,
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        output_dir=output_dir,
        max_samples=max_samples,
        image_mode=image_mode,
        scoring_mode=scoring_mode,
        benchmark=bool(args.benchmark),
    )


if __name__ == "__main__":
    main()
