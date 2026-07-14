from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from snu_order.data.validate_submission import validate_submission
from snu_order.pipeline.make_submission import save_submission
from snu_order.utils.config import load_config
from snu_order.utils.io import write_csv_rows
from snu_order.vlm24.candidates import (
    build_24_candidates,
    deterministic_shuffle_candidates,
    validate_answer,
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
    if args.max_new_tokens is not None:
        out.setdefault("scoring", {})["max_new_tokens"] = int(args.max_new_tokens)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--image-mode", choices=["multi_image", "grid_2x2"], default=None)
    parser.add_argument("--scoring-mode", choices=["option_label_logprob", "direct_generation"], default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = _apply_overrides(load_config(args.config), args)
    image_mode = args.image_mode or str(cfg.get("input", {}).get("image_mode", "multi_image"))
    scoring_mode = args.scoring_mode or str(cfg.get("scoring", {}).get("mode", "option_label_logprob"))
    dataset = VLM24MetadataDataset(
        args.metadata_csv,
        args.image_root,
        has_answer=False,
        max_samples=args.max_samples if args.max_samples >= 0 else None,
    )
    adapter = Qwen25VLAdapter(cfg, image_mode=image_mode, scoring_mode=scoring_mode)
    adapter.load_model_and_processor()

    candidate_cfg = cfg.get("candidate", {})
    prompt_cfg = cfg.get("prompt", {})
    input_cfg = cfg.get("input", {})
    option_labels = str(candidate_cfg.get("option_labels", "ABCDEFGHIJKLMNOPQRSTUVWX"))
    do_shuffle = bool(candidate_cfg.get("deterministic_option_shuffle", False))
    frame_labels = list(input_cfg.get("frame_labels", ["F1", "F2", "F3", "F4"]))
    base_candidates = build_24_candidates(option_labels=option_labels)

    ids: list[str] = []
    answers: list[list[int]] = []
    debug_rows: list[dict[str, Any]] = []
    output_csv = Path(args.output_csv)
    debug_csv = output_csv.with_name(output_csv.stem + "_debug.csv")

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
            use_cot=bool(prompt_cfg.get("use_cot", False)),
        )
        start = time.perf_counter()
        result = adapter.predict_one(prompt, sample["frames"], candidates)
        latency = time.perf_counter() - start
        pred_answer = result.get("pred_answer")
        if pred_answer is None:
            pred_answer = [1, 2, 3, 4]
        pred_answer = validate_answer(pred_answer)
        ids.append(str(sample["id"]))
        answers.append(pred_answer)
        scores = result.get("scores") or []
        sorted_scores = sorted([float(v) for v in scores], reverse=True)
        top1 = sorted_scores[0] if sorted_scores else ""
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else ""
        margin = "" if top1 == "" or top2 == "" else float(top1) - float(top2)
        debug_rows.append(
            {
                "id": sample["id"],
                "pred_answer": json.dumps(pred_answer),
                "pred_option": result.get("pred_option") or "",
                "pred_order": json.dumps(result.get("pred_order")),
                "margin": margin,
                "top1_score": top1,
                "top2_score": top2,
                "parse_status": result.get("parse_status", ""),
                "latency_sec": latency,
                "raw_output": result.get("raw_output", ""),
            }
        )
        print(f"[{index + 1}/{len(dataset)}] id={sample['id']} answer={pred_answer} latency={latency:.2f}s")

    full_run = args.max_samples is None or args.max_samples < 0
    save_submission(ids, answers, output_csv, reference=args.sample_submission if full_run else None)
    if full_run:
        validate_submission(output_csv, args.sample_submission)
    write_csv_rows(
        debug_csv,
        debug_rows,
        [
            "id",
            "pred_answer",
            "pred_option",
            "pred_order",
            "margin",
            "top1_score",
            "top2_score",
            "parse_status",
            "latency_sec",
            "raw_output",
        ],
    )
    print(f"saved submission: {output_csv}")
    print(f"saved debug csv: {debug_csv}")


if __name__ == "__main__":
    main()
