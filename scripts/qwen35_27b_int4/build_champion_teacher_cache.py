from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from snu_order.data.validate_submission import find_column, parse_answer
from snu_order.qwen3vl.calibration_stage_pair import load_raw_stage_pair_logits
from snu_order.qwen3vl.champion_retention import CACHE_FORMAT, file_sha256, permutation_semantic_sha256
from snu_order.qwen3vl.permutations import answer_to_perm_index
from snu_order.qwen3vl.stage_pair_scorer import structured_permutation_logits
from snu_order.utils.io import read_csv_rows, write_json


ID_CANDIDATES = ["Id", "ID", "id", "sample_id", "sampleId"]
ANSWER_CANDIDATES = ["Answer", "answer", "label", "true_answer"]


def shard_csv(source: Path, output_dir: Path, count: int) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("shard count must be positive")
    rows = read_csv_rows(source)
    if not rows:
        raise RuntimeError("teacher source CSV is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    shards: list[dict[str, Any]] = []
    for shard in range(count):
        selected = rows[shard::count]
        path = output_dir / f"teacher_shard_{shard:02d}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
        shards.append({"index": shard, "path": str(path), "sample_count": len(selected), "sha256": file_sha256(path)})
    manifest = {
        "status": "PASS",
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "sample_count": len(rows),
        "shard_count": count,
        "shards": shards,
    }
    write_json(output_dir / "shard_manifest.json", manifest)
    return manifest


def merge_cache(source_csv: Path, raw_paths: list[Path], checkpoint: Path, output: Path) -> dict[str, Any]:
    rows = read_csv_rows(source_csv)
    columns = list(rows[0])
    id_col = find_column(columns, ID_CANDIDATES)
    answer_col = find_column(columns, ANSWER_CANDIDATES)
    if id_col is None or answer_col is None:
        raise RuntimeError("teacher source CSV lacks Id or Answer")
    expected_ids = [str(row[id_col]) for row in rows]
    expected_answers = {str(row[id_col]): parse_answer(row[answer_col]) for row in rows}
    merged: dict[str, dict[str, torch.Tensor]] = {}
    raw_hashes: list[dict[str, str]] = []
    for raw_path in raw_paths:
        payload = load_raw_stage_pair_logits(raw_path)
        raw_hashes.append({"path": str(raw_path.resolve()), "sha256": file_sha256(raw_path)})
        for index, sample_id_value in enumerate(payload["ids"]):
            sample_id = str(sample_id_value)
            if sample_id in merged:
                raise RuntimeError(f"duplicate teacher sample Id: {sample_id}")
            merged[sample_id] = {
                "stage_logits": payload["stage_logits"][index].float(),
                "pair_logits": payload["pair_logits"][index].float(),
                "target_perm_idx": payload["target_perm_idx"][index].long(),
                "answer": payload["answer"][index].long(),
            }
    if set(merged) != set(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("teacher shard merge does not exactly cover the source split")
    for sample_id in expected_ids:
        observed = merged[sample_id]["answer"].tolist()
        if observed != expected_answers[sample_id]:
            raise RuntimeError(f"teacher Answer binding mismatch for Id={sample_id}")
        if int(merged[sample_id]["target_perm_idx"].item()) != answer_to_perm_index(observed):
            raise RuntimeError(f"teacher target mapping mismatch for Id={sample_id}")
    stage = torch.stack([merged[sample_id]["stage_logits"] for sample_id in expected_ids])
    pair = torch.stack([merged[sample_id]["pair_logits"] for sample_id in expected_ids])
    targets = torch.stack([merged[sample_id]["target_perm_idx"] for sample_id in expected_ids])
    answers = torch.stack([merged[sample_id]["answer"] for sample_id in expected_ids])
    final = structured_permutation_logits(stage, pair, stage_weight=1.0, pair_weight=0.3)
    correct = final.argmax(dim=1).eq(targets)
    identity = {
        "train_split": str(source_csv.resolve()),
        "train_split_sha256": file_sha256(source_csv),
        "teacher_checkpoint": str(checkpoint.resolve()),
        "teacher_checkpoint_manifest_sha256": file_sha256(checkpoint / "checkpoint_manifest.json"),
        "teacher_adapter_sha256": file_sha256(checkpoint / "adapter" / "adapter_model.safetensors"),
        "teacher_heads_sha256": file_sha256(checkpoint / "heads.pt"),
        "permutation_semantic_sha256": permutation_semantic_sha256(),
        "raw_shards": raw_hashes,
    }
    payload = {
        "cache_format": CACHE_FORMAT,
        "ids": expected_ids,
        "stage_logits": stage,
        "pair_logits": pair,
        "target_perm_idx": targets,
        "answer": answers,
        "teacher_correct": correct,
        "identity": identity,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "status": "PASS",
        "cache": str(output.resolve()),
        "cache_sha256": file_sha256(output),
        "sample_count": len(expected_ids),
        "teacher_correct_count": int(correct.sum().item()),
        "teacher_exact": float(correct.float().mean().item()),
        "identity": identity,
    }
    write_json(output.with_suffix(".manifest.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--source-csv", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--count", type=int, default=8)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--source-csv", required=True)
    merge.add_argument("--raw", action="append", required=True)
    merge.add_argument("--checkpoint", required=True)
    merge.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "shard":
        result = shard_csv(Path(args.source_csv), Path(args.output_dir), args.count)
    else:
        result = merge_cache(Path(args.source_csv), [Path(value) for value in args.raw], Path(args.checkpoint), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
