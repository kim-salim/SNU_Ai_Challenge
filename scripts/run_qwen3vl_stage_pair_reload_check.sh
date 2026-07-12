#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/.cache/torchinductor}"

LOG_DIR="$ROOT_DIR/logs/qwen3vl_stage_pair"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-qlora_stage_pair_fixed_ddp}"
CONFIG="${CONFIG:-configs/exp/qwen3vl_stage_pair_qlora.yaml}"
CHECKPOINT="${CHECKPOINT:-weights/qwen3vl_stage_pair/$RUN_ID/best}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/experiments/qwen3vl_stage_pair/$RUN_ID/reload_valid_a}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
MIN_EXACT="${MIN_EXACT:-0.30}"

CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.evaluate_stage_pair
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --mode qlora_stage_pair
  --output-dir "$OUTPUT_DIR"
  --max-samples "$MAX_SAMPLES"
  --device-index "$DEVICE_INDEX")

echo "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_ID}_reload_check.log"

"$PYTHON_BIN" - "$OUTPUT_DIR/metrics.json" "$MIN_EXACT" <<'PY'
import json
import sys

metrics_path = sys.argv[1]
min_exact = float(sys.argv[2])
metrics = json.load(open(metrics_path, encoding="utf-8"))
exact = float(metrics.get("exact_match", 0.0))
correct = int(metrics.get("correct_count", 0))
count = int(metrics.get("sample_count", 0))
print(json.dumps({"reload_exact_match": exact, "correct_count": correct, "sample_count": count, "min_exact": min_exact}, ensure_ascii=False))
if exact < min_exact:
    raise SystemExit(f"reload check failed: exact_match={exact:.6f} < {min_exact:.6f}")
PY
