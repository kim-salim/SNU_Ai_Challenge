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
RUN_ID="${RUN_ID:-qlora_stage_pair_fixed_ddp}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/experiments/qwen3vl_stage_pair/$RUN_ID}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/exp/qwen3vl_stage_pair_qlora.yaml}"
CHECKPOINT="${CHECKPOINT:-weights/qwen3vl_stage_pair/$RUN_ID/best}"
METADATA_CSV="${METADATA_CSV:-data/raw/test.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-data/raw}"
SAMPLE_SUBMISSION="${SAMPLE_SUBMISSION:-data/raw/sample_submission.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-$OUT_DIR/submission.csv}"
DEBUG_CSV="${DEBUG_CSV:-$OUT_DIR/submission_debug.csv}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"

if [[ -e "$OUTPUT_CSV" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_CSV"
  echo "Set OVERWRITE=1 to replace it."
  exit 1
fi

CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.inference_stage_pair
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --metadata-csv "$METADATA_CSV"
  --image-root "$IMAGE_ROOT"
  --sample-submission "$SAMPLE_SUBMISSION"
  --output-csv "$OUTPUT_CSV"
  --debug-csv "$DEBUG_CSV"
  --max-samples "$MAX_SAMPLES"
  --device-index "$DEVICE_INDEX")

echo "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/submission.log"
echo "Submission: $OUTPUT_CSV"
echo "Debug: $DEBUG_CSV"
