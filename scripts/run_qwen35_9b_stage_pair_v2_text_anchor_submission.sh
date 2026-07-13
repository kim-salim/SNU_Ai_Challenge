#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/.cache/torchinductor}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-qwen35_9b_stage_pair_v2_text_anchor}"
OVERWRITE="${OVERWRITE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
CONFIG="${CONFIG:-configs/exp/qwen35_9b_stage_pair_v2_text_anchor.yaml}"
CHECKPOINT="${CHECKPOINT:-weights/qwen35_9b_stage_pair_v2_text_anchor/$RUN_ID/best}"
CALIBRATION="${CALIBRATION:-$CHECKPOINT/calibration.json}"
METADATA_CSV="${METADATA_CSV:-data/raw/test.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-data/raw}"
SAMPLE_SUBMISSION="${SAMPLE_SUBMISSION:-data/raw/sample_submission.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/experiments/qwen35_9b_stage_pair_v2_text_anchor/$RUN_ID/submission}"
OUTPUT_CSV="${OUTPUT_CSV:-$OUTPUT_DIR/submission.csv}"
DEBUG_CSV="${DEBUG_CSV:-$OUTPUT_DIR/submission_debug.csv}"
PROFILE_JSON="${PROFILE_JSON:-$OUTPUT_DIR/inference_profile.json}"
LOG_DIR="$ROOT_DIR/logs/qwen35_9b_stage_pair_v2_text_anchor"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ ! -f "$CALIBRATION" ]]; then
  echo "Valid-A calibration file is required: $CALIBRATION"
  exit 1
fi
if [[ -e "$OUTPUT_CSV" && "$OVERWRITE" != "1" ]]; then
  echo "Refusing to overwrite existing submission: $OUTPUT_CSV"
  exit 1
fi

VERIFY_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.verify_stage_pair_checkpoint
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --metadata-csv data/splits/ab_v1/valid_a_v1.csv
  --image-root data/raw
  --max-samples 8
  --output-json "$OUTPUT_DIR/checkpoint_verification.json")
echo "${VERIFY_CMD[*]}"
"${VERIFY_CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_ID}_submission_verify.log"

INFER_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.inference_stage_pair
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --calibration "$CALIBRATION"
  --metadata-csv "$METADATA_CSV"
  --image-root "$IMAGE_ROOT"
  --sample-submission "$SAMPLE_SUBMISSION"
  --output-csv "$OUTPUT_CSV"
  --debug-csv "$DEBUG_CSV"
  --profile-json "$PROFILE_JSON"
  --max-samples "$MAX_SAMPLES"
  --device-index 0)
echo "${INFER_CMD[*]}"
"${INFER_CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_ID}_submission.log"

echo "Submission: $OUTPUT_CSV"
echo "Debug: $DEBUG_CSV"
echo "Runtime profile: $PROFILE_JSON"
