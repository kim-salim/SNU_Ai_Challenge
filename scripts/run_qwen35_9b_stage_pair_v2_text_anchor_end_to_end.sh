#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ID="${RUN_ID:-qwen35_9b_stage_pair_v2_text_anchor}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
SUBMISSION_MAX_SAMPLES="${SUBMISSION_MAX_SAMPLES:--1}"
EPOCHS="${EPOCHS:-4}"
RUN_LOCKBOX="${RUN_LOCKBOX:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29571}"
SMOKE="${SMOKE:-0}"

if [[ "$SMOKE" == "1" ]]; then
  echo "End-to-end mode does not submit from a smoke checkpoint."
  echo "Run SMOKE=1 scripts/run_qwen35_9b_stage_pair_v2_text_anchor.sh instead."
  exit 2
fi

TRAIN_SCRIPT="$ROOT_DIR/scripts/run_qwen35_9b_stage_pair_v2_text_anchor.sh"
SUBMISSION_SCRIPT="$ROOT_DIR/scripts/run_qwen35_9b_stage_pair_v2_text_anchor_submission.sh"

echo "Starting E1 training pipeline: RUN_ID=$RUN_ID"
PYTHON_BIN="$PYTHON_BIN" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
RUN_ID="$RUN_ID" \
MAX_SAMPLES="$TRAIN_MAX_SAMPLES" \
EPOCHS="$EPOCHS" \
RUN_LOCKBOX="$RUN_LOCKBOX" \
OVERWRITE="$OVERWRITE" \
MAIN_PROCESS_PORT="$MAIN_PROCESS_PORT" \
SMOKE=0 \
bash "$TRAIN_SCRIPT"

echo "Training, checkpoint verification, valid-A evaluation, and calibration completed."
echo "Starting deterministic calibrated submission inference: RUN_ID=$RUN_ID"
PYTHON_BIN="$PYTHON_BIN" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
RUN_ID="$RUN_ID" \
MAX_SAMPLES="$SUBMISSION_MAX_SAMPLES" \
OVERWRITE="$OVERWRITE" \
bash "$SUBMISSION_SCRIPT"

SUBMISSION="$ROOT_DIR/outputs/experiments/qwen35_9b_stage_pair_v2_text_anchor/$RUN_ID/submission/submission.csv"
PROFILE="$ROOT_DIR/outputs/experiments/qwen35_9b_stage_pair_v2_text_anchor/$RUN_ID/submission/inference_profile.json"
if [[ ! -s "$SUBMISSION" || ! -s "$PROFILE" ]]; then
  echo "End-to-end pipeline finished without required submission artifacts."
  exit 1
fi

echo "End-to-end E1 pipeline completed successfully."
echo "Submission: $SUBMISSION"
echo "Runtime profile: $PROFILE"
