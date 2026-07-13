#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/.cache/torchinductor}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-qwen35_9b_stage_pair_v2_text_anchor}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
EPOCHS="${EPOCHS:-4}"
RUN_LOCKBOX="${RUN_LOCKBOX:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29571}"
SMOKE="${SMOKE:-0}"
CONFIG="configs/exp/qwen35_9b_stage_pair_v2_text_anchor.yaml"

if [[ "$SMOKE" == "1" ]]; then
  RUN_ID="${RUN_ID}_smoke"
  MAX_SAMPLES=8
  MAX_VALID_SAMPLES=8
  EPOCHS=1
else
  MAX_VALID_SAMPLES=-1
fi

OUTPUT_DIR="$ROOT_DIR/outputs/experiments/qwen35_9b_stage_pair_v2_text_anchor/$RUN_ID"
CHECKPOINT_ROOT="$ROOT_DIR/weights/qwen35_9b_stage_pair_v2_text_anchor/$RUN_ID"
BEST_CHECKPOINT="$CHECKPOINT_ROOT/best"
LOG_DIR="$ROOT_DIR/logs/qwen35_9b_stage_pair_v2_text_anchor"
VALID_A_DIR="$OUTPUT_DIR/valid_a_best"
mkdir -p "$LOG_DIR"

if [[ "$OVERWRITE" != "1" ]] && { [[ -e "$OUTPUT_DIR" ]] || [[ -e "$CHECKPOINT_ROOT" ]]; }; then
  echo "Refusing to reuse an existing v2 run: $RUN_ID"
  echo "Choose a new RUN_ID or set OVERWRITE=1 explicitly."
  exit 1
fi

TRAIN_CMD=("$PYTHON_BIN" -m accelerate.commands.launch
  --num_processes 1
  --num_machines 1
  --main_process_port "$MAIN_PROCESS_PORT"
  --mixed_precision bf16
  -m snu_order.qwen3vl.train_stage_pair
  --config "$CONFIG"
  --mode qlora_stage_pair
  --source base
  --run-id "$RUN_ID"
  --max-samples "$MAX_SAMPLES"
  --max-valid-samples "$MAX_VALID_SAMPLES"
  --epochs "$EPOCHS")

echo "${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_ID}_train.log"

VERIFY_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.verify_stage_pair_checkpoint
  --config "$CONFIG"
  --checkpoint "$BEST_CHECKPOINT"
  --metadata-csv data/splits/ab_v1/valid_a_v1.csv
  --image-root data/raw
  --max-samples 8
  --output-json "$OUTPUT_DIR/checkpoint_verification.json")
echo "${VERIFY_CMD[*]}"
"${VERIFY_CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_ID}_verify.log"

EVAL_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.evaluate_stage_pair
  --config "$CONFIG"
  --checkpoint "$BEST_CHECKPOINT"
  --metadata-csv data/splits/ab_v1/valid_a_v1.csv
  --image-root data/raw
  --output-dir "$VALID_A_DIR"
  --max-samples "$MAX_VALID_SAMPLES"
  --split-name valid_a)
echo "${EVAL_CMD[*]}"
"${EVAL_CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_ID}_valid_a.log"

CALIBRATION_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.calibration_stage_pair
  --config "$CONFIG"
  --raw-logits "$VALID_A_DIR/raw_stage_pair_logits.pt"
  --output-dir "$BEST_CHECKPOINT"
  --tune-split valid_a)
echo "${CALIBRATION_CMD[*]}"
"${CALIBRATION_CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_ID}_calibration.log"

if [[ "$RUN_LOCKBOX" == "1" ]]; then
  LOCKBOX_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.evaluate_stage_pair
    --config "$CONFIG"
    --checkpoint "$BEST_CHECKPOINT"
    --metadata-csv data/splits/ab_v1/valid_b_v1.csv
    --image-root data/raw
    --output-dir "$OUTPUT_DIR/valid_b_once"
    --max-samples -1
    --split-name valid_b
    --calibration "$BEST_CHECKPOINT/calibration.json")
  echo "${LOCKBOX_CMD[*]}"
  "${LOCKBOX_CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_ID}_valid_b_once.log"
fi

echo "Best checkpoint: $BEST_CHECKPOINT"
echo "Valid-A raw logits: $VALID_A_DIR/raw_stage_pair_logits.pt"
echo "Calibration: $BEST_CHECKPOINT/calibration.json"
