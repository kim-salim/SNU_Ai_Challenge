#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger_full_20260714}"
SMOKE="${SMOKE:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29573}"
BASE_CONFIG="configs/exp/qwen35_9b_stage_pair_v2_text_anchor.yaml"
E2_CONFIG="configs/exp/qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger.yaml"
OUTPUT_DIR="$ROOT_DIR/outputs/experiments/qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger/$RUN_ID"
CHECKPOINT_ROOT="$ROOT_DIR/weights/qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger/$RUN_ID"
BEST_CHECKPOINT="$CHECKPOINT_ROOT/best"
HARDENING_DIR="$ROOT_DIR/outputs/hardening_20260714/p3_e2_vision_merger/$RUN_ID"
LOG_DIR="$HARDENING_DIR/logs"
VALID_A_DIR="$HARDENING_DIR/valid_a_raw"
mkdir -p "$LOG_DIR"

if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "E2 requires WORLD_SIZE=1, got ${WORLD_SIZE:-unset}" >&2
  exit 1
fi
if [[ "$OVERWRITE" != "1" ]] && { [[ -e "$OUTPUT_DIR" ]] || [[ -e "$CHECKPOINT_ROOT" ]]; }; then
  echo "Refusing to reuse existing E2 output/checkpoint paths: $RUN_ID" >&2
  exit 1
fi

DIFF_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.compare_experiment_configs
  --base "$BASE_CONFIG"
  --candidate "$E2_CONFIG"
  --allow-path experiment.id
  --allow-path experiment.run_id
  --allow-path output.dir
  --allow-path output.checkpoint_dir
  --allow-path vision_merger_lora.enabled)
printf '%q ' "${DIFF_CMD[@]}"
printf '\n'
"${DIFF_CMD[@]}" | tee "$HARDENING_DIR/config_semantic_diff.json"

if [[ "$SMOKE" == "1" ]]; then
  TRAIN_SAMPLES=16
  VALID_SAMPLES=16
  EPOCHS=1
else
  TRAIN_SAMPLES=-1
  VALID_SAMPLES=-1
  EPOCHS=4
fi

export GRADIENT_HEALTH_OUTPUT="$HARDENING_DIR/gradient_health.json"
TRAIN_CMD=("$PYTHON_BIN" -m accelerate.commands.launch
  --num_processes 1
  --num_machines 1
  --main_process_port "$MAIN_PROCESS_PORT"
  --mixed_precision bf16
  -m snu_order.qwen3vl.train_stage_pair
  --config "$E2_CONFIG"
  --mode qlora_stage_pair
  --source base
  --run-id "$RUN_ID"
  --max-samples "$TRAIN_SAMPLES"
  --max-valid-samples "$VALID_SAMPLES"
  --epochs "$EPOCHS")
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_DIR/train.log"

for verification_run in 1 2; do
  VERIFY_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.verify_stage_pair_checkpoint
    --config "$E2_CONFIG"
    --checkpoint "$BEST_CHECKPOINT"
    --metadata-csv data/splits/ab_v1/valid_a_v1.csv
    --image-root data/raw
    --max-samples 8
    --output-json "$HARDENING_DIR/checkpoint_verification_${verification_run}.json")
  printf '%q ' "${VERIFY_CMD[@]}"
  printf '\n'
  "${VERIFY_CMD[@]}" 2>&1 | tee "$LOG_DIR/verify_${verification_run}.log"
done

EVAL_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.evaluate_stage_pair
  --config "$E2_CONFIG"
  --checkpoint "$BEST_CHECKPOINT"
  --metadata-csv data/splits/ab_v1/valid_a_v1.csv
  --image-root data/raw
  --output-dir "$VALID_A_DIR"
  --max-samples "$VALID_SAMPLES"
  --split-name valid_a)
printf '%q ' "${EVAL_CMD[@]}"
printf '\n'
"${EVAL_CMD[@]}" 2>&1 | tee "$LOG_DIR/valid_a_raw.log"

CALIBRATION_DIR="$HARDENING_DIR/calibration"
CALIBRATION_CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.calibration_stage_pair
  --config "$E2_CONFIG"
  --raw-logits "$VALID_A_DIR/raw_stage_pair_logits.pt"
  --output-dir "$CALIBRATION_DIR"
  --tune-split valid_a
  --fold-count 5
  --fixed-pair-weight 1.0
  --fixed-stage-temperature 1.5
  --fixed-pair-temperature 0.8
  --binding "checkpoint_manifest_sha256=$BEST_CHECKPOINT/checkpoint_manifest.json"
  --binding "adapter_sha256=$BEST_CHECKPOINT/adapter/adapter_model.safetensors"
  --binding "heads_sha256=$BEST_CHECKPOINT/heads.pt"
  --binding "prompt_fingerprint_sha256=$BEST_CHECKPOINT/prompt_fingerprint.json"
  --binding "processor_fingerprint_sha256=$BEST_CHECKPOINT/processor/tokenizer_config.json"
  --binding "permutation_mapping_sha256=$BEST_CHECKPOINT/permutations.json"
  --binding "validation_split_sha256=data/splits/ab_v1/valid_a_v1.csv"
  --binding "scorer_code_sha256=src/snu_order/qwen3vl/stage_pair_scorer.py")
printf '%q ' "${CALIBRATION_CMD[@]}"
printf '\n'
"${CALIBRATION_CMD[@]}" 2>&1 | tee "$LOG_DIR/calibration.log"

echo "E2 checkpoint: $BEST_CHECKPOINT"
echo "E2 raw logits: $VALID_A_DIR/raw_stage_pair_logits.pt"
echo "E2 calibration: $CALIBRATION_DIR/calibration.json"
echo "E2 gradient health: $GRADIENT_HEALTH_OUTPUT"
