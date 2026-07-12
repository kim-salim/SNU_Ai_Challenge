#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/qwen3vl_large_24candidate.yaml}"
MODEL_NAME="${MODEL_NAME:-}"
RUN_NAME="${RUN_NAME:-qwen_vlm24_valid_a_full}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
IMAGE_MODE="${IMAGE_MODE:-multi_image}"
SCORING_MODE="${SCORING_MODE:-option_label_logprob}"
METADATA_CSV="${METADATA_CSV:-data/splits/ab_v1/valid_a_v1.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-data/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/predictions/${RUN_NAME}/valid_a_full}"
LOG_DIR="${LOG_DIR:-logs/qwen_vlm24}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/src}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CMD=(
  python3 -m snu_order.vlm24.eval
  --config "$CONFIG"
  --metadata-csv "$METADATA_CSV"
  --image-root "$IMAGE_ROOT"
  --output-dir "$OUTPUT_DIR"
  --max-samples "$MAX_SAMPLES"
  --image-mode "$IMAGE_MODE"
  --scoring-mode "$SCORING_MODE"
  --benchmark
)

if [[ -n "$MODEL_NAME" ]]; then
  CMD+=(--model-name "$MODEL_NAME")
fi
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  CMD+=(--local-files-only)
fi

printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_NAME}.log"
