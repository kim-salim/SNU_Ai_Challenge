#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/qwen3vl_large_24candidate.yaml}"
MODEL_NAME="${MODEL_NAME:-}"
RUN_NAME="${RUN_NAME:-qwen_vlm24_test}"
IMAGE_MODE="${IMAGE_MODE:-multi_image}"
SCORING_MODE="${SCORING_MODE:-option_label_logprob}"
METADATA_CSV="${METADATA_CSV:-data/raw/test.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-data/raw}"
SAMPLE_SUBMISSION="${SAMPLE_SUBMISSION:-data/raw/sample_submission.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/predictions/${RUN_NAME}/submission.csv}"
LOG_DIR="${LOG_DIR:-logs/qwen_vlm24}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_CSV")"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/src}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CMD=(
  python3 -m snu_order.vlm24.inference
  --config "$CONFIG"
  --metadata-csv "$METADATA_CSV"
  --image-root "$IMAGE_ROOT"
  --sample-submission "$SAMPLE_SUBMISSION"
  --output-csv "$OUTPUT_CSV"
  --image-mode "$IMAGE_MODE"
  --scoring-mode "$SCORING_MODE"
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
