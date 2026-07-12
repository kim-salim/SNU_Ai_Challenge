#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs outputs/predictions/qwen25vl_7b_24candidate

PYTHON_BIN="${PYTHON_BIN:-.conda-stage2/bin/python}"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"

CMD=(
  "$PYTHON_BIN" -m snu_order.vlm24.inference
  --config configs/qwen25vl_7b_24candidate.yaml
  --metadata-csv data/raw/test.csv
  --image-root data/raw
  --sample-submission data/raw/sample_submission.csv
  --output-csv outputs/predictions/qwen25vl_7b_24candidate/submission.csv
  --image-mode multi_image
  --scoring-mode option_label_logprob
)

echo "${CMD[@]}"
"${CMD[@]}" 2>&1 | tee logs/qwen25vl7b_test.log
