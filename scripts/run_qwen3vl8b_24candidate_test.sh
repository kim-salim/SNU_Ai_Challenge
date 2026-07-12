#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs outputs/predictions/qwen3vl_8b_24candidate

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export PYTHONPATH="$ROOT_DIR/src"

CMD=(
  python3 -m snu_order.vlm24.inference
  --config configs/qwen3vl_8b_24candidate.yaml
  --local-files-only
  --metadata-csv data/raw/test.csv
  --image-root data/raw
  --sample-submission data/raw/sample_submission.csv
  --output-csv outputs/predictions/qwen3vl_8b_24candidate/submission.csv
  --image-mode multi_image
  --scoring-mode option_label_logprob
)

printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee logs/qwen3vl8b_test.log
