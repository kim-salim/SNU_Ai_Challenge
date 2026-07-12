#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export PYTHONPATH="$ROOT_DIR/src"

CMD=(
  python3 -m snu_order.vlm24.eval
  --config configs/qwen3vl_8b_24candidate.yaml
  --local-files-only
  --metadata-csv data/splits/ab_v1/valid_b_v1.csv
  --image-root data/raw
  --output-dir outputs/predictions/qwen3vl_8b_24candidate/valid_b_300
  --max-samples 300
  --image-mode multi_image
  --scoring-mode option_label_logprob
  --benchmark
)

printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee logs/qwen3vl8b_valid_b_300.log
