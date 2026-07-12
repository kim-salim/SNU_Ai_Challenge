#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs outputs/predictions/qwen25vl_7b_24candidate/valid_b_300

PYTHON_BIN="${PYTHON_BIN:-.conda-stage2/bin/python}"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"

CMD=(
  "$PYTHON_BIN" -m snu_order.vlm24.eval
  --config configs/qwen25vl_7b_24candidate.yaml
  --metadata-csv data/splits/ab_v1/valid_b_v1.csv
  --image-root data/raw
  --output-dir outputs/predictions/qwen25vl_7b_24candidate/valid_b_300
  --max-samples 300
  --image-mode multi_image
  --scoring-mode option_label_logprob
  --benchmark
)

echo "${CMD[@]}"
"${CMD[@]}" 2>&1 | tee logs/qwen25vl7b_valid_b_300.log
