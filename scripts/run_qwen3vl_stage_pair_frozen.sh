#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-base}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

LOG_DIR="$ROOT_DIR/logs/qwen3vl_stage_pair"
mkdir -p "$LOG_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.train_stage_pair
  --config configs/exp/qwen3vl_stage_pair_frozen.yaml
  --mode frozen_stage_pair
  --source "$SOURCE")
echo "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/frozen_stage_pair_${SOURCE}.log"
