#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

LOG_DIR="$ROOT_DIR/logs/qwen3vl_lora24"
mkdir -p "$LOG_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CMD=("$PYTHON_BIN" -m snu_order.qwen3vl.evaluate_lockbox
  --config configs/exp/qwen3vl_8b_lora24.yaml
  --checkpoint weights/qwen3vl_lora24/best
  "$@")
echo "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/valid_b_lockbox.log"
