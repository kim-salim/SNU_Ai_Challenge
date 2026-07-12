#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-base}"
if [[ "$SOURCE" != "base" && "$SOURCE" != "existing_lora" ]]; then
  echo "usage: $0 base|existing_lora" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/.cache/torchinductor}"

LOG_DIR="$ROOT_DIR/logs/qwen3vl_stage_pair"
mkdir -p "$LOG_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CFG="${CFG:-configs/exp/qwen3vl_stage_pair_qlora.yaml}"

run_cache() {
  local split="$1"
  local out="outputs/features/qwen3vl_stage_pair/$SOURCE/${split}.pt"
  if [[ "$split" == "train_ab" ]]; then
    out="outputs/features/qwen3vl_stage_pair/$SOURCE/train_ab.pt"
  else
    out="outputs/features/qwen3vl_stage_pair/$SOURCE/valid_a.pt"
  fi
  if [[ -f "$out" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "cache exists, skip: $out"
    return
  fi
  local cmd=("$PYTHON_BIN" -m snu_order.qwen3vl.cache_frame_features --config "$CFG" --source "$SOURCE" --split "$split" --output "$out")
  echo "${cmd[*]}"
  "${cmd[@]}"
}

{
  run_cache train_ab
  run_cache valid_a
} 2>&1 | tee "$LOG_DIR/cache_${SOURCE}.log"
