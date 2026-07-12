#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/.cache/torchinductor}"

LOG_DIR="$ROOT_DIR/logs/qwen3vl_stage_pair"
mkdir -p "$LOG_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
SOURCE="${SOURCE:-base}"
RUN_ID="${RUN_ID:-qlora_stage_pair_fixed_ddp}"

CMD=("$PYTHON_BIN" -m accelerate.commands.launch
  --num_processes "$NUM_PROCESSES"
  --num_machines 1
  --main_process_port "$MAIN_PROCESS_PORT"
  --mixed_precision bf16
  -m snu_order.qwen3vl.train_stage_pair
  --config configs/exp/qwen3vl_stage_pair_qlora.yaml
  --mode qlora_stage_pair
  --source "$SOURCE"
  --run-id "$RUN_ID")
echo "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/${RUN_ID}_${SOURCE}.log"
