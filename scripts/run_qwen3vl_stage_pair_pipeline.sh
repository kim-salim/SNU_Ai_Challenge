#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="$ROOT_DIR/logs/qwen3vl_stage_pair"
mkdir -p "$LOG_DIR"

{
  "$PYTHON_BIN" -m pytest -q tests/test_stage_assignment.py tests/test_stage_pair_scorer.py tests/test_stage_pair_equivariance.py tests/test_stage_pair_dataset.py tests/test_stage_pair_checkpoint.py
  bash scripts/run_qwen3vl_stage_cache.sh base
  bash scripts/run_qwen3vl_stage_cache.sh existing_lora
  bash scripts/run_qwen3vl_stage_frozen.sh base
  bash scripts/run_qwen3vl_stage_frozen.sh existing_lora
  bash scripts/run_qwen3vl_stage_set_frozen.sh base
  bash scripts/run_qwen3vl_stage_set_frozen.sh existing_lora
  bash scripts/run_qwen3vl_stage_pair_frozen.sh base
  bash scripts/run_qwen3vl_stage_pair_frozen.sh existing_lora
  echo "Frozen probes complete. Review outputs/experiments/qwen3vl_stage_pair before running full QLoRA."
  echo "Full QLoRA command, only if frozen result justifies it:"
  echo "bash scripts/run_qwen3vl_stage_pair_qlora_full.sh"
} 2>&1 | tee "$LOG_DIR/pipeline.log"
