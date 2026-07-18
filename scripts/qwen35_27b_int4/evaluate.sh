#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/a/snu-ai-frame-ordering/.conda-stage2/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/exp/qwen35_27b_stage_pair_e1_int4_champion_port_v1.yaml}"
DATA_ROOT="${SNU_DATA_ROOT:-/home/a/snu-ai-frame-ordering}"
METADATA_CSV="${METADATA_CSV:-${DATA_ROOT}/data/splits/full_train_90_10_v1/valid_10_v1.csv}"
case "${METADATA_CSV,,}" in *test*|*sample_submission*) echo "Official Test path rejected" >&2; exit 2;; esac
: "${CHECKPOINT:?CHECKPOINT is required}"
export PYTHONPATH="${ROOT}/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
MODEL_ARGS=()
if [[ -n "${QWEN35_27B_BASE_PATH:-}" ]]; then MODEL_ARGS+=(--base-model-path "${QWEN35_27B_BASE_PATH}"); fi
if [[ -n "${QWEN35_27B_REVISION:-}" ]]; then MODEL_ARGS+=(--base-model-revision "${QWEN35_27B_REVISION}"); fi
"${PYTHON_BIN}" -m snu_order.qwen3vl.evaluate_stage_pair --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
  --metadata-csv "${METADATA_CSV}" --image-root "${DATA_ROOT}/data/raw" --split-name valid_a \
  --output-dir "${OUTPUT_DIR:-${ROOT}/outputs/evaluation/qwen35_27b_int4_$(date -u +%Y%m%dT%H%M%SZ)}" \
  "${MODEL_ARGS[@]}"
