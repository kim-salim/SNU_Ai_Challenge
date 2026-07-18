#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/a/snu-ai-frame-ordering/.conda-stage2/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/exp/qwen35_27b_stage_pair_e1_int4_champion_port_v1.yaml}"
DATA_ROOT="${SNU_DATA_ROOT:-/home/a/snu-ai-frame-ordering}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/outputs/features/qwen35_27b_stage_pair_e1_int4_v1/base}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/data/splits/full_train_90_10_v1/train_90_v1.csv}"
VALID_CSV="${VALID_CSV:-${DATA_ROOT}/data/splits/full_train_90_10_v1/valid_10_v1.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_ROOT}/data/raw}"
case "${TRAIN_CSV,,}:${VALID_CSV,,}" in *test*|*sample_submission*) echo "Official Test path rejected" >&2; exit 2;; esac
export PYTHONPATH="${ROOT}/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
mkdir -p "${CACHE_DIR}"
MODEL_ARGS=()
if [[ -n "${QWEN35_27B_BASE_PATH:-}" ]]; then MODEL_ARGS+=(--base-model-path "${QWEN35_27B_BASE_PATH}"); fi
if [[ -n "${QWEN35_27B_REVISION:-}" ]]; then MODEL_ARGS+=(--base-model-revision "${QWEN35_27B_REVISION}"); fi
"${PYTHON_BIN}" -m snu_order.qwen3vl.extract_stage_pair_features --config "${CONFIG}" \
  --metadata-csv "${TRAIN_CSV}" --image-root "${IMAGE_ROOT}" --split-name train \
  --output "${CACHE_DIR}/train_90.pt" --max-samples "${MAX_TRAIN_SAMPLES:--1}" "${MODEL_ARGS[@]}"
"${PYTHON_BIN}" -m snu_order.qwen3vl.extract_stage_pair_features --config "${CONFIG}" \
  --metadata-csv "${VALID_CSV}" --image-root "${IMAGE_ROOT}" --split-name valid_a \
  --output "${CACHE_DIR}/valid_10.pt" --max-samples "${MAX_VALID_SAMPLES:--1}" "${MODEL_ARGS[@]}"
