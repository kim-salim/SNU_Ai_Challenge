#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/a/snu-ai-frame-ordering/.conda-stage2/bin/python}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${METADATA_CSV:?METADATA_CSV is required}"
: "${IMAGE_ROOT:?IMAGE_ROOT is required}"
: "${OUTPUT_CSV:?OUTPUT_CSV is required}"
: "${SAMPLE_SUBMISSION:?SAMPLE_SUBMISSION is required}"
case "${METADATA_CSV,,}" in *test*) test "${ALLOW_OFFICIAL_TEST_INFERENCE:-0}" = "1" || { echo "Official Test inference requires explicit ALLOW_OFFICIAL_TEST_INFERENCE=1" >&2; exit 2; };; esac
export PYTHONPATH="${ROOT}/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
MODEL_ARGS=()
if [[ -n "${QWEN35_27B_BASE_PATH:-}" ]]; then MODEL_ARGS+=(--base-model-path "${QWEN35_27B_BASE_PATH}"); fi
if [[ -n "${QWEN35_27B_REVISION:-}" ]]; then MODEL_ARGS+=(--base-model-revision "${QWEN35_27B_REVISION}"); fi
"${PYTHON_BIN}" -m snu_order.qwen3vl.inference_stage_pair \
  --config "${CONFIG:-${ROOT}/configs/exp/qwen35_27b_stage_pair_e1_int4_champion_port_v1.yaml}" \
  --checkpoint "${CHECKPOINT}" --metadata-csv "${METADATA_CSV}" --image-root "${IMAGE_ROOT}" \
  --sample-submission "${SAMPLE_SUBMISSION}" --output-csv "${OUTPUT_CSV}" \
  --calibration "${CALIBRATION:?CALIBRATION is required}" \
  --valid-split-binding "${VALID_SPLIT_BINDING:-/home/a/snu-ai-frame-ordering/data/splits/full_train_90_10_v1/valid_10_v1.csv}" \
  "${MODEL_ARGS[@]}"
