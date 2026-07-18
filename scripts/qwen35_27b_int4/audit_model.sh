#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/a/snu-ai-frame-ordering/.conda-stage2/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/exp/qwen35_27b_stage_pair_e1_int4_champion_port_v1.yaml}"
OUTPUT="${OUTPUT:-${ROOT}/artifacts/qwen35_27b_int4_port/04_model_audit/model_audit.json}"
export PYTHONPATH="${ROOT}/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
mkdir -p "$(dirname "${OUTPUT}")"
ARGS=(--config "${CONFIG}" --output "${OUTPUT}")
if [[ -n "${QWEN35_27B_BASE_PATH:-}" ]]; then ARGS+=(--base-path "${QWEN35_27B_BASE_PATH}"); fi
if [[ -n "${QWEN35_27B_REVISION:-}" ]]; then ARGS+=(--revision "${QWEN35_27B_REVISION}"); fi
"${PYTHON_BIN}" -m snu_order.qwen3vl.audit_qwen35_27b_port "${ARGS[@]}"
