#!/usr/bin/env bash
set -euo pipefail

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${QWEN35_27B_CUDA_DEVICE:-0}")"
case "${GPU_NAME}" in *RTX*3090*) ;; *) echo "TRAIN_READY_PROVISIONAL_3090_PENDING: ${GPU_NAME}" >&2; exit 3;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OUTPUT="${OUTPUT:-${ROOT}/artifacts/qwen35_27b_int4_port/06_budget/benchmark_3090.json}"
"${ROOT}/scripts/qwen35_27b_int4/smoke_synthetic.sh"

