#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/a/snu-ai-frame-ordering/.conda-stage2/bin/python}"
MANIFEST="${SPLIT_MANIFEST:-/home/a/snu-ai-frame-ordering/data/splits/full_train_90_10_v1/split_manifest.json}"
OUTPUT="${OUTPUT:-${ROOT}/artifacts/qwen35_27b_int4_port/00_preflight/SPLIT_90_10_AUDIT.json}"
case "${MANIFEST,,}" in *test*|*sample_submission*) echo "Official Test path rejected" >&2; exit 2;; esac
export PYTHONPATH="${ROOT}/src"
mkdir -p "$(dirname "${OUTPUT}")"
"${PYTHON_BIN}" -m snu_order.qwen3vl.audit_train_manifest_90_10 --manifest "${MANIFEST}" --output "${OUTPUT}"

