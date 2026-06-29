#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
CFG="${1:-configs/final.yaml}"
"${PYTHON}" -m snu_order.pipeline.inference --config "${CFG}"

