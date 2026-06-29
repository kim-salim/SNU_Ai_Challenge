#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
CFG="${1:-configs/exp/exp003_siglip2_quality_pair_aux.yaml}"
"${PYTHON}" -m snu_order.features.extract_siglip2 --config "${CFG}" --split train
"${PYTHON}" -m snu_order.features.extract_siglip2 --config "${CFG}" --split valid

