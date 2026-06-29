#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
FILE="${1:-outputs/submissions/final_submission.csv}"
REFERENCE="${2:-data/raw/sample_submission.csv}"
"${PYTHON}" -m snu_order.data.validate_submission --file "${FILE}" --reference "${REFERENCE}"

