#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?CHECKPOINT is required}"
: "${CALIBRATION:?CALIBRATION is required}"
: "${PROCESSOR_DIR:?PROCESSOR_DIR is required}"
: "${OUTPUT_TAR:?OUTPUT_TAR is required}"
test -d "${CHECKPOINT}"
test -f "${CALIBRATION}"
test -d "${PROCESSOR_DIR}"
find "${CHECKPOINT}" -type f -name 'adapter_model.*' | grep -q .
tar -czf "${OUTPUT_TAR}" "${CHECKPOINT}" "${CALIBRATION}" "${PROCESSOR_DIR}"
SIZE_BYTES="$(stat -c %s "${OUTPUT_TAR}")"
test "${SIZE_BYTES}" -le 80530636800 || { echo "Package exceeds 75 GiB target/80 GB limit" >&2; exit 2; }
sha256sum "${OUTPUT_TAR}" > "${OUTPUT_TAR}.sha256"

