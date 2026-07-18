#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/a/snu-ai-frame-ordering/.conda-stage2/bin/python}"
: "${RAW_LOGITS:?RAW_LOGITS from frozen 27B checkpoint evaluation is required}"
: "${CHECKPOINT:?CHECKPOINT is required for calibration binding}"
case "${RAW_LOGITS,,}" in *test*|*valid_b*) echo "Calibration input must be Valid-A only" >&2; exit 2;; esac
VALID_SPLIT="${VALID_SPLIT:-/home/a/snu-ai-frame-ordering/data/splits/full_train_90_10_v1/valid_10_v1.csv}"
export PYTHONPATH="${ROOT}/src"
"${PYTHON_BIN}" -m snu_order.qwen3vl.calibration_stage_pair --raw-logits "${RAW_LOGITS}" \
  --output-dir "${OUTPUT_DIR:-${ROOT}/outputs/calibration/qwen35_27b_int4_$(date -u +%Y%m%dT%H%M%SZ)}" \
  --config "${CONFIG:-${ROOT}/configs/exp/qwen35_27b_stage_pair_e1_int4_champion_port_v1.yaml}" \
  --binding "checkpoint_manifest_sha256=${CHECKPOINT}/checkpoint_manifest.json" \
  --binding "adapter_sha256=${CHECKPOINT}/adapter/adapter_model.safetensors" \
  --binding "heads_sha256=${CHECKPOINT}/heads.pt" \
  --binding "prompt_fingerprint_sha256=${CHECKPOINT}/prompt_fingerprint.json" \
  --binding "processor_fingerprint_sha256=${CHECKPOINT}/processor/tokenizer_config.json" \
  --binding "permutation_mapping_sha256=${CHECKPOINT}/permutations.json" \
  --binding "validation_split_sha256=${VALID_SPLIT}" \
  --binding "scorer_code_sha256=${ROOT}/src/snu_order/qwen3vl/stage_pair_scorer.py"
