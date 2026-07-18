#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/a/snu-ai-frame-ordering/.conda-stage2/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/exp/qwen35_27b_stage_pair_e1_int4_champion_port_v1.yaml}"
CHAMPION="${CHAMPION_ROOT:-/home/a/snu-ai-frame-ordering/.worktrees/e1-canonical-84p2-release/model_artifacts/qwen35_stage_e1_canonical_84p2/checkpoint}"
MIGRATED="${MIGRATED_HEADS:-${ROOT}/artifacts/qwen35_27b_int4_port/02_implementation/migrated_heads.pt}"
REPORT="${ROOT}/artifacts/qwen35_27b_int4_port/02_implementation/migration_report.json"
case "${CONFIG,,}" in *test*) echo "Official Test path rejected" >&2; exit 2;; esac
export PYTHONPATH="${ROOT}/src"
mkdir -p "$(dirname "${MIGRATED}")"
"${PYTHON_BIN}" "${ROOT}/scripts/qwen35_27b_int4/migrate_champion_heads.py" \
  --source-checkpoint "${CHAMPION}" --config "${CONFIG}" --output "${MIGRATED}" --report "${REPORT}"
"${PYTHON_BIN}" -m snu_order.qwen3vl.train_stage_pair --config "${CONFIG}" \
  --mode frozen_stage_pair --init-head-from "${MIGRATED}" --epochs "${PROBE_EPOCHS:-1}" \
  --max-samples "${PROBE_TRAIN_SAMPLES:-256}" --max-valid-samples "${PROBE_VALID_SAMPLES:-128}" \
  --run-id "qwen35_27b_int4_frozen_probe_$(date -u +%Y%m%dT%H%M%SZ)"

