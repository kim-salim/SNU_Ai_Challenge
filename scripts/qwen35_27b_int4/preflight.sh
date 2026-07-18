#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${ROOT}/artifacts/qwen35_27b_int4_port}"
CHAMPION="${CHAMPION_ROOT:-/home/a/snu-ai-frame-ordering/.worktrees/e1-canonical-84p2-release}"
PYTHON_BIN="${PYTHON_BIN:-/home/a/snu-ai-frame-ordering/.conda-stage2/bin/python}"
export PYTHONPATH="${ROOT}/src"
export PYTHONPYCACHEPREFIX="/tmp/qwen35_27b_port_pycache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
mkdir -p "${ARTIFACT_ROOT}/00_preflight" "${ARTIFACT_ROOT}/03_tests"
test "$(git -C "${CHAMPION}" rev-parse HEAD)" = "003c035c5335355f03a45cbf01af45866c1af85e"
test "$(git -C "${CHAMPION}" rev-parse 'HEAD^{tree}')" = "4cff067f0bc39ddd06cf41b68e5c566deb9dbf62"
test -z "$(git -C "${CHAMPION}" status --porcelain=v1 --untracked-files=all)"
(
  cd "${CHAMPION}/model_artifacts/qwen35_stage_e1_canonical_84p2"
  sha256sum -c checksums.sha256
)
"${PYTHON_BIN}" -m snu_order.qwen3vl.audit_train_manifest_90_10 \
  --manifest /home/a/snu-ai-frame-ordering/data/splits/full_train_90_10_v1/split_manifest.json \
  --output "${ARTIFACT_ROOT}/00_preflight/SPLIT_90_10_AUDIT.json"
"${PYTHON_BIN}" -m pytest -q \
  "${ROOT}/tests/qwen35_27b_int4" \
  "${ROOT}/tests/test_stage_pair_lora_targets_v2.py" \
  "${ROOT}/tests/test_stage_pair_checkpoint_v2.py" \
  "${ROOT}/tests/test_stage_pair_anchor_v2.py" \
  "${ROOT}/tests/test_stage_pair_equivariance.py" \
  "${ROOT}/tests/test_stage_pair_scorer.py" \
  | tee "${ARTIFACT_ROOT}/03_tests/preflight_pytest.txt"
