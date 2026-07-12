#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

LOG_DIR="$ROOT_DIR/logs/qwen3vl_lora24"
mkdir -p "$LOG_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"

bash scripts/check_qwen3vl_lora_env.sh
bash scripts/run_qwen3vl_zero_shot_diagnostics.sh
"$PYTHON_BIN" -m pytest -q 2>&1 | tee "$LOG_DIR/pytest.log"
bash scripts/run_qwen3vl_lora_overfit64.sh

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("outputs/experiments/qwen3vl_8b_lora24/overfit64_summary.json")
status = json.loads(path.read_text()).get("candidate_status")
print(f"overfit64_status={status}")
if status != "OVERFIT64_PASS":
    raise SystemExit(2)
PY

bash scripts/run_qwen3vl_frozen_probe.sh
bash scripts/run_qwen3vl_lora_subset.sh
bash scripts/run_qwen3vl_lora_full.sh

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("outputs/experiments/qwen3vl_8b_lora24/full_summary.json")
summary = json.loads(path.read_text())
print(f"full_candidate_status={summary.get('candidate_status')}")
if summary.get("candidate_status") == "READY_FOR_LOCKBOX":
    print("valid_b lockbox command:")
    print("bash scripts/run_qwen3vl_lora_valid_b_lockbox.sh --unlock-valid-b")
PY
