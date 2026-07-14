#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CONTAINER_RUNNER="scripts/run_qwen35_hardening_exact_container.sh"
E2_RUNNER="scripts/run_qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger.sh"
P2_SELECTION="outputs/hardening_20260714/p2_chunked_inference/selected_chunk_config.json"
SMOKE_RUN_ID="qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger_smoke_20260714"
FULL_RUN_ID="qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger_full_20260714"
SMOKE_ROOT="outputs/hardening_20260714/p3_e2_vision_merger/$SMOKE_RUN_ID"
FULL_ROOT="outputs/hardening_20260714/p3_e2_vision_merger/$FULL_RUN_ID"

mkdir -p outputs/hardening_20260714/p3_e2_vision_merger
until [[ -s "$P2_SELECTION" ]]; do
  echo "Waiting for P2 certification: $P2_SELECTION"
  sleep 30
done
P2_STATUS="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] in {"PASS", "FAIL"}; print(d["status"])' "$P2_SELECTION")"
echo "P2 terminal status: $P2_STATUS"
if [[ "$P2_STATUS" == "FAIL" ]]; then
  echo "P2 chunking was rejected; continuing independent E2 with the unchunked four-frame path."
fi

if [[ -e "$SMOKE_ROOT" ]]; then
  echo "Refusing to overwrite existing E2 smoke artifacts: $SMOKE_ROOT" >&2
  exit 1
fi
bash "$CONTAINER_RUNNER" /usr/bin/env \
  PYTHON_BIN=/opt/venv/bin/python3 \
  RUN_ID="$SMOKE_RUN_ID" \
  SMOKE=1 \
  OVERWRITE=0 \
  MAIN_PROCESS_PORT=29573 \
  bash "$E2_RUNNER" \
  2>&1 | tee outputs/hardening_20260714/p3_e2_vision_merger/smoke_pipeline.log

python3 - "$SMOKE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
health = json.loads((root / "gradient_health.json").read_text())
if health.get("status") != "PASS" or health.get("captured_completed_optimizer_steps") != 2:
    raise SystemExit(f"E2 smoke gradient health failed: {health}")
verifications = [
    json.loads((root / f"checkpoint_verification_{index}.json").read_text())
    for index in (1, 2)
]
if any(value.get("status") != "ok" or not value.get("finite_logits") for value in verifications):
    raise SystemExit(f"E2 smoke checkpoint verification failed: {verifications}")
if verifications[0]["prediction_indices"] != verifications[1]["prediction_indices"]:
    raise SystemExit("E2 smoke fresh-process predictions differ")
PY

if [[ -e "$FULL_ROOT" ]]; then
  echo "Refusing to overwrite existing E2 full artifacts: $FULL_ROOT" >&2
  exit 1
fi
bash "$CONTAINER_RUNNER" /usr/bin/env \
  PYTHON_BIN=/opt/venv/bin/python3 \
  RUN_ID="$FULL_RUN_ID" \
  SMOKE=0 \
  OVERWRITE=0 \
  MAIN_PROCESS_PORT=29574 \
  bash "$E2_RUNNER" \
  2>&1 | tee outputs/hardening_20260714/p3_e2_vision_merger/full_pipeline.log

echo "P2 certification, E2 smoke, and E2 full training/evaluation/calibration completed."
