#!/usr/bin/env bash
set -euo pipefail

ROOT="${RETENTION_V3_ROOT:-/home/shpark/snu-ai-challenge/repositories/qwen35-27b-retention-v3-component-safe}"
V1_ROOT="/home/shpark/snu-ai-challenge/repositories/qwen35-27b-int4-stage-e1-v1"
IMAGE="${QWEN27_IMAGE:-snu-four-slot-teacher:v1-tmpfix}"
MODEL_REVISION="fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_PATH="/home/shpark/snu-ai-challenge/.hf-cache/hub/models--Qwen--Qwen3.5-27B/snapshots/$MODEL_REVISION"
DATA_ROOT="/home/shpark/snu-ai-challenge/data/raw"
VALID_CSV="/home/shpark/snu-ai-challenge/data/splits/full_train_90_10_v1/valid_10_v1.csv"
P0="$ROOT/outputs/retention_v3/p0/v1_best_parity.json"
P0_REPEAT="$ROOT/outputs/retention_v3/p0/v1_fresh_repeatability.json"
CACHE_AUDIT="$ROOT/outputs/retention_v3/cache/component_cache_audit.json"
W0_CONFIG="$ROOT/configs/exp/qwen35_27b_retention_v3_w0_shared_warmup_20260720.yaml"
C0_CONFIG="$ROOT/configs/exp/qwen35_27b_retention_v3_c0_exact_control_20260720.yaml"
K0_CONFIG="$ROOT/configs/exp/qwen35_27b_retention_v3_k0_component_safe_20260720.yaml"
W0_RUN="qwen35_27b_retention_v3_w0_shared_warmup_ddp8_20260720"
C0_RUN="qwen35_27b_retention_v3_c0_exact_control_ddp8_20260720"
K0_RUN="qwen35_27b_retention_v3_k0_component_safe_ddp8_20260720"
W0_CHECKPOINT="$ROOT/weights/retention_v3/w0/$W0_RUN/fork_10pct"
C0_RUN_ROOT="$ROOT/outputs/retention_v3/c0/$C0_RUN"
K0_RUN_ROOT="$ROOT/outputs/retention_v3/k0/$K0_RUN"
C0_CHECKPOINT="$ROOT/weights/retention_v3/c0/$C0_RUN/best"
K0_CHECKPOINT="$ROOT/weights/retention_v3/k0/$K0_RUN/best"
V1_RAW="$V1_ROOT/outputs/experiments/qwen35_27b_stage_pair_e1_int4_v1_full90_20260719/qwen35_27b_stage_pair_e1_int4_v1_full90_ddp8_20260719/post_training_submission/valid10_best_raw/raw_stage_pair_logits.pt"
FINAL_ROOT="$ROOT/outputs/retention_v3/final"

mkdir -p "$FINAL_ROOT"
CURRENT_PHASE="PRECHECK"
record_failure() {
  local exit_code="$?"
  trap - ERR
  python3 - "$FINAL_ROOT/final_decision.json" "$CURRENT_PHASE" "$exit_code" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "status": f"EXPERIMENT_STOPPED_{sys.argv[2]}",
    "failed_phase": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "production_decision": "RETAIN_QWEN35_E1_FULL90_84P642",
    "automatic_followup_started": False,
    "valid_b_accessed": False,
    "test_accessed": False,
    "submission_generated": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  exit "$exit_code"
}
trap record_failure ERR

python3 - "$P0" "$P0_REPEAT" "$CACHE_AUDIT" <<'PY'
import json,sys
p0=json.load(open(sys.argv[1]))
repeat=json.load(open(sys.argv[2]))
cache=json.load(open(sys.argv[3]))
assert p0["status"] == "P0_CANONICAL_SCORER_PARITY_PASS", p0["status"]
assert repeat["status"] == "P0_FRESH_PROCESS_REPEATABILITY_PASS", repeat["status"]
assert cache["status"] == "COMPONENT_CACHE_AUDIT_PASS", cache["status"]
PY

docker_common=(
  docker run --rm --network none --ipc=host --shm-size=64g
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
  -e PYTHONHASHSEED=42
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e HF_DATASETS_OFFLINE=1
  -e HF_HUB_DISABLE_TELEMETRY=1
  -e TOKENIZERS_PARALLELISM=false
  -e WANDB_DISABLED=true
  -e PYTHONPATH="$ROOT/src"
  -v /home/shpark/snu-ai-challenge:/home/shpark/snu-ai-challenge
  -w "$ROOT"
)

run_ddp() {
  local config="$1"; shift
  "${docker_common[@]}" --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e NCCL_DEBUG=WARN "$IMAGE" \
    /opt/venv/bin/torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port="$1" \
    -m snu_order.qwen3vl.train_stage_pair --config "$config" --mode qlora_stage_pair "${@:2}"
}

evaluate_raw() {
  local config="$1" checkpoint="$2" output="$3"
  "${docker_common[@]}" --gpus device=0 -e CUDA_VISIBLE_DEVICES=0 "$IMAGE" \
    /opt/venv/bin/python -m snu_order.qwen3vl.evaluate_stage_pair \
    --config "$config" --checkpoint "$checkpoint" --metadata-csv "$VALID_CSV" \
    --image-root "$DATA_ROOT" --output-dir "$output" --max-samples -1 --split-name valid_a \
    --frame-chunk-size 1 --base-model-path "$MODEL_PATH" --base-model-revision "$MODEL_REVISION"
}

calibrate() {
  local config="$1" checkpoint="$2" raw="$3" output="$4"
  "${docker_common[@]}" "$IMAGE" /opt/venv/bin/python -m snu_order.qwen3vl.calibration_stage_pair \
    --config "$config" --raw-logits "$raw" --output-dir "$output" --tune-split valid_a \
    --binding "checkpoint_manifest_sha256=$checkpoint/checkpoint_manifest.json" \
    --binding "adapter_sha256=$checkpoint/adapter/adapter_model.safetensors" \
    --binding "heads_sha256=$checkpoint/heads.pt" \
    --binding "prompt_fingerprint_sha256=$checkpoint/prompt_fingerprint.json" \
    --binding "processor_fingerprint_sha256=$checkpoint/processor/tokenizer_config.json" \
    --binding "permutation_mapping_sha256=$checkpoint/permutations.json" \
    --binding "validation_split_sha256=$VALID_CSV" \
    --binding "scorer_code_sha256=$ROOT/src/snu_order/qwen3vl/stage_pair_scorer.py"
}

CURRENT_PHASE="W0_SHARED_WARMUP"
echo "[$(date --iso-8601=seconds)] W0 shared 10% warmup starting"
run_ddp "$W0_CONFIG" 29630 --fork-after-head-warmup
if [[ ! -s "$W0_CHECKPOINT/training_state.pt" ]]; then
  echo "W0_FORK_MISSING" >&2
  false
fi
if [[ ! -d "${W0_CHECKPOINT}_branch_rng" ]]; then
  echo "W0_FORK_RNG_MISSING" >&2
  false
fi

CURRENT_PHASE="C0_EXACT_CONTROL"
echo "[$(date --iso-8601=seconds)] C0 exact-contract continuation starting"
run_ddp "$C0_CONFIG" 29631 --resume "$W0_CHECKPOINT" --resume-training-state
evaluate_raw "$C0_CONFIG" "$C0_CHECKPOINT" "$C0_RUN_ROOT/post_training_raw"

CURRENT_PHASE="C0_REPRODUCTION_GATE"
"${docker_common[@]}" "$IMAGE" /opt/venv/bin/python -m snu_order.qwen3vl.c0_reproduction_gate \
  --reference-raw "$V1_RAW" \
  --control-raw "$C0_RUN_ROOT/post_training_raw/raw_stage_pair_logits.pt" \
  --output "$FINAL_ROOT/c0_reproduction_gate.json"

CURRENT_PHASE="K0_COMPONENT_SAFE"
echo "[$(date --iso-8601=seconds)] K0 component-safe continuation starting"
run_ddp "$K0_CONFIG" 29632 --resume "$W0_CHECKPOINT" --resume-training-state
evaluate_raw "$K0_CONFIG" "$K0_CHECKPOINT" "$K0_RUN_ROOT/post_training_raw"

CURRENT_PHASE="FROZEN_LOGIT_CALIBRATION"
calibrate "$C0_CONFIG" "$C0_CHECKPOINT" "$C0_RUN_ROOT/post_training_raw/raw_stage_pair_logits.pt" "$C0_RUN_ROOT/calibration"
calibrate "$K0_CONFIG" "$K0_CHECKPOINT" "$K0_RUN_ROOT/post_training_raw/raw_stage_pair_logits.pt" "$K0_RUN_ROOT/calibration"

CURRENT_PHASE="PAIRED_EVALUATION"
"${docker_common[@]}" "$IMAGE" /opt/venv/bin/python -m snu_order.qwen3vl.compare_retention_v3 \
  --control-raw "$C0_RUN_ROOT/post_training_raw/raw_stage_pair_logits.pt" \
  --candidate-raw "$K0_RUN_ROOT/post_training_raw/raw_stage_pair_logits.pt" \
  --control-calibration "$C0_RUN_ROOT/calibration/calibration.json" \
  --candidate-calibration "$K0_RUN_ROOT/calibration/calibration.json" \
  --output "$FINAL_ROOT/k0_vs_c0.json"

python3 - "$FINAL_ROOT/k0_vs_c0.json" "$FINAL_ROOT/final_decision.json" <<'PY'
import json,sys
comparison=json.load(open(sys.argv[1]))
decision={
  "status": comparison["status"],
  "production_decision": "RETAIN_QWEN35_E1_FULL90_84P642",
  "next_action": comparison["next_action"],
  "automatic_followup_started": False,
  "valid_b_accessed": False,
  "test_accessed": False,
  "submission_generated": False,
}
with open(sys.argv[2],"w") as handle:
  json.dump(decision,handle,indent=2)
PY

echo "RETENTION_V3_SEQUENCE_COMPLETE: $FINAL_ROOT/final_decision.json"
trap - ERR
