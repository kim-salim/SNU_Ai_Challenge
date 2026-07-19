#!/usr/bin/env bash
set -euo pipefail
umask 022

ROOT="${QWEN27_ROOT:-/home/shpark/snu-ai-challenge/repositories/qwen35-27b-int4-stage-e1-v1}"
IMAGE="${QWEN27_IMAGE:-snu-four-slot-teacher:v1-tmpfix}"
CONFIG="$ROOT/configs/exp/qwen35_27b_stage_pair_e1_int4_full90_server.yaml"
RUN_ID="qwen35_27b_stage_pair_e1_int4_v1_full90_ddp8_20260719"
MODEL_REVISION="fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_PATH="/home/shpark/snu-ai-challenge/.hf-cache/hub/models--Qwen--Qwen3.5-27B/snapshots/$MODEL_REVISION"
MIGRATED_HEADS="$ROOT/artifacts/qwen35_27b_int4_port/02_implementation/migrated_heads.pt"
SPLIT_ROOT="/home/shpark/snu-ai-challenge/data/splits/full_train_90_10_v1"
TRAIN_CSV="$SPLIT_ROOT/train_90_v1.csv"
VALID_CSV="$SPLIT_ROOT/valid_10_v1.csv"
DATA_ROOT="/home/shpark/snu-ai-challenge/data/raw"
TEST_CSV="$DATA_ROOT/test.csv"
REFERENCE_CSV="$DATA_ROOT/sample_submission.csv"
HF_HOME_HOST="/home/shpark/snu-ai-challenge/.hf-cache"
EXPERIMENT_ROOT="$ROOT/outputs/experiments/qwen35_27b_stage_pair_e1_int4_v1_full90_20260719"
RUN_ROOT="$EXPERIMENT_ROOT/$RUN_ID"
SUMMARY="$RUN_ROOT/summary.json"
CHECKPOINT="$ROOT/weights/qwen35_27b_stage_pair_e1_int4_v1_full90_20260719/$RUN_ID/best"
POST_ROOT="$RUN_ROOT/post_training_submission"
VALID_RAW="$POST_ROOT/valid10_best_raw"
CALIBRATION_ROOT="$POST_ROOT/calibration"
FINAL_ROOT="/home/shpark/snu-ai-challenge/artifacts/submissions/qwen35_27b_e1_int4_full90_20260719"
FINAL_NAME="VERIFIED__SUBMIT_THIS_QWEN35_27B_E1_INT4_FULL90_0719.csv"
FINAL_CSV="$FINAL_ROOT/$FINAL_NAME"
EASY_COPY="/home/shpark/snu-ai-challenge/$FINAL_NAME"
STATUS_JSON="$RUN_ROOT/pipeline_status.json"

mkdir -p "$RUN_ROOT"
on_error() {
  status=$?
  printf '{"status":"FAILED","exit_code":%d,"run_id":"%s"}\n' "$status" "$RUN_ID" > "$STATUS_JSON"
  exit "$status"
}
trap on_error ERR

for required in "$CONFIG" "$MIGRATED_HEADS" "$TRAIN_CSV" "$VALID_CSV" "$TEST_CSV" "$REFERENCE_CSV" "$MODEL_PATH/config.json"; do
  [[ -s "$required" ]] || { echo "BLOCKED: required input missing: $required" >&2; exit 2; }
done
[[ "$(find "$MODEL_PATH" -maxdepth 1 \( -type l -o -type f \) -name 'model.safetensors-*.safetensors' | wc -l)" -eq 11 ]] || {
  echo "BLOCKED: expected 11 model shards" >&2
  exit 2
}
[[ "$(find "$HF_HOME_HOST/hub/models--Qwen--Qwen3.5-27B" -name '*.incomplete' | wc -l)" -eq 0 ]] || {
  echo "BLOCKED: 27B model download is incomplete" >&2
  exit 2
}
[[ ! -e "$FINAL_ROOT" && ! -e "$EASY_COPY" ]] || {
  echo "BLOCKED: refusing to overwrite an existing submission artifact" >&2
  exit 3
}

docker_common=(
  docker run --rm --gpus all --network none --ipc=host --shm-size=64g
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
  -e HF_HOME="$HF_HOME_HOST"
  -e PYTHONPATH="$ROOT/src"
  -v "$ROOT:$ROOT"
  -v "$DATA_ROOT:$DATA_ROOT:ro"
  -v "$SPLIT_ROOT:$SPLIT_ROOT:ro"
  -v "$HF_HOME_HOST:$HF_HOME_HOST:ro"
  -v /home/shpark/snu-ai-challenge/artifacts:/home/shpark/snu-ai-challenge/artifacts
  -w "$ROOT"
)

printf '{"status":"TRAINING_STARTING","run_id":"%s","world_size":8,"train_samples":8581,"valid_samples":954}\n' "$RUN_ID" > "$STATUS_JSON"
echo "$(date --iso-8601=seconds) starting 27B NF4 QLoRA on GPUs 0-7"
"${docker_common[@]}" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e NCCL_DEBUG=WARN \
  "$IMAGE" \
  /opt/venv/bin/torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29527 \
  -m snu_order.qwen3vl.train_stage_pair \
  --config "$CONFIG" \
  --mode qlora_stage_pair \
  --init-head-from "$MIGRATED_HEADS" \
  --epochs 3 \
  --run-id "$RUN_ID"

[[ -s "$SUMMARY" ]] || { echo "BLOCKED: training ended without summary.json" >&2; exit 4; }
for required in checkpoint_manifest.json heads.pt prompt_fingerprint.json permutations.json lora_target_manifest.json; do
  [[ -s "$CHECKPOINT/$required" ]] || { echo "BLOCKED: incomplete best checkpoint: $required" >&2; exit 5; }
done
[[ -s "$CHECKPOINT/adapter/adapter_model.safetensors" ]] || { echo "BLOCKED: adapter missing" >&2; exit 5; }
mkdir -p "$POST_ROOT" "$FINAL_ROOT"

docker_single() {
  "${docker_common[@]}" -e CUDA_VISIBLE_DEVICES=0 "$IMAGE" "$@"
}

echo "Verifying the selected 27B checkpoint."
docker_single /opt/venv/bin/python -m snu_order.qwen3vl.verify_stage_pair_checkpoint \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --metadata-csv "$VALID_CSV" \
  --image-root "$DATA_ROOT" --max-samples 8 --output-json "$POST_ROOT/checkpoint_verification.json" \
  --base-model-path "$MODEL_PATH" --base-model-revision "$MODEL_REVISION"

echo "Evaluating the selected checkpoint on the fixed 10% validation split."
docker_single /opt/venv/bin/python -m snu_order.qwen3vl.evaluate_stage_pair \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --metadata-csv "$VALID_CSV" \
  --image-root "$DATA_ROOT" --output-dir "$VALID_RAW" --max-samples -1 --split-name valid_a \
  --frame-chunk-size 1 --base-model-path "$MODEL_PATH" --base-model-revision "$MODEL_REVISION"

echo "Fitting one bounded calibration on the fixed validation split."
docker_single /opt/venv/bin/python -m snu_order.qwen3vl.calibration_stage_pair \
  --config "$CONFIG" --raw-logits "$VALID_RAW/raw_stage_pair_logits.pt" \
  --output-dir "$CALIBRATION_ROOT" --tune-split valid_a \
  --binding "checkpoint_manifest_sha256=$CHECKPOINT/checkpoint_manifest.json" \
  --binding "adapter_sha256=$CHECKPOINT/adapter/adapter_model.safetensors" \
  --binding "heads_sha256=$CHECKPOINT/heads.pt" \
  --binding "prompt_fingerprint_sha256=$CHECKPOINT/prompt_fingerprint.json" \
  --binding "processor_fingerprint_sha256=$CHECKPOINT/processor/tokenizer_config.json" \
  --binding "permutation_mapping_sha256=$CHECKPOINT/permutations.json" \
  --binding "validation_split_sha256=$VALID_CSV" \
  --binding "scorer_code_sha256=$ROOT/src/snu_order/qwen3vl/stage_pair_scorer.py"

cat > "$POST_ROOT/selection_decision.json" <<EOF
{
  "selected_candidate": "$RUN_ID",
  "selection_split": "full_train_90_10_v1/valid_10_v1",
  "checkpoint": "$CHECKPOINT",
  "submission_generated_after_successful_training": true
}
EOF

echo "Running deterministic calibrated Official Test inference."
docker_single /opt/venv/bin/python -m snu_order.qwen3vl.inference_stage_pair \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --calibration "$CALIBRATION_ROOT/calibration.json" \
  --metadata-csv "$TEST_CSV" --image-root "$DATA_ROOT" --sample-submission "$REFERENCE_CSV" \
  --output-csv "$FINAL_CSV" --debug-csv "$POST_ROOT/test_prediction_debug.csv" \
  --profile-json "$POST_ROOT/inference_profile.json" --max-samples -1 --frame-chunk-size 4 --device-index 0 \
  --valid-split-binding "$VALID_CSV" --base-model-path "$MODEL_PATH" --base-model-revision "$MODEL_REVISION"

echo "Running strict 819-row submission hardening."
docker_single /opt/venv/bin/python -m snu_order.qwen3vl.finalize_hardening \
  --output-dir "$FINAL_ROOT" --submission "$FINAL_CSV" --sample-submission "$REFERENCE_CSV" \
  --decision "$POST_ROOT/selection_decision.json" --selected-candidate "$RUN_ID" \
  --checkpoint "$CHECKPOINT" --config "$CONFIG" --calibration "$CALIBRATION_ROOT/calibration.json" \
  --inference-profile "$POST_ROOT/inference_profile.json"
docker_single /opt/venv/bin/python -m snu_order.data.validate_submission \
  --file "$FINAL_CSV" --reference "$REFERENCE_CSV" | tee "$POST_ROOT/final_validator_stdout.json"

cp "$FINAL_CSV" "$EASY_COPY"
submission_sha="$(sha256sum "$FINAL_CSV" | awk '{print $1}')"
copy_sha="$(sha256sum "$EASY_COPY" | awk '{print $1}')"
[[ "$submission_sha" == "$copy_sha" ]] || { echo "BLOCKED: easy-copy hash mismatch" >&2; exit 6; }
printf '{"status":"SUBMISSION_READY","run_id":"%s","submission":"%s","sha256":"%s","row_count":819}\n' \
  "$RUN_ID" "$FINAL_CSV" "$submission_sha" > "$STATUS_JSON"
trap - ERR
echo "SUBMISSION_READY: $FINAL_CSV"
echo "EASY_COPY: $EASY_COPY"
echo "SHA256: $submission_sha"
