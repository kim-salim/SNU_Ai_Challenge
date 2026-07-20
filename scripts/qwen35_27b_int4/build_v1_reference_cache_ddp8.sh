#!/usr/bin/env bash
set -euo pipefail

ROOT="${RETENTION_V3_ROOT:-/home/shpark/snu-ai-challenge/repositories/qwen35-27b-retention-v3-component-safe}"
V1_ROOT="/home/shpark/snu-ai-challenge/repositories/qwen35-27b-int4-stage-e1-v1"
V2_ROOT="/home/shpark/snu-ai-challenge/repositories/qwen35-27b-e1-champion-retention-v2"
IMAGE="${QWEN27_IMAGE:-snu-four-slot-teacher:v1-tmpfix}"
CONFIG="$V1_ROOT/configs/exp/qwen35_27b_stage_pair_e1_int4_full90_server.yaml"
MODEL_REVISION="fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_PATH="/home/shpark/snu-ai-challenge/.hf-cache/hub/models--Qwen--Qwen3.5-27B/snapshots/$MODEL_REVISION"
DATA_ROOT="/home/shpark/snu-ai-challenge/data/raw"
TRAIN_CSV="/home/shpark/snu-ai-challenge/data/splits/full_train_90_10_v1/train_90_v1.csv"
CHECKPOINT="$V1_ROOT/weights/qwen35_27b_stage_pair_e1_int4_v1_full90_20260719/qwen35_27b_stage_pair_e1_int4_v1_full90_ddp8_20260719/best"
SHARD_ROOT="$V2_ROOT/outputs/teacher_cache/shards"
OUTPUT_ROOT="$ROOT/outputs/retention_v3/cache"
ATTEMPT_ID="$(date +%Y%m%d_%H%M%S)"
REFERENCE_SHARDS="$OUTPUT_ROOT/v1_reference_shards_${ATTEMPT_ID}"
REFERENCE_CACHE="$OUTPUT_ROOT/qwen35_27b_v1_train_reference.pt"
AUDIT="$OUTPUT_ROOT/component_cache_audit.json"
TEACHER_CACHE="$V2_ROOT/outputs/teacher_cache/qwen35_e1_full90_train_teacher.pt"

for required in "$CONFIG" "$MODEL_PATH/config.json" "$TRAIN_CSV" "$CHECKPOINT/checkpoint_manifest.json" "$TEACHER_CACHE"; do
  [[ -s "$required" ]] || { echo "BLOCKED_MISSING_INPUT: $required" >&2; exit 2; }
done
[[ ! -e "$REFERENCE_CACHE" ]] || {
  echo "BLOCKED_REFUSING_REFERENCE_CACHE_OVERWRITE: $REFERENCE_CACHE" >&2
  exit 3
}
mkdir -p "$REFERENCE_SHARDS"

docker_common=(
  docker run --rm --network none --ipc=host --shm-size=32g
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
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

pids=()
for shard in $(seq 0 7); do
  shard_id=$(printf '%02d' "$shard")
  shard_csv="$SHARD_ROOT/teacher_shard_${shard_id}.csv"
  shard_out="$REFERENCE_SHARDS/output_${shard_id}"
  [[ -s "$shard_csv" ]] || { echo "BLOCKED_MISSING_SHARD: $shard_csv" >&2; exit 4; }
  mkdir -p "$shard_out"
  (
    "${docker_common[@]}" --gpus "device=$shard" -e CUDA_VISIBLE_DEVICES=0 "$IMAGE" \
      /opt/venv/bin/python -m snu_order.qwen3vl.evaluate_stage_pair \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --metadata-csv "$shard_csv" \
      --image-root "$DATA_ROOT" \
      --output-dir "$shard_out" \
      --max-samples -1 \
      --split-name train_teacher \
      --frame-chunk-size 1 \
      --base-model-path "$MODEL_PATH" \
      --base-model-revision "$MODEL_REVISION"
  ) >"$shard_out/stdout.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || { echo "V1_REFERENCE_SHARD_FAILURE" >&2; exit 5; }

raw_args=()
for shard in $(seq 0 7); do
  shard_id=$(printf '%02d' "$shard")
  raw="$REFERENCE_SHARDS/output_${shard_id}/raw_stage_pair_logits.pt"
  [[ -s "$raw" ]] || { echo "BLOCKED_MISSING_RAW_SHARD: $raw" >&2; exit 6; }
  raw_args+=(--raw "$raw")
done

"${docker_common[@]}" "$IMAGE" /opt/venv/bin/python \
  "$ROOT/scripts/qwen35_27b_int4/build_champion_teacher_cache.py" merge \
  --source-csv "$TRAIN_CSV" \
  "${raw_args[@]}" \
  --checkpoint "$CHECKPOINT" \
  --output "$REFERENCE_CACHE"

"${docker_common[@]}" "$IMAGE" /opt/venv/bin/python \
  "$ROOT/scripts/qwen35_27b_int4/analyze_component_caches.py" \
  --teacher-cache "$TEACHER_CACHE" \
  --reference-cache "$REFERENCE_CACHE" \
  --output "$AUDIT"

echo "V1_REFERENCE_CACHE_READY: $REFERENCE_CACHE"
echo "COMPONENT_CACHE_AUDIT: $AUDIT"
