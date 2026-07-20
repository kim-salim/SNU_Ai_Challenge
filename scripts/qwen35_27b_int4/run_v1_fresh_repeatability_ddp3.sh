#!/usr/bin/env bash
set -euo pipefail

ROOT="${RETENTION_V3_ROOT:-/home/shpark/snu-ai-challenge/repositories/qwen35-27b-retention-v3-component-safe}"
V1_ROOT="/home/shpark/snu-ai-challenge/repositories/qwen35-27b-int4-stage-e1-v1"
IMAGE="${QWEN27_IMAGE:-snu-four-slot-teacher:v1-tmpfix}"
CONFIG="$V1_ROOT/configs/exp/qwen35_27b_stage_pair_e1_int4_full90_server.yaml"
CHECKPOINT="$V1_ROOT/weights/qwen35_27b_stage_pair_e1_int4_v1_full90_20260719/qwen35_27b_stage_pair_e1_int4_v1_full90_ddp8_20260719/best"
VALID_CSV="/home/shpark/snu-ai-challenge/data/splits/full_train_90_10_v1/valid_10_v1.csv"
DATA_ROOT="/home/shpark/snu-ai-challenge/data/raw"
MODEL_REVISION="fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_PATH="/home/shpark/snu-ai-challenge/.hf-cache/hub/models--Qwen--Qwen3.5-27B/snapshots/$MODEL_REVISION"
OUTPUT_ROOT="$ROOT/outputs/retention_v3/p0/fresh_repeats"
REPORT="$ROOT/outputs/retention_v3/p0/v1_fresh_repeatability.json"

[[ ! -e "$OUTPUT_ROOT" && ! -e "$REPORT" ]] || {
  echo "BLOCKED_REFUSING_FRESH_REPEAT_OVERWRITE" >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT"

docker_common=(
  docker run --rm --network none --ipc=host --shm-size=32g
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

pids=()
for run in 1 2 3; do
  gpu=$((run - 1))
  output="$OUTPUT_ROOT/run_$run"
  mkdir -p "$output"
  (
    "${docker_common[@]}" --gpus "device=$gpu" -e CUDA_VISIBLE_DEVICES=0 "$IMAGE" \
      /opt/venv/bin/python -m snu_order.qwen3vl.evaluate_stage_pair \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --metadata-csv "$VALID_CSV" \
      --image-root "$DATA_ROOT" --output-dir "$output" --max-samples -1 --split-name valid_a \
      --frame-chunk-size 1 --base-model-path "$MODEL_PATH" --base-model-revision "$MODEL_REVISION"
  ) >"$output/stdout.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || { echo "P0_FRESH_REPEAT_FORWARD_FAILURE" >&2; exit 3; }

"${docker_common[@]}" "$IMAGE" /opt/venv/bin/python -m snu_order.qwen3vl.runtime_repeatability_gate \
  --raw "$OUTPUT_ROOT/run_1/raw_stage_pair_logits.pt" \
  --raw "$OUTPUT_ROOT/run_2/raw_stage_pair_logits.pt" \
  --raw "$OUTPUT_ROOT/run_3/raw_stage_pair_logits.pt" \
  --output "$REPORT"

echo "P0_FRESH_REPEATABILITY_READY: $REPORT"
