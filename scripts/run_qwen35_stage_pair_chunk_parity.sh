#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python3}"
CONTAINER_RUNNER="scripts/run_qwen35_hardening_exact_container.sh"
CONFIG="configs/exp/qwen35_9b_stage_pair_v2_text_anchor.yaml"
CHECKPOINT="weights/qwen35_9b_stage_pair_v2_text_anchor/qwen35_9b_stage_pair_v2_text_anchor_full_20260714/best"
CALIBRATION="outputs/hardening_20260714/p1_state_reference_calibration/calibration.json"
OUTPUT_ROOT="outputs/hardening_20260714/p2_chunked_inference"
BASELINE_SUMMARY="$OUTPUT_ROOT/legacy_run_1/evaluation_summary.json"

mkdir -p "$OUTPUT_ROOT"
until [[ -s "$BASELINE_SUMMARY" ]]; do
  echo "Waiting for the first fresh unchunked baseline: $BASELINE_SUMMARY"
  sleep 15
done

run_evaluation() {
  local label="$1"
  local chunk_size="$2"
  local save_features="$3"
  local output_dir="$OUTPUT_ROOT/$label"
  if [[ -s "$output_dir/evaluation_summary.json" ]]; then
    echo "Refusing to overwrite completed parity run: $output_dir" >&2
    exit 1
  fi
  mkdir -p "$output_dir"
  local command=(bash "$CONTAINER_RUNNER" "$PYTHON_BIN" -m snu_order.qwen3vl.evaluate_stage_pair
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --metadata-csv data/splits/ab_v1/valid_a_v1.csv
    --image-root data/raw
    --output-dir "$output_dir"
    --max-samples -1
    --split-name valid_a
    --calibration "$CALIBRATION"
    --frame-chunk-size "$chunk_size")
  if [[ "$save_features" == "1" ]]; then
    command+=(--save-frame-features)
  fi
  printf '%q ' "${command[@]}"
  printf '\n'
  { time "${command[@]}"; } 2>&1 | tee "$OUTPUT_ROOT/$label.log"
}

run_evaluation legacy_run_2 4 0
run_evaluation chunk2 2 1
run_evaluation chunk1 1 1

for chunk_size in 2 1; do
  bash "$CONTAINER_RUNNER" "$PYTHON_BIN" -m snu_order.qwen3vl.certify_chunked_inference \
    --legacy-raw "$OUTPUT_ROOT/legacy_run_1/raw_stage_pair_logits.pt" \
    --legacy-repeat-raw "$OUTPUT_ROOT/legacy_run_2/raw_stage_pair_logits.pt" \
    --chunked-raw "$OUTPUT_ROOT/chunk${chunk_size}/raw_stage_pair_logits.pt" \
    --legacy-features "$OUTPUT_ROOT/legacy_run_1/frame_features.pt" \
    --chunked-features "$OUTPUT_ROOT/chunk${chunk_size}/frame_features.pt" \
    --calibration "$CALIBRATION" \
    --output "$OUTPUT_ROOT/parity_full_valid_a_chunk${chunk_size}.json" \
    2>&1 | tee "$OUTPUT_ROOT/certify_chunk${chunk_size}.log"
done

bash "$CONTAINER_RUNNER" "$PYTHON_BIN" -m snu_order.qwen3vl.summarize_chunked_inference \
  --root "$OUTPUT_ROOT" \
  2>&1 | tee "$OUTPUT_ROOT/summarize.log"

echo "P2 chunked inference certification completed: $OUTPUT_ROOT"
