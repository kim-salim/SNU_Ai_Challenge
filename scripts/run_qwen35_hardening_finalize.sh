#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CONTAINER_RUNNER="scripts/run_qwen35_hardening_exact_container.sh"
E1_CONFIG="configs/exp/qwen35_9b_stage_pair_v2_text_anchor.yaml"
E2_CONFIG="configs/exp/qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger.yaml"
E1_CHECKPOINT="weights/qwen35_9b_stage_pair_v2_text_anchor/qwen35_9b_stage_pair_v2_text_anchor_full_20260714/best"
E1_RAW="outputs/hardening_20260714/p2_chunked_inference/legacy_run_1/raw_stage_pair_logits.pt"
E1_CALIBRATION_DIR="outputs/hardening_20260714/p1_state_server_recalibration"
E2_RUN_ID="qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger_full_20260714"
E2_CHECKPOINT="weights/qwen35_9b_stage_pair_v2_text_anchor_e2_vision_merger/$E2_RUN_ID/best"
E2_ROOT="outputs/hardening_20260714/p3_e2_vision_merger/$E2_RUN_ID"
E2_RAW="$E2_ROOT/valid_a_raw/raw_stage_pair_logits.pt"
E2_CALIBRATION_DIR="$E2_ROOT/calibration"
COMPARISON_DIR="outputs/hardening_20260714/p3_e2_vision_merger"
DECISION="$COMPARISON_DIR/decision.json"
FINAL_DIR="outputs/hardening_20260714/final"
FINAL_LOG_DIR="outputs/hardening_20260714/final_logs"
SAMPLE_SUBMISSION="data/raw/sample_submission.csv"
TEST_METADATA="data/raw/test.csv"

mkdir -p "$FINAL_LOG_DIR"
if [[ -e "$FINAL_DIR" ]]; then
  echo "Refusing to overwrite final output directory: $FINAL_DIR" >&2
  exit 1
fi

echo "Waiting for the E2 full run to finish before comparison."
while [[ ! -s "$E2_CALIBRATION_DIR/calibration.json" ]] || pgrep -f "$E2_RUN_ID" >/dev/null; do
  if ! pgrep -f "run_qwen35_hardening_server_pipeline.sh" >/dev/null \
    && ! pgrep -f "$E2_RUN_ID" >/dev/null \
    && [[ ! -s "$E2_CALIBRATION_DIR/calibration.json" ]]; then
    echo "E2 pipeline exited without a calibration artifact." >&2
    exit 1
  fi
  date '+%Y-%m-%d %H:%M:%S waiting for E2 completion'
  sleep 60
done

COMPARE_CMD=(/opt/venv/bin/python3 -m snu_order.qwen3vl.compare_e1_e2
  --e1-raw "$E1_RAW"
  --e1-calibration-dir "$E1_CALIBRATION_DIR"
  --e2-raw "$E2_RAW"
  --e2-calibration-dir "$E2_CALIBRATION_DIR"
  --e2-verification "$E2_ROOT/checkpoint_verification_1.json"
  --e2-verification "$E2_ROOT/checkpoint_verification_2.json"
  --gradient-health "$E2_ROOT/gradient_health.json"
  --semantic-diff "$E2_ROOT/config_semantic_diff.json"
  --chunk-selection outputs/hardening_20260714/p2_chunked_inference/selected_chunk_config.json
  --output-dir "$COMPARISON_DIR")
printf '%q ' "${COMPARE_CMD[@]}"
printf '\n'
bash "$CONTAINER_RUNNER" "${COMPARE_CMD[@]}" \
  2>&1 | tee "$FINAL_LOG_DIR/e1_e2_comparison.log"

SELECTED_CANDIDATE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"])' "$DECISION")"
case "$SELECTED_CANDIDATE" in
  state_e1)
    SELECTED_CONFIG="$E1_CONFIG"
    SELECTED_CHECKPOINT="$E1_CHECKPOINT"
    SELECTED_CALIBRATION="$E1_CALIBRATION_DIR/calibration.json"
    ;;
  state_e2_vision_merger)
    SELECTED_CONFIG="$E2_CONFIG"
    SELECTED_CHECKPOINT="$E2_CHECKPOINT"
    SELECTED_CALIBRATION="$E2_CALIBRATION_DIR/calibration.json"
    ;;
  *)
    echo "Unexpected selected candidate: $SELECTED_CANDIDATE" >&2
    exit 1
    ;;
esac
echo "Performance-first champion: $SELECTED_CANDIDATE"

VERIFY_JSON="$FINAL_LOG_DIR/final_checkpoint_verification.json"
bash "$CONTAINER_RUNNER" /opt/venv/bin/python3 \
  -m snu_order.qwen3vl.verify_stage_pair_checkpoint \
  --config "$SELECTED_CONFIG" \
  --checkpoint "$SELECTED_CHECKPOINT" \
  --metadata-csv data/splits/ab_v1/valid_a_v1.csv \
  --image-root data/raw \
  --max-samples 8 \
  --output-json "$VERIFY_JSON" \
  2>&1 | tee "$FINAL_LOG_DIR/final_checkpoint_verification.log"

mkdir "$FINAL_DIR"
INFERENCE_PROFILE="$FINAL_LOG_DIR/final_inference_profile.json"
bash "$CONTAINER_RUNNER" /opt/venv/bin/python3 \
  -m snu_order.qwen3vl.inference_stage_pair \
  --config "$SELECTED_CONFIG" \
  --checkpoint "$SELECTED_CHECKPOINT" \
  --metadata-csv "$TEST_METADATA" \
  --image-root data/raw \
  --sample-submission "$SAMPLE_SUBMISSION" \
  --output-csv "$FINAL_DIR/submission.csv" \
  --calibration "$SELECTED_CALIBRATION" \
  --frame-chunk-size 4 \
  --profile-json "$INFERENCE_PROFILE" \
  2>&1 | tee "$FINAL_LOG_DIR/final_inference.log"

bash "$CONTAINER_RUNNER" /opt/venv/bin/python3 \
  -m snu_order.qwen3vl.finalize_hardening \
  --output-dir "$FINAL_DIR" \
  --submission "$FINAL_DIR/submission.csv" \
  --sample-submission "$SAMPLE_SUBMISSION" \
  --decision "$DECISION" \
  --selected-candidate "$SELECTED_CANDIDATE" \
  --checkpoint "$SELECTED_CHECKPOINT" \
  --config "$SELECTED_CONFIG" \
  --calibration "$SELECTED_CALIBRATION" \
  --inference-profile "$INFERENCE_PROFILE" \
  2>&1 | tee "$FINAL_LOG_DIR/finalize.log"

head -n 1 "$FINAL_DIR/submission.csv"
wc -l "$FINAL_DIR/submission.csv"
sha256sum "$FINAL_DIR/submission.csv"
echo "Final hardening pipeline completed: $ROOT_DIR/$FINAL_DIR/submission.csv"
