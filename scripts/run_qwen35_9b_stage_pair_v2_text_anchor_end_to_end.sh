#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ID="${RUN_ID:-qwen35_9b_stage_pair_v2_text_anchor}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
SUBMISSION_MAX_SAMPLES="${SUBMISSION_MAX_SAMPLES:--1}"
EPOCHS="${EPOCHS:-4}"
RUN_LOCKBOX="${RUN_LOCKBOX:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29571}"
SMOKE="${SMOKE:-0}"
USE_DOCKER="${USE_DOCKER:-1}"
DOCKER_IMAGE="${DOCKER_IMAGE:-snu-qwen3vl-desktop:latest}"
DOCKER_DRY_RUN="${DOCKER_DRY_RUN:-0}"
PIPELINE_DRY_RUN="${PIPELINE_DRY_RUN:-0}"
SNU_E1_IN_CONTAINER="${SNU_E1_IN_CONTAINER:-0}"

if [[ "$SMOKE" == "1" ]]; then
  echo "End-to-end mode does not submit from a smoke checkpoint."
  echo "Run SMOKE=1 scripts/run_qwen35_9b_stage_pair_v2_text_anchor.sh instead."
  exit 2
fi

if [[ "$USE_DOCKER" == "1" && "$SNU_E1_IN_CONTAINER" != "1" && ! -f /.dockerenv ]]; then
  COMMON_GIT_DIR="$(readlink -f "$(git rev-parse --git-common-dir)")"
  HOST_REPO_ROOT="$(dirname "$COMMON_GIT_DIR")"
  HOST_DATA_DIR="${HOST_DATA_DIR:-$HOST_REPO_ROOT/data}"
  HOST_HF_HOME="${HOST_HF_HOME:-$HOST_REPO_ROOT/.hf-cache}"
  if [[ ! -d "$HOST_DATA_DIR" ]]; then
    echo "Host data directory does not exist: $HOST_DATA_DIR"
    exit 1
  fi
  if [[ ! -d "$HOST_HF_HOME" ]]; then
    echo "Host Hugging Face cache does not exist: $HOST_HF_HOME"
    exit 1
  fi

  DOCKER_CMD=(docker run --rm
    --gpus all
    --ipc host
    --network none
    --user "$(id -u):$(id -g)"
    --volume "$HOST_REPO_ROOT:$HOST_REPO_ROOT"
    --volume "$HOST_DATA_DIR:$ROOT_DIR/data:ro"
    --workdir "$ROOT_DIR"
    --env HOME=/tmp
    --env "HF_HOME=$HOST_HF_HOME"
    --env "HF_HUB_CACHE=$HOST_HF_HOME/hub"
    --env SNU_E1_IN_CONTAINER=1
    --env USE_DOCKER=0
    --env "PYTHON_BIN=$PYTHON_BIN"
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    --env "RUN_ID=$RUN_ID"
    --env "TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES"
    --env "SUBMISSION_MAX_SAMPLES=$SUBMISSION_MAX_SAMPLES"
    --env "EPOCHS=$EPOCHS"
    --env "RUN_LOCKBOX=$RUN_LOCKBOX"
    --env "OVERWRITE=$OVERWRITE"
    --env "MAIN_PROCESS_PORT=$MAIN_PROCESS_PORT"
    --env "PIPELINE_DRY_RUN=$PIPELINE_DRY_RUN"
    "$DOCKER_IMAGE"
    bash "$ROOT_DIR/scripts/run_qwen35_9b_stage_pair_v2_text_anchor_end_to_end.sh")
  printf -v DOCKER_CMD_TEXT '%q ' "${DOCKER_CMD[@]}"
  echo "Re-entering the pinned desktop environment: $DOCKER_IMAGE"
  echo "$DOCKER_CMD_TEXT"
  if [[ "$DOCKER_DRY_RUN" == "1" ]]; then
    exit 0
  fi
  if docker info >/dev/null 2>&1; then
    exec "${DOCKER_CMD[@]}"
  fi
  if command -v sg >/dev/null 2>&1 && getent group docker >/dev/null 2>&1; then
    exec sg docker -c "$DOCKER_CMD_TEXT"
  fi
  echo "Docker is required but the current shell cannot access the Docker daemon."
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import accelerate, bitsandbytes, peft, torch, transformers' >/dev/null 2>&1; then
  echo "The selected runtime is missing Qwen training dependencies."
  echo "Use the default Docker path or set PYTHON_BIN to the pinned environment."
  exit 1
fi

for required_input in \
  data/splits/ab_v1/train_ab_v1.csv \
  data/splits/ab_v1/valid_a_v1.csv; do
  if [[ ! -f "$required_input" ]]; then
    echo "Required training input is not mounted: $ROOT_DIR/$required_input"
    exit 1
  fi
done

TRAIN_SCRIPT="$ROOT_DIR/scripts/run_qwen35_9b_stage_pair_v2_text_anchor.sh"
SUBMISSION_SCRIPT="$ROOT_DIR/scripts/run_qwen35_9b_stage_pair_v2_text_anchor_submission.sh"

if [[ "$PIPELINE_DRY_RUN" == "1" ]]; then
  echo "Pinned runtime and Docker mounts are ready."
  echo "Training script: $TRAIN_SCRIPT"
  echo "Submission script: $SUBMISSION_SCRIPT"
  exit 0
fi

echo "Starting E1 training pipeline: RUN_ID=$RUN_ID"
PYTHON_BIN="$PYTHON_BIN" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
RUN_ID="$RUN_ID" \
MAX_SAMPLES="$TRAIN_MAX_SAMPLES" \
EPOCHS="$EPOCHS" \
RUN_LOCKBOX="$RUN_LOCKBOX" \
OVERWRITE="$OVERWRITE" \
MAIN_PROCESS_PORT="$MAIN_PROCESS_PORT" \
SMOKE=0 \
bash "$TRAIN_SCRIPT"

echo "Training, checkpoint verification, valid-A evaluation, and calibration completed."
echo "Starting deterministic calibrated submission inference: RUN_ID=$RUN_ID"
PYTHON_BIN="$PYTHON_BIN" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
RUN_ID="$RUN_ID" \
MAX_SAMPLES="$SUBMISSION_MAX_SAMPLES" \
OVERWRITE="$OVERWRITE" \
bash "$SUBMISSION_SCRIPT"

SUBMISSION="$ROOT_DIR/outputs/experiments/qwen35_9b_stage_pair_v2_text_anchor/$RUN_ID/submission/submission.csv"
PROFILE="$ROOT_DIR/outputs/experiments/qwen35_9b_stage_pair_v2_text_anchor/$RUN_ID/submission/inference_profile.json"
if [[ ! -s "$SUBMISSION" || ! -s "$PROFILE" ]]; then
  echo "End-to-end pipeline finished without required submission artifacts."
  exit 1
fi

echo "End-to-end E1 pipeline completed successfully."
echo "Submission: $SUBMISSION"
echo "Runtime profile: $PROFILE"
