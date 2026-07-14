#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-snu-qwen3vl-desktop:exact-20260714}"
HOST_LEGACY_ROOT="${HOST_LEGACY_ROOT:-/home/slkim/snu-ai-frame-ordering}"
HOST_DATA_DIR="${HOST_DATA_DIR:-$HOST_LEGACY_ROOT/data}"
HOST_HF_HOME="${HOST_HF_HOME:-$HOST_LEGACY_ROOT/.hf-cache}"
FINGERPRINT_HF_HOME="/home/a/snu-ai-frame-ordering/.hf-cache"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi
for required in "$ROOT_DIR" "$HOST_DATA_DIR" "$HOST_HF_HOME"; do
  if [[ ! -d "$required" ]]; then
    echo "Required container mount does not exist: $required" >&2
    exit 1
  fi
done
if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  echo "Hardening runs require CUDA_VISIBLE_DEVICES=0" >&2
  exit 1
fi
if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "Hardening runs require WORLD_SIZE=1" >&2
  exit 1
fi

DOCKER_CMD=(docker run --rm
  --gpus device=0
  --ipc host
  --tmpfs /tmp:rw,exec,nosuid,size=8g,mode=1777
  --network none
  --user "$(id -u):$(id -g)"
  --volume "$ROOT_DIR:/workspace"
  --volume "$HOST_DATA_DIR:/workspace/data:ro"
  --volume "$HOST_LEGACY_ROOT:/legacy:ro"
  --volume "$HOST_HF_HOME:$FINGERPRINT_HF_HOME:ro"
  --workdir /workspace
  --env HOME=/tmp
  --env "HF_HOME=$FINGERPRINT_HF_HOME"
  --env "HF_HUB_CACHE=$FINGERPRINT_HF_HOME/hub"
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  --env TOKENIZERS_PARALLELISM=false
  --env CUDA_VISIBLE_DEVICES=0
  --env WORLD_SIZE=1
  --env PYTHONPATH=/workspace/src
  --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
  "$DOCKER_IMAGE"
  "$@")

printf -v COMMAND_TEXT '%q ' "${DOCKER_CMD[@]}"
echo "$COMMAND_TEXT"
if docker info >/dev/null 2>&1; then
  exec "${DOCKER_CMD[@]}"
fi
if command -v sg >/dev/null 2>&1 && getent group docker >/dev/null 2>&1; then
  exec sg docker -c "$COMMAND_TEXT"
fi
echo "The current user cannot access the Docker daemon." >&2
exit 1
