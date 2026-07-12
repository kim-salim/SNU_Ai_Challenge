#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p .hf-cache logs outputs
mkdir -p .cache .torchinductor-cache

GPU_DEVICE="${GPU_DEVICE:-all}"
if [[ "$GPU_DEVICE" == "all" ]]; then
  GPU_REQUEST="all"
else
  GPU_REQUEST="device=$GPU_DEVICE"
fi

docker run --rm -it \
  --gpus "$GPU_REQUEST" \
  --user "$(id -u):$(id -g)" \
  --ipc=host \
  --shm-size=32g \
  -e PATH=/home/shpark/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e HOME=/workspace \
  -e HF_HOME=/workspace/.hf-cache \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e PYTHONPATH=/workspace/src \
  -e XDG_CACHE_HOME=/workspace/.cache \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor-cache \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  snu-qwen3vl:latest \
  bash
