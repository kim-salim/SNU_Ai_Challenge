#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

docker build \
  -f docker/qwen3vl/Dockerfile \
  -t snu-qwen3vl:latest \
  .
