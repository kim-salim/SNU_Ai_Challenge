#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-python3}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"${PYTHON}" - <<'PY'
from pathlib import Path
model_dir = Path("weights/pretrained/siglip2_base_224")
if not model_dir.exists() or not any(model_dir.iterdir()):
    raise SystemExit(
        "Missing local SigLIP2 weights in weights/pretrained/siglip2_base_224. "
        "Place pretrained files there before running offline inference."
    )
print("offline environment variables and model directory are present")
PY

