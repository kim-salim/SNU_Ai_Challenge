# Qwen3-VL-8B Docker Runbook

This runbook uses the existing 24-candidate VLM pipeline with
`Qwen/Qwen3-VL-8B-Instruct`.

## Model

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Adapter mode: `model.type: qwen3_vl`
- Scoring: same option-label logprob scoring over A-X candidates
- Quantization: disabled by default for RTX PRO 5000 48GB
- Ensemble: none

## Docker

The target server already has Docker and NVIDIA Container Toolkit. Build the
project image from the server project root:

```bash
bash scripts/docker_build_qwen3vl.sh
```

Start an interactive container:

```bash
bash scripts/docker_run_qwen3vl.sh
```

Inside the container, run a small validation pass first:

```bash
bash scripts/run_qwen3vl8b_24candidate_valid_a_300.sh
```

Full validation:

```bash
bash scripts/run_qwen3vl8b_24candidate_valid_a_full.sh
bash scripts/run_qwen3vl8b_24candidate_valid_b_full.sh
```

Submission:

```bash
bash scripts/run_qwen3vl8b_24candidate_test.sh
```

The first manual dry-run can download model weights into `.hf-cache`. The
provided validation and submission scripts pass `--local-files-only`, so they
run offline once the cache is complete.
