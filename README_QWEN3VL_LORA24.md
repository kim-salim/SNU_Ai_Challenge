# Qwen3-VL 8B LoRA24 Pipeline

This pipeline trains a single Qwen3-VL 8B + QLoRA model with one 24-way permutation classifier head.

It does not ensemble models, does not use external APIs, and does not combine Qwen predictions with SigLIP predictions at inference.

## Structure

- Base model: `Qwen/Qwen3-VL-8B-Instruct`, loaded offline with `local_files_only=True`.
- Input: sentence plus four supplied images in order.
- Labels: the official answer `[1,2,3,4]` style is mapped to one lexicographic permutation class in `[0,23]`.
- Head: last non-padding token hidden state -> LayerNorm -> Dropout -> Linear(24).
- Training: attention-only QLoRA on `q_proj/k_proj/v_proj/o_proj` plus classifier head.
- Loss: 24-way cross entropy plus pairwise and position marginal losses derived from the same 24-way probability distribution.

## Split Policy

- `train_ab`: gradient updates only.
- `valid_a`: early stopping and single checkpoint selection only.
- `valid_b`: lockbox evaluation only after explicit approval.

The default pipeline stops after valid_a. If best valid_a exact match is below `0.33`, valid_b is blocked. valid_b pass requires exact correct count `ceil(0.30 * N)`, so for `N=1430` the gate is `429` correct.

## Commands

Environment check:

```bash
bash scripts/check_qwen3vl_lora_env.sh
```

Zero-shot valid_a diagnostics:

```bash
bash scripts/run_qwen3vl_zero_shot_diagnostics.sh
```

64-sample overfit sanity:

```bash
bash scripts/run_qwen3vl_lora_overfit64.sh
```

Frozen representation probe:

```bash
bash scripts/run_qwen3vl_frozen_probe.sh
```

Subset QLoRA:

```bash
bash scripts/run_qwen3vl_lora_subset.sh
```

Full QLoRA:

```bash
bash scripts/run_qwen3vl_lora_full.sh
```

valid_b lockbox, only after valid_a reaches the threshold and the user explicitly approves:

```bash
bash scripts/run_qwen3vl_lora_valid_b_lockbox.sh --unlock-valid-b
```

## Checkpoints

Best and last checkpoints are saved under:

- `weights/qwen3vl_lora24/best`
- `weights/qwen3vl_lora24/last`

Only adapter weights, classifier weights, config, metrics, permutation ordering, and training metadata are saved. The base model is not copied into the checkpoint.

## Offline/Docker Notes

The Docker image installs Python dependencies only. Model weights must already exist in the mounted Hugging Face cache. Runtime scripts set:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `TOKENIZERS_PARALLELISM=false`

Final timing and memory must still be profiled on the target RTX 3090 24GB profile. Recommended submission gate is peak allocated VRAM <= 22GB, estimated full test time <= 20 hours, parse failure 0, invalid answer 0.

## Known Risks

- QLoRA depends on the installed CUDA, PyTorch, transformers, peft, and bitsandbytes combination.
- Multi-image Qwen3 processor behavior is reused from the existing zero-shot path and is intentionally batch-size 1 in this first pass.
- valid_b detailed errors must not be used for tuning after lockbox evaluation.
