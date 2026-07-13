# Qwen3.5 Stage/Pair v2 experiments

## E1 scope

E1 starts from the 81.4% Qwen3.5-9B Stage/Pair QLoRA baseline. The train/valid split,
4-bit NF4 double quantization, BF16 compute, Stage/Set/Pair heads, structured loss,
permutation augmentation, optimizer, scheduler, seed, and four-epoch schedule are unchanged.

E1 changes two model variables:

1. The existing eight full-attention layers keep LoRA on `q_proj`, `k_proj`, `v_proj`,
   and `o_proj` with rank 16 and alpha 32.
2. The 24 Gated DeltaNet layers add LoRA on `in_proj_qkv`, `in_proj_z`, and
   `out_proj` with rank 8 and alpha 16.
3. Frame representations use mean pooling over the token span for literal `STATE:`.
   The prompt uses `enable_thinking=false` and `add_generation_prompt=false`.

The pinned revision exposes `model.language_model.layers` with 32 layers: full
attention at indices 3, 7, 11, 15, 19, 23, 27, and 31, and linear attention at the
remaining 24 indices. The exact text adapter plan contains 32 full-attention and 72
linear-attention projections. Tail-only matching is not used.

## Common reproducibility layer

Checkpoint format v2 atomically stores the Stage/Set/Pair heads, the single PEFT
adapter, processor/tokenizer, config, metrics, permutations, exact LoRA plan, prompt
fingerprint, and a SHA-256 file manifest. Loading verifies every manifest entry,
runtime architecture fields, head state dicts, adapter tensors, and the regenerated
prompt fingerprint. A v2 config cannot use the legacy permissive loader.

The prompt fingerprint uses the fixed sentence `A person performs an action.` and a
blank RGB image. It records package versions, template/tokenizer hashes, rendered
prompt, complete input-ID hash, final 64 IDs and decoded text, anchor IDs/span,
thinking/generation settings, pooling mode, and image grid.

Best checkpoint selection still uses raw `stage_weight=1.0` and `pair_weight=0.3`
scores, ordered by exact match, MRR, and top-3 accuracy. After training, the best
checkpoint is evaluated again on valid-A and a deterministic grid selects pair weight,
stage temperature, and pair temperature. Raw and calibrated metrics are reported
separately. Valid-B is never used to select calibration values and is evaluated only
when `RUN_LOCKBOX=1` is explicitly set.

## Vision merger support

The shared planner supports exactly `model.visual.merger.linear_fc1` and
`model.visual.merger.linear_fc2`. E1 sets `vision_merger_lora.enabled: false`, so its
trainable vision parameter count is zero. Vision blocks, attention, MLPs, and patch
embedding remain frozen. A later E2 config should change only
`vision_merger_lora.enabled` plus experiment/output path identifiers. Use
`compare_experiment_configs` to enforce that boundary.

## Commands

The config uses one visible CUDA device and rejects CPU or disk offload. On a 24 GB
RTX 3090, run the smoke command first; actual fit depends on the installed CUDA,
bitsandbytes, and allocator state.

```bash
SMOKE=1 bash scripts/run_qwen35_9b_stage_pair_v2_text_anchor.sh
```

```bash
bash scripts/run_qwen35_9b_stage_pair_v2_text_anchor.sh
```

```bash
bash scripts/run_qwen35_9b_stage_pair_v2_text_anchor_submission.sh
```

To run training, strict checkpoint verification, valid-A calibration, test inference,
and submission validation as one fail-fast pipeline, use:

```bash
bash scripts/run_qwen35_9b_stage_pair_v2_text_anchor_end_to_end.sh
```

The end-to-end wrapper starts submission inference only after every training-side
step succeeds. `TRAIN_MAX_SAMPLES` and `SUBMISSION_MAX_SAMPLES` are separate to avoid
accidentally producing a partial submission. It rejects `SMOKE=1`; smoke runs never
read test data. When launched from the desktop host, the wrapper automatically
re-enters `snu-qwen3vl-desktop:latest` with GPU access, the repository Hugging Face
cache, and read-only data. Set `USE_DOCKER=0` only when already using an equivalent
pinned Python environment.

Optional one-time lockbox evaluation:

```bash
RUN_LOCKBOX=1 bash scripts/run_qwen35_9b_stage_pair_v2_text_anchor.sh
```

Config comparison for E2:

```bash
python3 -m snu_order.qwen3vl.compare_experiment_configs \
  --base configs/exp/qwen35_9b_stage_pair_v2_text_anchor.yaml \
  --candidate configs/exp/qwen35_9b_stage_pair_v2_merger.yaml \
  --allow-path experiment.id \
  --allow-path experiment.run_id \
  --allow-path output.dir \
  --allow-path output.checkpoint_dir \
  --allow-path cache.dir \
  --allow-path vision_merger_lora.enabled
```

## Artifacts

- Run outputs: `outputs/experiments/qwen35_9b_stage_pair_v2_text_anchor/<run_id>/`
- Checkpoints: `weights/qwen35_9b_stage_pair_v2_text_anchor/<run_id>/{best,last}`
- Raw valid-A logits: `<run output>/valid_a_best/raw_stage_pair_logits.pt`
- Calibration: `<best checkpoint>/calibration.json`
- Calibration grid and comparison: `<best checkpoint>/calibration_grid.csv` and
  `<best checkpoint>/raw_vs_calibrated_comparison.json`
- Submission: `<run output>/submission/submission.csv`
- Runtime profile: `<run output>/submission/inference_profile.json`

Inference uses one Qwen/Qwen3.5-9B model and one Stage/Pair head. It does not ensemble,
vote, rerank, fuse another model, or add test-time augmentation.
