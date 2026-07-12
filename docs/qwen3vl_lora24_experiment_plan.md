# Qwen3-VL LoRA24 Experiment Plan

## Purpose

The zero-shot 24-candidate Qwen3-VL run reached about 16% exact match on both valid_a and valid_b. This experiment keeps the same visual-language backbone and input ordering but replaces option-label logprob scoring with a trained single 24-way classifier.

## Phases

Phase A: analyze valid_a zero-shot raw scores only.

Phase B: train a frozen Qwen representation probe with only the classifier head trainable.

Phase C: run a 64-sample overfit check. Full training should not start if train exact match stays below 0.90.

Phase D: run a subset QLoRA experiment.

Phase E: train on full train_ab with valid_a early stopping.

Phase F: evaluate valid_b only with `--unlock-valid-b` and only if valid_a best exact match is at least 0.33.

Phase G: write the valid_b 30% gate result.

## Prompt

The classifier prompt maps images to labels explicitly:

```text
Frame A is the first supplied image.
Frame B is the second supplied image.
Frame C is the third supplied image.
Frame D is the fourth supplied image.
Sentence: ...
ORDER:
```

The four PIL images are passed to Qwen3 in the same order as Frame A, B, C, D.

## Frame Permutation Augmentation

Training samples may be reshuffled by a uniformly sampled permutation. If `shuffle_idx` maps each new input slot to an old input frame index, then:

```python
new_frames = [old_frames[i] for i in shuffle_idx]
new_answer = [old_answer[i] for i in shuffle_idx]
```

The answer values are not reranked. Validation never uses this augmentation.

## Structured Loss

The model outputs one `[B,24]` logit tensor. Pairwise and position losses are marginal likelihoods over the same 24 classes. No independent pairwise head is used, so every prediction remains a valid global permutation.

## Lockbox Policy

valid_b is not run by the trainer or pipeline. `evaluate_lockbox.py` requires `--unlock-valid-b`. It also checks the best checkpoint's valid_a metric and blocks by default below 0.33. The result is written to:

```text
outputs/experiments/qwen3vl_8b_lora24/lockbox_gate.json
```

## Resume

Use:

```bash
python3 -m snu_order.qwen3vl.train_lora24 \
  --config configs/exp/qwen3vl_8b_lora24.yaml \
  --mode full \
  --resume weights/qwen3vl_lora24/last
```

## Competition Notes

This is a single open-source local model pipeline. It does not use commercial APIs, external training data, test feedback, model ensembles, or SigLIP predictions at Qwen inference time.
