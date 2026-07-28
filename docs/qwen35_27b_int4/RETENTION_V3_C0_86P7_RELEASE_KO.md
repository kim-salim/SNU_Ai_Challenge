# Qwen3.5-27B Retention v3 C0 86.7 Release

## Status

- 역할: 현재 최고 점수 보호 모델의 소스 및 실행 계약
- 사용자 보고 leaderboard score: 86.7
- 실험 ID: `qwen35_27b_retention_v3_c0_exact_control_20260720`
- Retention/KD: 비활성화
- Test submission 파일과 모델 weight: Git 저장소에 포함하지 않음

## Submission Identity

- submission SHA-256: `8574e06aa94ba361a383cf869d6f77b9786ea8db4213fe701ebbdee5095b345b`
- checkpoint manifest SHA-256: `e0e633f7b822719784cee1416ae11109ab97d47ffeaa7989c79145564dc5480e`
- adapter SHA-256: `6ffd945eb0c5d46a0f809e9ce04350a8ad805b345b3dd897e6c93ce4f6167d16`
- heads SHA-256: `54c04b55c9a23b7df197c4a0f1c6be82310fb312fb796c2e858a3ba7178c691d`

Leaderboard score는 사용자가 제출 후 보고한 값이다. 저장소는 submission identity와 실행 계약을 기록하지만 leaderboard 결과를 독립적으로 검증하지 않는다.

## Model Contract

- Qwen3.5-27B revision: `fc05daec18b0a78c049392ed2e771dde82bdf654`
- quantization: NF4 INT4, double quantization, BF16 compute
- Full Attention LoRA: rank 16, alpha 32
- Linear Attention/DeltaNet LoRA: rank 8, alpha 16
- vision encoder and merger: frozen
- frame representation: `STATE:` anchor-span mean pooling
- frame projector: 5120 to 512
- position-free Set Encoder, Stage Head, anti-symmetric Pair Head
- scorer: Stage weight 1.0, Pair weight 0.3
- split: `full_train_90_10_v1`, Train 8581 / Valid 954
- training: seed 42, 3 epochs, first 10% head/projector warmup

## Winning Inference Contract

- calibration: none
- frame chunk size: 1
- Stage/Pair logits are detached and moved to CPU float32
- the canonical 24-permutation scorer is applied once
- exact score ties use ascending permutation class index
- submission answer, debug prediction, top-2, and margin use the same stable ranking

The raw inference contract is part of the winning artifact. It must not be replaced by GPU recomposition, calibrated logits, a different frame chunk size, or a separate `argmax` path without a new experiment identity.

## Validation Snapshot

- Exact: 507 / 954
- MRR: 0.642408
- Top-3: 0.696017
- Stage accuracy: 0.666405
- Pair accuracy: 0.807827

The historical C0 training-loop evaluation reported 503 while standalone canonical evaluation reported 507. New comparisons must use the standalone canonical CPU-float32 path until that runtime difference is fully resolved.

## Artifact Policy

Git tracks source, configs, tests, scripts, and this identity document only. Base weights, LoRA/checkpoint weights, raw data, cached logits, API artifacts, and submissions remain external artifacts and must be verified against the hashes above.
