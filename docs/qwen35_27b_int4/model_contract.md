# Model Contract

## 보존되는 경로

`dataset_single_frame`, `stage_pair_prompt`, `PositionFree Set Encoder`, `StageHead`, `AntiSymmetricPairwiseHead`, `structured_permutation_logits`, `PERMS`를 Champion source에서 재사용한다. Prompt는 image 다음 text, `enable_thinking=False`, `add_generation_prompt=False`, literal `STATE:` 단일 anchor, anchor-span mean pooling이다.

## 27B에서만 달라지는 부분

- Architecture ID: `qwen35_27b_stage_pair_e1_int4_v1`
- Language hidden width: 5,120
- Language layers: 64
- Gated/Full Attention: 16개, q/k/v/o LoRA r16 alpha32
- Gated DeltaNet/Linear Attention: 48개, qkv/z/out LoRA r8 alpha16
- Fresh `FrameProjector`: LayerNorm(5120) + Linear(5120, 512)
- NF4 INT4, double quantization, BF16 compute

FFN, Vision Encoder, Vision Merger에는 LoRA를 두지 않는다. Vision 계열 parameter는 frozen이다.

## Hidden-only 계약

Conditional-generation wrapper의 `lm_head` 아래에 있는 실제 multimodal core를 찾고 그 core만 호출한다. `use_cache=False`, `output_hidden_states=False`, `return_dict=True`를 강제하며 `last_hidden_state`만 받는다. `generate`, autoregressive decoding, MTP, vocabulary logits는 호출하지 않는다. core를 확정할 수 없으면 `HOLD_HIDDEN_STATE_CONTRACT_FAILURE`다.

## 단일 모델 계약

최종 package에는 base model locator 하나, adapter 하나, head checkpoint 하나, processor 하나, calibration 하나만 존재해야 한다. 9B fallback, router, ensemble, averaging, TTA는 금지한다.

