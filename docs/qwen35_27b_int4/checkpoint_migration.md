# Checkpoint Migration

`scripts/qwen35_27b_int4/migrate_champion_heads.py`는 Champion `heads.pt`만 읽는다.

이식 대상:

- `set_encoder.encoder.*`
- `set_encoder.output_norm.*`
- `stage_head.*`
- `pair_head.*`

Fresh 대상:

- `frame_projector.input_norm.*`
- `frame_projector.proj.*`

거부 대상:

- 9B `set_encoder.input_norm.*`
- 9B `set_encoder.proj.*`
- 모든 9B LoRA tensor
- 9B calibration scalar

모든 destination key는 exact set equality와 `strict=True`로 검증한다. `strict=False`, reshape, interpolation, repeat는 사용하지 않는다. 실제 migration report는 `artifacts/qwen35_27b_int4_port/02_implementation/migration_report.json`에 기록된다.

Checkpoint format v3는 architecture, local base locator/revision, backbone config hash, 5120/64/16/48 구조, quantization, package version, processor/tokenizer, prompt/anchor, image/text policy, permutation, adapter/head/calibration, Git HEAD/TREE를 binding한다. 기존 format v2 loader는 변경 없이 별도 분기로 유지한다.

