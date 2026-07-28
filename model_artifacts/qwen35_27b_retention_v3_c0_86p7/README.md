# Qwen3.5-27B Retention v3 C0 86.7 Artifact

This directory contains the exact experiment-owned checkpoint files bound to
the C0 submission that recorded the user-reported leaderboard score of 0.867.

## Contents

- `checkpoint/adapter/`: QLoRA adapter configuration and weights
- `checkpoint/heads.pt`: Frame Projector, Set Encoder, Stage Head, and Pair Head
- `checkpoint/processor/`: processor and tokenizer binding
- `checkpoint/checkpoint_manifest.json`: complete checkpoint identity
- `checkpoint/config.json`: resolved training configuration
- `checkpoint/permutations.json`: canonical permutation mapping
- `checkpoint/prompt_fingerprint.json`: prompt binding
- `submission/`: exact verified submission CSV
- `SHA256SUMS`: checksums for every artifact in this directory

The Qwen3.5-27B base weights are not included. The required base revision is
`fc05daec18b0a78c049392ed2e771dde82bdf654`.

## Primary Identity

- checkpoint manifest: `e0e633f7b822719784cee1416ae11109ab97d47ffeaa7989c79145564dc5480e`
- adapter: `6ffd945eb0c5d46a0f809e9ce04350a8ad805b345b3dd897e6c93ce4f6167d16`
- heads: `54c04b55c9a23b7df197c4a0f1c6be82310fb312fb796c2e858a3ba7178c691d`
- submission: `8574e06aa94ba361a383cf869d6f77b9786ea8db4213fe701ebbdee5095b345b`

See `docs/qwen35_27b_int4/RETENTION_V3_C0_86P7_RELEASE_KO.md` for the
training and inference contract.
