# Qwen3.5 Stage E1 Canonical 84.2 Bundle

This directory preserves the exact E1 checkpoint associated with
`VERIFIED__SUBMIT_THIS_Qwen35_STAGE_E1_CANONICAL_0715.csv`.

- Base model: `Qwen/Qwen3.5-9B`
- Base revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Checkpoint manifest SHA-256: `49f014c4f7f957cf3757dcae8099fd5a43adf71a60c952d9eeb40cd591b22039`
- Adapter SHA-256: `0496cdad78d71ce604755a290e51b55332daaa85c5e44f427ac3d2ae95742b7f`
- Heads SHA-256: `4982890f1de2ec93bcad0e9251b49aa5767935b894955859adf0ae70f995c83a`
- Prompt fingerprint SHA-256: `ceebb96cf0f76eb7349f3a9ec8f2efb84dcb31628d338540936d690079ef69cc`
- Submission SHA-256: `4787d1fb0f1ebe8621387add7c2ab42765d5d8d2a79dfa25f7222d84c1245117`

The 9B base weights are intentionally not duplicated. Load the pinned base
revision, then load `checkpoint/adapter` and `checkpoint/heads.pt` through the
repository's strict stage-pair checkpoint loader. The bundled processor, prompt
fingerprint, permutation mapping, calibration artifact, and manifest are part of
the inference contract and must not be substituted.

`checksums.sha256` covers every file in this release directory except itself.
