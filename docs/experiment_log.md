# Experiment Log

## exp000_random

- Purpose: deterministic random permutation baseline
- Config: `configs/exp/exp000_random.yaml`
- Output: `outputs/submissions/exp000_random.csv`

## exp001_siglip2_perm_mlp

- Purpose: frozen SigLIP2 cache plus permutation ranker
- Pairwise auxiliary: disabled
- Quality features: cache contains quality, config disables quality emphasis

## exp002_siglip2_perm_pair_aux

- Purpose: add pairwise auxiliary loss
- Pairwise weight: `0.3`

## exp003_siglip2_quality_pair_aux

- Purpose: main B pipeline with quality features and pairwise auxiliary loss
- Primary metric: validation Exact Match Accuracy
- Best checkpoint path: `weights/heads/exp003_siglip2_quality_pair_aux.pt`

