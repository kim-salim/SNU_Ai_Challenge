# Inference Profile

Fill this before final submission on the target RTX 3090 environment.

## Command

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m snu_order.pipeline.inference --config configs/final.yaml
```

## Environment

- GPU:
- Driver:
- CUDA:
- PyTorch:
- Transformers:
- Batch size:

## Timing

- Number of test samples:
- Total runtime:
- Samples per second:
- Peak GPU memory:

## Notes

- Inference uses one SigLIP2 encoder and one trained ranking head.
- No feature cache is required for final test inference.
- No ensemble or external API call is used.

