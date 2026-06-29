# Model Registry

## Pretrained Encoder

- Name: `google/siglip2-base-patch16-224`
- Local path: `weights/pretrained/siglip2_base_224/`
- Use: frozen image/text embedding extraction
- Loading: Hugging Face `AutoModel` and `AutoProcessor` with `local_files_only=True`
- Public weight date check: record source evidence before final submission

## Trainable Head

- FrameProjector: MLP from text/frame embedding interactions plus quality features to frame tokens
- PermutationRanker: MLP scorer over all 24 permutations
- PairwiseHead: auxiliary pair ordering classifier over 6 fixed pairs
- Final checkpoint: `weights/heads/final_ranker.pt`

