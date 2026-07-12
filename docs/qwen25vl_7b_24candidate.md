# Qwen2.5-VL 7B 24-Candidate Pipeline

## Purpose

This pipeline evaluates SNU AI frame ordering samples directly from raw images with a single local VLM:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- no SigLIP feature cache dependency
- no ensemble
- no external commercial API
- no test-set manual inspection

The model receives the sentence and four shuffled frames, then selects one of the 24 possible chronological frame orders.

## Model

Default config:

```text
configs/qwen25vl_7b_24candidate.yaml
```

The config uses local files only by default:

```yaml
model:
  name: Qwen/Qwen2.5-VL-7B-Instruct
  local_files_only: true
  torch_dtype: bfloat16
  device_map: auto

quantization:
  enabled: true
  load_in_4bit: true
```

The model weights must already exist in the local Hugging Face cache before offline inference.

## Prompt Format

The prompt is intentionally short and does not ask for chain-of-thought:

```text
You are solving a video frame ordering task.

The sentence describes the original chronological event.
The four frames are shuffled and labeled F1, F2, F3, F4 in the same order as the provided images.

Choose the option that lists the frames from earliest to latest.

Return exactly one option letter from A to X.
Do not explain.

Sentence:
...

Options:
A: F1 F2 F3 F4
...
X: F4 F3 F2 F1

Answer:
```

## Candidate Mapping

Candidates are fixed lexicographic permutations of input frame indices `[0,1,2,3]`.

Example:

```text
A: F1 F2 F3 F4 -> order (0,1,2,3) -> answer [1,2,3,4]
```

Official answer format stores each input frame's chronological position.

Example:

```text
order (0,2,3,1) -> answer [1,4,2,3]
```

## Scoring Method

Primary scoring mode is `option_label_logprob`.

For each sample:

1. Build one prompt with all 24 options.
2. Run one forward pass for the prompt and images.
3. Score labels `A` to `X` using next-token/sequence log probabilities.
4. Score label variants:
   - `A`
   - ` A`
   - `A:`
   - ` A:`
5. Use the maximum variant score per option.
6. Choose the highest scoring option.

Direct generation is implemented only as fallback.

## Image Modes

Default:

```text
multi_image
```

The model receives four PIL images in frame order `F1` to `F4`.

Fallback:

```text
grid_2x2
```

The pipeline builds one deterministic 2x2 labeled grid image.

## Evaluation Commands

300-sample smoke runs:

```bash
bash scripts/run_qwen25vl7b_24candidate_valid_a_300.sh
bash scripts/run_qwen25vl7b_24candidate_valid_b_300.sh
```

Full A/B validation:

```bash
bash scripts/run_qwen25vl7b_24candidate_valid_a_full.sh
bash scripts/run_qwen25vl7b_24candidate_valid_b_full.sh
```

Direct module command:

```bash
python -m snu_order.vlm24.eval \
  --config configs/qwen25vl_7b_24candidate.yaml \
  --metadata-csv data/splits/ab_v1/valid_a_v1.csv \
  --image-root data/raw \
  --output-dir outputs/predictions/qwen25vl_7b_24candidate/valid_a_300 \
  --max-samples 300 \
  --image-mode multi_image \
  --scoring-mode option_label_logprob \
  --benchmark
```

## Inference Command

```bash
bash scripts/run_qwen25vl7b_24candidate_test.sh
```

Direct module command:

```bash
python -m snu_order.vlm24.inference \
  --config configs/qwen25vl_7b_24candidate.yaml \
  --metadata-csv data/raw/test.csv \
  --image-root data/raw \
  --sample-submission data/raw/sample_submission.csv \
  --output-csv outputs/predictions/qwen25vl_7b_24candidate/submission.csv \
  --image-mode multi_image \
  --scoring-mode option_label_logprob
```

## Benchmark

```bash
python -m snu_order.vlm24.benchmark \
  --config configs/qwen25vl_7b_24candidate.yaml \
  --metadata-csv data/splits/ab_v1/valid_a_v1.csv \
  --image-root data/raw \
  --max-samples 50
```

The benchmark reports:

- model load time
- average seconds per sample
- p50/p90 latency
- max VRAM
- estimated full test time
- whether the estimate is under 24 hours

## Competition-Rule Notes

- Single model only.
- No ensemble logic is introduced.
- No SigLIP prediction combination at inference.
- No external commercial API.
- No external training data.
- Qwen weights must be downloaded locally before offline inference.
- Do not inspect test images manually.
- Do not alter preprocessing based on test-set characteristics.

## Known Limitations

- Logprob scoring can be sensitive to tokenizer variants; variants are configurable.
- Multi-image and grid modes may differ in model behavior.
- RTX 3090 24GB should use 4-bit quantization.
- The 24-hour inference limit must be checked with `benchmark.py` before full test inference.
- LoRA placeholders exist in config, but LoRA training is intentionally disabled in this first pass.
