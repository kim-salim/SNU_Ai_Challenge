# SNU AI Frame Ordering

Python repository for SNU AI Challenge 2026 frame ordering.

The task input is one sentence plus four shuffled video frames. The model predicts each input frame's original temporal position in the official `answer` format.

## Model

Initial B pipeline:

```text
Sentence + 4 shuffled frames
-> frozen local SigLIP2 image/text encoder
-> image and embedding quality features
-> FrameProjector
-> PermutationRanker over all 24 frame orders
-> optional PairwiseHead auxiliary loss
-> argmax permutation
-> official answer format
```

Final inference is a single deterministic pipeline. It does not use external APIs, external data, test-time manual inspection, or ensembling.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

The local environment used while scaffolding this repository did not provide `python` or `torch`; use `python3` unless your shell already maps `python` correctly.

## Data

Place competition files under `data/raw/`:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Frame paths in CSV files should be relative to the repository root or to the CSV directory. Supported text columns include `Sentence`, `sentence`, `text`, and `caption`. Supported frame path columns include `frame0`/`frame_0`/`image0`/`image_0`/`path_0` style names for indices 0 through 3.

## Pretrained Weights

Place the local Hugging Face SigLIP2 files under:

```text
weights/pretrained/siglip2_base_224/
```

The encoder always uses `local_files_only=True`. Offline runs expect:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## Feature Extraction

```bash
python3 -m snu_order.features.extract_siglip2 \
  --config configs/exp/exp003_siglip2_quality_pair_aux.yaml \
  --split train

python3 -m snu_order.features.extract_siglip2 \
  --config configs/exp/exp003_siglip2_quality_pair_aux.yaml \
  --split valid
```

Caches are written to `data/features/siglip2_base_224/`.

## Train

```bash
python3 -m snu_order.pipeline.train \
  --config configs/exp/exp003_siglip2_quality_pair_aux.yaml
```

The best checkpoint is selected by validation Exact Match and saved under `weights/heads/`.

## Evaluate

```bash
python3 -m snu_order.pipeline.evaluate \
  --config configs/exp/exp003_siglip2_quality_pair_aux.yaml
```

Evaluation writes `valid_predictions.csv`, `valid_errors.csv`, and metrics JSON under `outputs/experiments/`.

## Inference

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m snu_order.pipeline.inference \
  --config configs/final.yaml
```

Inference reads test CSV/images on the fly, without feature cache, and writes:

```text
outputs/submissions/final_submission.csv
```

## Submission

Random baseline:

```bash
python3 -m snu_order.pipeline.random_baseline \
  --config configs/exp/exp000_random.yaml
```

Validate a submission:

```bash
python3 -m snu_order.data.validate_submission \
  --file outputs/submissions/final_submission.csv \
  --reference data/raw/sample_submission.csv
```

## Offline Check

```bash
bash scripts/verify_offline.sh
```

This verifies offline environment variables and the presence of local pretrained model files.

## Tests

```bash
make test
```

On environments without PyTorch, model shape tests are skipped. Core answer conversion, permutation, metrics, submission, and feature cache tests still run.

## Environment Record

Record final environment details before submission:

```bash
python3 --version
python3 -m pip freeze > outputs/experiments/final_pip_freeze.txt
nvidia-smi > outputs/experiments/final_nvidia_smi.txt
```
