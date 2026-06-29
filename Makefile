PYTHON ?= python3
CFG ?= configs/exp/exp003_siglip2_quality_pair_aux.yaml
FILE ?= outputs/submissions/final_submission.csv

.PHONY: setup test split extract train eval infer validate-submission offline-check

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests -q

split:
	$(PYTHON) -m snu_order.data.split --config configs/exp/exp000_random.yaml

extract:
	$(PYTHON) -m snu_order.features.extract_siglip2 --config $(CFG)

train:
	$(PYTHON) -m snu_order.pipeline.train --config $(CFG)

eval:
	$(PYTHON) -m snu_order.pipeline.evaluate --config $(CFG)

infer:
	$(PYTHON) -m snu_order.pipeline.inference --config $(CFG)

validate-submission:
	$(PYTHON) -m snu_order.data.validate_submission --file $(FILE)

offline-check:
	bash scripts/verify_offline.sh
