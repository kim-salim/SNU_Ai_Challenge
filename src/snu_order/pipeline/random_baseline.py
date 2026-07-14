from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from snu_order.data.validate_submission import read_reference_ids, validate_submission
from snu_order.order.answer_convert import perm_to_answer
from snu_order.order.permutation24 import get_all_perms
from snu_order.pipeline.make_submission import save_submission
from snu_order.utils.config import get_by_path, load_config
from snu_order.utils.seed import seed_everything


def make_random_answers(num_samples: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    perms = get_all_perms()
    indices = rng.integers(0, len(perms), size=num_samples)
    return [perm_to_answer(perms[int(idx)]) for idx in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(get_by_path(cfg, "experiment.seed", 42))
    seed_everything(seed)
    reference = get_by_path(cfg, "data.sample_submission_csv") or get_by_path(cfg, "data.test_csv")
    if reference is None:
        raise ValueError("Config must set data.sample_submission_csv or data.test_csv")
    output = Path(get_by_path(cfg, "output.submission", "outputs/submissions/exp000_random.csv"))

    ids = read_reference_ids(reference)
    answers = make_random_answers(len(ids), seed)
    save_submission(ids, answers, output, reference=reference)
    validate_submission(output, reference)
    print(f"saved random baseline: {output}")


if __name__ == "__main__":
    main()
