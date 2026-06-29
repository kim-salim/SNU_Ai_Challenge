from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from snu_order.data.validate_submission import read_reference_ids, validate_submission
from snu_order.order.answer_convert import validate_answer
from snu_order.utils.io import write_csv_rows


def answer_to_cell(answer: Sequence[int]) -> str:
    return json.dumps(validate_answer(answer), separators=(",", ":"))


def save_submission(ids: Sequence[str], answers: Sequence[Sequence[int]], path: str | Path) -> None:
    if len(ids) != len(answers):
        raise ValueError(f"ids and answers length mismatch: {len(ids)} vs {len(answers)}")
    rows = [{"Id": str(sample_id), "answer": answer_to_cell(answer)} for sample_id, answer in zip(ids, answers, strict=True)]
    write_csv_rows(path, rows, ["Id", "answer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--answer", default="[1,2,3,4]")
    args = parser.parse_args()
    ids = read_reference_ids(args.reference)
    import ast

    answer = validate_answer(ast.literal_eval(args.answer))
    save_submission(ids, [answer] * len(ids), args.output)
    validate_submission(args.output, args.reference)
    print(f"saved submission: {args.output}")


if __name__ == "__main__":
    main()

