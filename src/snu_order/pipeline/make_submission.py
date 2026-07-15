from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections.abc import Sequence
import tempfile

from snu_order.data.submission_schema import SUBMISSION_COLUMNS
from snu_order.data.validate_submission import read_reference_ids, validate_submission
from snu_order.order.answer_convert import validate_answer


def answer_to_cell(answer: Sequence[int]) -> str:
    # Match the competition's canonical Answer strings exactly.
    return json.dumps(validate_answer(answer))


def save_submission(
    ids: Sequence[str],
    answers: Sequence[Sequence[int]],
    path: str | Path,
    *,
    reference: str | Path | None = None,
) -> dict[str, object]:
    if len(ids) != len(answers):
        raise ValueError(f"ids and answers length mismatch: {len(ids)} vs {len(answers)}")
    rows = [
        {SUBMISSION_COLUMNS[0]: str(sample_id), SUBMISSION_COLUMNS[1]: answer_to_cell(answer)}
        for sample_id, answer in zip(ids, answers, strict=True)
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SUBMISSION_COLUMNS), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        report = validate_submission(temp_path, reference)
        os.replace(temp_path, output)
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return {**report, "path": str(output)}
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--answer", default="[1,2,3,4]")
    args = parser.parse_args()
    ids = read_reference_ids(args.reference)
    import ast

    answer = validate_answer(ast.literal_eval(args.answer))
    save_submission(ids, [answer] * len(ids), args.output, reference=args.reference)
    print(f"saved submission: {args.output}")


if __name__ == "__main__":
    main()
