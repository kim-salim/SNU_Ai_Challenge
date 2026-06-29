from pathlib import Path

import pytest

from snu_order.data.validate_submission import parse_answer, validate_submission
from snu_order.pipeline.make_submission import save_submission
from snu_order.pipeline.random_baseline import make_random_answers


def test_parse_answer_variants():
    assert parse_answer("[1,4,2,3]") == [1, 4, 2, 3]
    assert parse_answer("1 4 2 3") == [1, 4, 2, 3]
    assert parse_answer([1, 4, 2, 3]) == [1, 4, 2, 3]


def test_save_and_validate_submission(tmp_path: Path):
    reference = tmp_path / "sample_submission.csv"
    reference.write_text("Id,answer\nA,\"[1,2,3,4]\"\nB,\"[1,2,3,4]\"\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"
    save_submission(["A", "B"], [[1, 4, 2, 3], [4, 3, 2, 1]], submission)
    validate_submission(submission, reference)


def test_validate_rejects_bad_answer(tmp_path: Path):
    reference = tmp_path / "sample_submission.csv"
    reference.write_text("Id,answer\nA,\"[1,2,3,4]\"\n", encoding="utf-8")
    submission = tmp_path / "bad.csv"
    submission.write_text("Id,answer\nA,\"[1,2,2,4]\"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_submission(submission, reference)


def test_random_baseline_deterministic():
    assert make_random_answers(5, seed=123) == make_random_answers(5, seed=123)

