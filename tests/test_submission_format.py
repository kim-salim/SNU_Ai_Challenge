from pathlib import Path

import pytest

from snu_order.data.submission_schema import SUBMISSION_COLUMNS
from snu_order.data.harden_submission import repair_submission_header_only
from snu_order.data.validate_submission import parse_answer, validate_submission
from snu_order.pipeline.make_submission import save_submission
from snu_order.pipeline.random_baseline import make_random_answers


def _reference(path: Path, ids: tuple[str, ...] = ("A", "B")) -> Path:
    rows = "".join(f'{sample_id},"[1,2,3,4]"\n' for sample_id in ids)
    path.write_text("Id,Answer\n" + rows, encoding="utf-8")
    return path


def test_parse_answer_variants():
    assert parse_answer("[1,4,2,3]") == [1, 4, 2, 3]
    assert parse_answer("1 4 2 3") == [1, 4, 2, 3]
    assert parse_answer([1, 4, 2, 3]) == [1, 4, 2, 3]


def test_save_is_atomic_and_uses_exact_schema(tmp_path: Path):
    reference = _reference(tmp_path / "sample_submission.csv")
    submission = tmp_path / "submission.csv"
    report = save_submission(
        ["A", "B"],
        [[1, 4, 2, 3], [4, 3, 2, 1]],
        submission,
        reference=reference,
    )
    assert submission.read_bytes().splitlines()[0] == b"Id,Answer"
    assert report["schema"] == list(SUBMISSION_COLUMNS)
    assert report["row_count"] == 2
    assert not list(tmp_path.glob(".submission.csv.tmp-*"))


@pytest.mark.parametrize(
    "header",
    [
        "Id,answer",
        "id,Answer",
        "ID,ANSWER",
        "Answer,Id",
        "Id,Answer,extra",
        "Unnamed: 0,Id,Answer",
    ],
)
def test_validate_rejects_noncanonical_header_case_order_or_extra(tmp_path: Path, header: str):
    reference = _reference(tmp_path / "sample_submission.csv", ("A",))
    submission = tmp_path / "bad.csv"
    width = len(header.split(","))
    submission.write_text(header + "\n" + ",".join(["A", '"[1,2,3,4]"'] + [""] * (width - 2)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header must be exactly"):
        validate_submission(submission, reference)


@pytest.mark.parametrize(
    "payload, message",
    [
        ('Id,Answer\nA,"[1,2,3,4]"\nA,"[4,3,2,1]"\n', "unique"),
        ('Id,Answer\nA,"[1,2,3,4]"\n', "row count"),
        ('Id,Answer\nB,"[1,2,3,4]"\nA,"[4,3,2,1]"\n', "order"),
        ('Id,Answer\nA,\nB,"[4,3,2,1]"\n', "empty"),
        ('Id,Answer\nA," [1,2,3,4]"\nB,"[4,3,2,1]"\n', "whitespace"),
        ('Id,Answer\nA,"[1,2,2,4]"\nB,"[4,3,2,1]"\n', "Invalid Answer"),
    ],
)
def test_validate_rejects_invalid_ids_or_answers(tmp_path: Path, payload: str, message: str):
    reference = _reference(tmp_path / "sample_submission.csv")
    submission = tmp_path / "bad.csv"
    submission.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_submission(submission, reference)


def test_validate_rejects_bom_in_submission_and_reference(tmp_path: Path):
    reference = _reference(tmp_path / "sample_submission.csv", ("A",))
    submission = tmp_path / "bad.csv"
    submission.write_bytes(b"\xef\xbb\xbfId,Answer\nA,\"[1,2,3,4]\"\n")
    with pytest.raises(ValueError, match="BOM"):
        validate_submission(submission, reference)

    clean_submission = tmp_path / "clean.csv"
    clean_submission.write_text('Id,Answer\nA,"[1,2,3,4]"\n', encoding="utf-8")
    reference.write_bytes(b"\xef\xbb\xbfId,Answer\nA,\"[1,2,3,4]\"\n")
    with pytest.raises(ValueError, match="BOM"):
        validate_submission(clean_submission, reference)


def test_header_only_repair_preserves_every_data_byte(tmp_path: Path):
    reference = _reference(tmp_path / "sample_submission.csv")
    original = tmp_path / "source.csv"
    original.write_bytes(b'Id,answer\r\nA,"[1,2,3,4]"\r\nB,"[4,3,2,1]"\r\n')
    checkpoint = tmp_path / "checkpoint_manifest.json"
    calibration = tmp_path / "calibration.json"
    scorer = tmp_path / "scorer.py"
    checkpoint.write_text("{}\n", encoding="utf-8")
    calibration.write_text("{}\n", encoding="utf-8")
    scorer.write_text("# scorer\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    manifest = repair_submission_header_only(
        original,
        reference,
        tmp_path / "hardened",
        checkpoint_manifest=checkpoint,
        calibration_artifact=calibration,
        scorer_code=scorer,
        repo_root=repo_root,
        expected_rows=2,
    )
    before = (tmp_path / "hardened" / "submission.pre_schema_fix.csv").read_bytes()
    after = (tmp_path / "hardened" / "submission.csv").read_bytes()
    assert before.splitlines(keepends=True)[1:] == after.splitlines(keepends=True)[1:]
    assert after.startswith(b"Id,Answer\r\n")
    assert manifest["data_rows_byte_identical"] is True


def test_random_baseline_deterministic():
    assert make_random_answers(5, seed=123) == make_random_answers(5, seed=123)
