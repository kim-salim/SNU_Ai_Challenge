import pytest

from snu_order.vlm24.candidates import answer_to_order, order_to_answer, validate_answer


def test_answer_conversion() -> None:
    assert order_to_answer((0, 2, 3, 1)) == [1, 4, 2, 3]
    assert answer_to_order([1, 4, 2, 3]) == (0, 2, 3, 1)


def test_validate_answer() -> None:
    assert validate_answer([4, 3, 2, 1]) == [4, 3, 2, 1]
    with pytest.raises(ValueError):
        validate_answer([1, 2, 3])
    with pytest.raises(ValueError):
        validate_answer([1, 1, 2, 3])
