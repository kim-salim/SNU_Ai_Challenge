import pytest

from snu_order.order.answer_convert import answer_to_perm, perm_to_answer


def test_answer_to_perm():
    assert answer_to_perm([1, 4, 2, 3]) == (0, 2, 3, 1)
    assert answer_to_perm([1, 2, 3, 4]) == (0, 1, 2, 3)
    assert answer_to_perm([4, 3, 2, 1]) == (3, 2, 1, 0)


def test_perm_to_answer():
    assert perm_to_answer((0, 2, 3, 1)) == [1, 4, 2, 3]
    assert perm_to_answer((0, 1, 2, 3)) == [1, 2, 3, 4]
    assert perm_to_answer((3, 2, 1, 0)) == [4, 3, 2, 1]


def test_round_trip():
    answers = [
        [1, 4, 2, 3],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 1, 4, 3],
    ]
    for ans in answers:
        assert perm_to_answer(answer_to_perm(ans)) == ans


def test_invalid_inputs_raise_value_error():
    with pytest.raises(ValueError):
        answer_to_perm([1, 2, 3, 3])
    with pytest.raises(ValueError):
        perm_to_answer([0, 1, 2, 2])

