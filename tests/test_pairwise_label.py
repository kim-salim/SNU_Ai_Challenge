import pytest

from snu_order.data.label import PAIRS, answer_to_pairwise_labels, answer_to_perm_index
from snu_order.order.answer_convert import answer_to_perm
from snu_order.order.permutation24 import perm_to_index


def test_pair_order():
    assert PAIRS == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def test_answer_to_pairwise_labels():
    assert answer_to_pairwise_labels([1, 4, 2, 3]) == [1, 1, 1, 0, 0, 1]


def test_answer_to_perm_index():
    answer = [1, 4, 2, 3]
    assert answer_to_perm_index(answer) == perm_to_index(answer_to_perm(answer))


def test_invalid_answer():
    with pytest.raises(ValueError):
        answer_to_pairwise_labels([1, 2, 2, 4])

