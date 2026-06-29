import pytest

from snu_order.data.metric import exact_match_accuracy, pairwise_accuracy, top1_top2_margin


def test_exact_match_accuracy():
    pred = [[1, 4, 2, 3], [1, 2, 3, 4]]
    true = [[1, 4, 2, 3], [4, 3, 2, 1]]
    assert exact_match_accuracy(pred, true) == 0.5


def test_pairwise_accuracy():
    assert pairwise_accuracy([[1, 4, 2, 3]], [[1, 4, 2, 3]]) == 1.0
    assert 0.0 <= pairwise_accuracy([[1, 2, 3, 4]], [[4, 3, 2, 1]]) <= 1.0


def test_metric_length_mismatch():
    with pytest.raises(ValueError):
        exact_match_accuracy([[1, 2, 3, 4]], [])


def test_top1_top2_margin():
    assert top1_top2_margin([[0.0] * 23 + [2.0]]) == 2.0

