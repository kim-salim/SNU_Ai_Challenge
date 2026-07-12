import torch

from snu_order.qwen3vl.permutations import answer_to_perm_index, perm_index_to_answer
from snu_order.qwen3vl.stage_pair_scorer import class_position_table, stage_targets_from_answer


def test_answer_to_stage_targets():
    answer = torch.tensor([[2, 4, 1, 3]])
    assert stage_targets_from_answer(answer).tolist() == [[1, 3, 0, 2]]


def test_perm_class_position_round_trip():
    for idx in range(24):
        answer = perm_index_to_answer(idx)
        table_answer = (class_position_table()[idx] + 1).tolist()
        assert table_answer == answer
        assert answer_to_perm_index(answer) == idx


def test_invalid_stage_answer_rejected():
    try:
        stage_targets_from_answer(torch.tensor([[1, 2, 3, 5]]))
    except ValueError:
        return
    raise AssertionError("invalid answer should raise")
