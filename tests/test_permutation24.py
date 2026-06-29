import pytest

from snu_order.order.permutation24 import PERMS, get_all_perms, index_to_perm, perm_to_index


def test_permutation_count():
    assert len(PERMS) == 24
    assert len(set(PERMS)) == 24
    assert get_all_perms() == PERMS


def test_perm_contents():
    for perm in PERMS:
        assert sorted(perm) == [0, 1, 2, 3]


def test_index_round_trip():
    for index, perm in enumerate(PERMS):
        assert perm_to_index(perm) == index
        assert index_to_perm(index) == perm


def test_bad_index():
    with pytest.raises(ValueError):
        index_to_perm(24)

