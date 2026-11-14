import pytest

from src.config import TARGET, TIME_COL
from src.splits import assert_chronological, time_split


def test_time_split_disjoint_and_complete(df):
    parts = time_split(df)
    idx = [set(X.index) for X, _ in parts.values()]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert sum(len(i) for i in idx) == len(df)


def test_time_split_is_chronological(df):
    parts = time_split(df)
    tr, va, te = (parts[k][0][TIME_COL] for k in ("train", "valid", "test"))
    assert tr.max() < va.min() and va.max() < te.min()


def test_target_not_in_features(df):
    for X, y in time_split(df).values():
        assert TARGET not in X.columns and len(X) == len(y)


def test_overlapping_or_unordered_months_rejected():
    with pytest.raises(ValueError):
        assert_chronological([0, 1, 2], [2, 3], [4])
    with pytest.raises(ValueError):
        assert_chronological([0, 1, 5], [3], [6])
