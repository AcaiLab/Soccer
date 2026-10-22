import numpy as np
from drop_halves import keep_mask


def test_only_the_named_halves_are_dropped():
    games = np.array(["07", "07", "08", "08"])
    halves = np.array(["1", "2", "1", "2"])
    assert keep_mask(games, halves, {("07", "2"), ("08", "1")}).tolist() == [True, False, False, True]


def test_nothing_dropped_without_halves():
    assert keep_mask(np.array(["07"]), np.array(["1"]), set()).tolist() == [True]
