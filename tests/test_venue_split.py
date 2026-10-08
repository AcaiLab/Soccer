from make_venue_split import assign

from soccer.manifest import assert_disjoint


def venues():
    # 10 stadiums hosting 1-4 games each (30 games in total, like the manifest)
    sizes = {"A": 4, "B": 4, "C": 4, "D": 3, "E": 3, "F": 3, "G": 3, "H": 2, "I": 2, "J": 2}
    return {v: [f"{v}{i}" for i in range(n)] for v, n in sizes.items()}


def test_every_venue_goes_to_exactly_one_split():
    split = assign(venues(), n_test=6, n_val=4, seed=42)
    assert set(split) == set(venues())
    by_split = {s: {v for v, x in split.items() if x == s} for s in ("train", "val", "test")}
    assert_disjoint(**by_split)


def test_targets_are_not_exceeded_and_train_keeps_the_rest():
    v = venues()
    split = assign(v, n_test=6, n_val=4, seed=42)
    count = {s: sum(len(v[x]) for x, y in split.items() if y == s) for s in ("train", "val", "test")}
    assert 0 < count["test"] <= 6 and 0 < count["val"] <= 4
    assert count["train"] == 30 - count["test"] - count["val"]


def test_assignment_is_reproducible():
    assert assign(venues(), 6, 4, seed=1) == assign(venues(), 6, 4, seed=1)
