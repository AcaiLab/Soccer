import numpy as np
from cluster_animation import kmeans_steps, purity
from sklearn.metrics import adjusted_rand_score


def blobs():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [0, 10]], float)
    return np.concatenate([c + rng.normal(size=(30, 2)) for c in centres]), np.repeat([0, 1, 2], 30)


def test_steps_end_when_nothing_moves_and_recover_the_blobs():
    x, truth = blobs()
    steps = kmeans_steps(x, 3, seed=0)
    assert len(steps) >= 1
    assert not any(np.array_equal(a, b) for a, b in zip(steps, steps[1:], strict=False))
    assert adjusted_rand_score(truth, steps[-1]) == 1.0


def test_steps_are_reproducible():
    x, _ = blobs()
    first, second = kmeans_steps(x, 3, seed=1), kmeans_steps(x, 3, seed=1)
    assert len(first) == len(second)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_purity():
    assert purity(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0
    assert purity(np.array([0, 1, 0, 1]), np.array([0, 0, 0, 0])) == 0.5
