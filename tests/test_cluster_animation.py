import numpy as np
from cluster_animation import kmeans_steps, purity, window_masks
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


def test_window_keeps_frames_near_the_event_in_tensor_order():
    masks = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], bool)
    offsets = [[-6, -2, 3, 7], [-20, 20], None]
    out, kept_all = window_masks(masks, offsets, window=5)
    assert out[0].tolist() == [False, True, True, False, False]
    assert out[1].tolist() == masks[1].tolist()  # nothing within 5 s: all frames kept
    assert out[2].tolist() == masks[2].tolist()  # unknown times: all frames kept
    assert kept_all == 2


def test_window_ignores_offsets_that_do_not_match_the_frame_count():
    masks = np.array([[1, 1, 1, 0]], bool)
    out, kept_all = window_masks(masks, [[-1, 0]], window=5)
    assert out.tolist() == masks.tolist() and kept_all == 1


def test_purity():
    assert purity(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0
    assert purity(np.array([0, 1, 0, 1]), np.array([0, 0, 0, 0])) == 0.5
