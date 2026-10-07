import numpy as np
import pytest

from soccer.features import (
    center_per_group,
    last_frame,
    load_temporal,
    max_pool,
    mean_pool,
)
from soccer.metrics import classification_scores, cluster_purity, random_purity


def padded():
    feats = np.zeros((2, 4, 2), dtype=np.float32)
    feats[0, :3] = [[1, 0], [0, 1], [1, 0]]      # 3 real frames
    feats[1, :1] = [[0, 1]]                     # 1 real frame
    feats[0, 3] = feats[1, 1:] = [9, 9]         # junk in padding must be ignored
    masks = np.array([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=bool)
    return feats, masks


def test_mean_pool_ignores_padding():
    feats, masks = padded()
    out = mean_pool(feats, masks)
    np.testing.assert_allclose(out[0], [0.894427, 0.447214], atol=1e-5)
    np.testing.assert_allclose(out[1], [0, 1], atol=1e-6)


def test_max_pool_and_last_frame_ignore_padding():
    feats, masks = padded()
    np.testing.assert_allclose(max_pool(feats, masks)[0], [0.707107, 0.707107], atol=1e-5)
    np.testing.assert_allclose(last_frame(feats, masks)[0], [1, 0])
    np.testing.assert_allclose(last_frame(feats, masks)[1], [0, 1])


def test_center_per_group_removes_group_means():
    x = np.array([[1, 0], [1, 1], [5, 5], [5, 6]], dtype=np.float32)
    groups = np.array(["a", "a", "b", "b"])
    out = center_per_group(x, groups)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(out[0], -out[1], atol=1e-5)


def test_load_temporal_normalises_game_ids(tmp_path):
    feats, masks = padded()
    np.savez_compressed(tmp_path / "clip_temporal_padded.npz", features=feats, masks=masks,
                        labels=np.array(["corner", "unknown"]), games=np.array(["0031", "7"]),
                        halves=np.array(["1", "2"]), event_tags=np.array(["00-10.000", "01-00.000"]))
    tf = load_temporal(tmp_path)
    assert list(tf.games) == ["31", "07"]
    kept = tf.drop_labels({"unknown"})
    assert len(kept) == 1 and kept.labels[0] == "corner" and kept.features.shape == (1, 4, 2)


def test_purity_and_its_random_baseline():
    labels = np.array(list("aaabbbccc"))
    perfect = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert cluster_purity(labels, perfect) == 1.0
    # Many tiny clusters are pure by construction; the shuffled baseline shows it.
    singletons = np.arange(9)
    assert cluster_purity(labels, singletons) == 1.0
    assert random_purity(labels, singletons) == 1.0
    assert random_purity(labels, perfect) < 0.8


def test_classification_scores_majority_baseline():
    s = classification_scores(y_true=["a", "a", "b", "c"], y_pred=["a", "b", "b", "c"],
                              y_train=["a", "a", "a", "b"])
    assert s["Accuracy"] == pytest.approx(0.75)
    assert s["Majority_baseline"] == pytest.approx(0.5)
    assert 0 < s["Macro_F1"] <= 1
