import numpy as np
import pytest

from soccer.features import (
    center_per_group,
    last_frame,
    load_temporal,
    max_pool,
    mean_pool,
)
from soccer.metrics import (
    bootstrap_ci,
    classification_scores,
    cluster_purity,
    game_vs_event_scores,
    random_purity,
)


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


def test_game_vs_event_scores_detect_clustering_by_game():
    games = np.repeat(["01", "02", "03"], 4)
    labels = np.tile(["corner", "goal"], 6)
    by_game = np.repeat([0, 1, 2], 4)  # clusters follow the match, not the event
    s = game_vs_event_scores(labels, games, by_game)
    assert s["Game_ARI"] == pytest.approx(1.0) and s["Game_NMI"] == pytest.approx(1.0)
    assert s["Event_NMI"] == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_ci_resamples_whole_games():
    games = np.repeat(["01", "02", "03", "04"], 25)
    y_true = np.zeros(100, dtype=int)
    y_pred = np.zeros(100, dtype=int)
    y_pred[:25] = 1  # game 01 entirely wrong, the others entirely right
    ci = bootstrap_ci(y_true, y_pred, games, n_boot=500, seed=0)
    lo, hi = ci["Accuracy_CI"]
    assert lo < 0.75 < hi and lo >= 0.0 and hi <= 1.0
    # A single game: no between-game variation, so the interval collapses.
    one = bootstrap_ci(y_true[25:50], y_pred[25:50], games[25:50], n_boot=50)
    assert one["Accuracy_CI"] == (1.0, 1.0)


def test_load_temporal_can_keep_only_manifest_games(tmp_path):
    feats, masks = padded()
    np.savez_compressed(tmp_path / "clip_temporal_padded.npz", features=feats, masks=masks,
                        labels=np.array(["corner", "goal"]), games=np.array(["01", "31"]),
                        halves=np.array(["1", "1"]), event_tags=np.array(["00-10.000", "00-20.000"]))
    tf = load_temporal(tmp_path, games=["1"])
    assert list(tf.games) == ["01"] and tf.features.shape[0] == 1
