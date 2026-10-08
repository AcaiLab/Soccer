"""Clustering and classification metrics with their chance baselines.

A purity or accuracy number means little without the score a trivial method
gets on the same data, so each metric here comes with its baseline.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score


def cluster_purity(labels: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Share of items whose cluster's majority label is their own label."""
    labels, cluster_ids = np.asarray(labels), np.asarray(cluster_ids)
    correct = sum(Counter(labels[cluster_ids == c]).most_common(1)[0][1]
                  for c in np.unique(cluster_ids))
    return correct / len(labels)


def random_purity(labels: np.ndarray, cluster_ids: np.ndarray,
                  n_repeats: int = 200, seed: int = 42) -> float:
    """Expected purity if items were shuffled between the same clusters.

    Keeps the cluster sizes, so it is the right baseline for purity, which
    rises on its own when clusters are small.
    """
    rng = np.random.default_rng(seed)
    cluster_ids = np.asarray(cluster_ids)
    return float(np.mean([cluster_purity(labels, rng.permutation(cluster_ids))
                          for _ in range(n_repeats)]))


def clustering_scores(labels: np.ndarray, cluster_ids: np.ndarray, seed: int = 42) -> dict:
    return {
        "Purity": cluster_purity(labels, cluster_ids),
        "Purity_random": random_purity(labels, cluster_ids, seed=seed),
        "ARI": adjusted_rand_score(labels, cluster_ids),
    }


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict:
    """Accuracy and macro-F1, plus the always-predict-the-commonest-training-class baseline."""
    majority = Counter(np.asarray(y_train).tolist()).most_common(1)[0][0]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Majority_baseline": float(np.mean(np.asarray(y_true) == majority)),
    }


def game_vs_event_scores(labels: np.ndarray, games: np.ndarray, cluster_ids: np.ndarray) -> dict:
    """How strongly a clustering follows event type versus game identity.

    The project's central claim is that frozen embeddings group events by
    match rather than by event; Game_ARI much larger than ARI is that claim.
    """
    return {
        "Event_NMI": normalized_mutual_info_score(labels, cluster_ids),
        "Game_ARI": adjusted_rand_score(games, cluster_ids),
        "Game_NMI": normalized_mutual_info_score(games, cluster_ids),
    }


def bootstrap_ci(y_true, y_pred, groups, n_boot: int = 1000, seed: int = 42, level: float = 0.95) -> dict:
    """Confidence intervals for accuracy and macro-F1, resampling whole games.

    Events of one match are correlated (same stadium, same broadcast), so
    resampling individual events would give intervals that are too narrow.
    """
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in unique}
    acc, f1 = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        acc.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"Accuracy_CI": (float(np.percentile(acc, lo)), float(np.percentile(acc, hi))),
            "Macro_F1_CI": (float(np.percentile(f1, lo)), float(np.percentile(f1, hi)))}
