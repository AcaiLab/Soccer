"""
Record K-Means forming its clusters, step by step, for an animated view.

Uses the same event vectors as cluster_analysis.py (B: mask-aware mean-pool of
each event's frames, dropped labels removed) and the same k (one cluster per
event type). K-Means is run by hand from a seeded k-means++ start so every
assignment step can be saved; each step also gets purity and ARI against the
event labels and ARI against the match, the numbers the analysis reports.

The 2-D map is one t-SNE of the event vectors, computed once: points stay put
and only their cluster colour changes from step to step.

Writes a single JSON file for the visualisation page.

Usage:
    python scripts/cluster_animation.py --out results/cluster_animation.json
    python scripts/cluster_animation.py --features-dir runs/epl1720/features/temporal_dinov2-base --out anim.json
"""

import argparse
import json
import pathlib

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from soccer.config import data_path, dropped_labels, load_config
from soccer.features import load_temporal, mean_pool
from soccer.manifest import manifest_games

MAX_STEPS = 300


def kmeans_steps(x: np.ndarray, k: int, seed: int, max_steps: int = MAX_STEPS) -> list[np.ndarray]:
    """Cluster assignments after the k-means++ start and after each Lloyd update, until nothing moves."""
    centres, _ = kmeans_plusplus(x, k, random_state=seed)
    steps = []
    for _ in range(max_steps):
        dist = (centres ** 2).sum(1)[None, :] - 2 * x @ centres.T  # |x|^2 is the same for every centre
        assign = dist.argmin(1)
        if steps and np.array_equal(assign, steps[-1]):
            break
        steps.append(assign)
        for c in range(k):
            members = x[assign == c]
            if len(members):  # an emptied cluster keeps its old centre
                centres[c] = members.mean(0)
    return steps


def purity(true_ids: np.ndarray, cluster_ids: np.ndarray) -> float:
    return sum(np.bincount(true_ids[cluster_ids == c]).max() for c in np.unique(cluster_ids)) / len(true_ids)


def main(argv=None):
    cfg = load_config()
    prefix = cfg["features"]["model"].split("/")[-1]
    parser = argparse.ArgumentParser(description="Save K-Means steps and a t-SNE map for an animated view.")
    parser.add_argument("--features-dir", default=None, help="default: features/temporal_<model name>")
    parser.add_argument("--out", required=True, help="JSON file to write")
    parser.add_argument("--title", default="", help="shown on the page")
    args = parser.parse_args(argv)

    seed = cfg["seed"]
    features_dir = pathlib.Path(args.features_dir or data_path(cfg, "features") / f"temporal_{prefix}")
    tf = load_temporal(features_dir, games=manifest_games(cfg)).drop_labels(dropped_labels(cfg))
    x = mean_pool(tf.features, tf.masks)
    labels = sorted(set(tf.labels))
    games = sorted(set(tf.games))
    label_ids = np.array([labels.index(v) for v in tf.labels])
    game_ids = np.array([games.index(v) for v in tf.games])
    k = len(labels)
    print(f"{len(x)} events, {len(games)} games, k = {k}")

    steps = kmeans_steps(x.astype(np.float64), k, seed)
    settled = len(steps) < MAX_STEPS
    print(f"K-Means {'settled after' if settled else 'stopped, still moving, at'} {len(steps)} assignment steps")
    xy = TSNE(n_components=2, perplexity=min(30, len(x) - 1), max_iter=1000,
              random_state=seed, init="pca").fit_transform(x)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0)).clip(min=1e-9)

    out = {
        "title": args.title,
        "k": k,
        "settled": settled,
        "labels": labels,
        "games": games,
        "points": {"x": np.round(xy[:, 0], 4).tolist(), "y": np.round(xy[:, 1], 4).tolist(),
                   "label": label_ids.tolist(), "game": game_ids.tolist(), "half": tf.halves.tolist()},
        "steps": [{
            "assign": a.tolist(),
            "purity": round(purity(label_ids, a), 4),
            "ari_label": round(adjusted_rand_score(label_ids, a), 4),
            "ari_game": round(adjusted_rand_score(game_ids, a), 4),
            "moved": int((a != steps[i - 1]).sum()) if i else len(a),
        } for i, a in enumerate(steps)],
    }
    pathlib.Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    last = out["steps"][-1]
    print(f"final: purity {last['purity']}, ARI vs label {last['ari_label']}, ARI vs match {last['ari_game']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
