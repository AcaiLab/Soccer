"""
Within-Game Clustering Analysis
================================
Clusters events separately within each game, which removes the game-level
visual bias (stadium, light, kit, camera) by construction.

Purity is reported next to its shuffled-cluster baseline: with ~60 events in
13-19 clusters many clusters hold only a few events and look pure by chance,
so a purity number on its own overstates the structure. ARI is chance-corrected.

Produces per-game t-SNE + K-Means plots and a summary metrics CSV.

Usage:
    python scripts/within_game_clustering.py
    python scripts/within_game_clustering.py --features-dir features/temporal_dinov2-base
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from soccer.config import data_path, dropped_labels, load_config  # noqa: E402
from soccer.features import load_temporal, mean_pool  # noqa: E402
from soccer.manifest import manifest_games  # noqa: E402
from soccer.metrics import clustering_scores  # noqa: E402


def main(argv=None):
    cfg = load_config()
    prefix = cfg["features"]["model"].split("/")[-1]
    parser = argparse.ArgumentParser(description="Cluster events separately inside each game.")
    parser.add_argument("--features-dir", default=str(data_path(cfg, "features") / f"temporal_{prefix}"))
    parser.add_argument("--out-dir", default=str(data_path(cfg, "results") / "within_game"))
    parser.add_argument("--keep-all-labels", action="store_true")
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    tf = load_temporal(args.features_dir, games=manifest_games(cfg))
    if not args.keep_all_labels:
        tf = tf.drop_labels(dropped_labels(cfg))
    feat_mean = mean_pool(tf.features, tf.masks)

    classes = sorted(set(tf.labels))
    cmap = plt.get_cmap("tab20" if len(classes) <= 20 else "turbo", len(classes))
    idx = {c: i for i, c in enumerate(classes)}

    rows = []
    for g in sorted(set(tf.games)):
        m = tf.games == g
        feats, labels = feat_mean[m], tf.labels[m]
        n_labels = len(set(labels))
        if len(labels) < 4 or n_labels < 2:
            print(f"Game {g}: too few events or labels, skipping")
            continue
        k = min(n_labels, len(labels) - 1)
        cluster_ids = KMeans(n_clusters=k, random_state=seed, n_init=15).fit_predict(feats)
        row = {"Game": g, "Events": len(labels), "Labels": n_labels, "k": k,
               **clustering_scores(labels, cluster_ids, seed=seed),
               "Silhouette": silhouette_score(feats, cluster_ids, random_state=seed)}
        rows.append(row)
        print(f"Game {g}: k={k}  ARI={row['ARI']:.3f}  Purity={row['Purity']:.3f} "
              f"(random {row['Purity_random']:.3f})")

        xy = TSNE(n_components=2, perplexity=min(30, len(labels) - 1), max_iter=1000,
                  random_state=seed).fit_transform(feats)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Game {g} -- within-game clustering ({len(labels)} events, {n_labels} labels)\n"
                     f"ARI={row['ARI']:.3f}  Purity={row['Purity']:.3f} (random {row['Purity_random']:.3f})",
                     fontsize=12, fontweight="bold")
        for ax, c, vmax, title in [(axes[0], [idx[x] for x in labels], len(classes) - 1, "Colored by Event Label"),
                                   (axes[1], cluster_ids, k - 1, f"K-Means (k={k})")]:
            ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap if title.startswith("Colored") else "tab20",
                       vmin=0, vmax=vmax, s=55, alpha=0.85, linewidths=0.4, edgecolors="white")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        present = sorted(set(labels))
        axes[0].legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(idx[c]),
                                           markersize=7, label=c) for c in present],
                       fontsize=7, loc="lower left", ncol=2, framealpha=0.7, handlelength=1)
        plt.tight_layout()
        fig.savefig(out_dir / f"game_{g}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "within_game_metrics.csv", index=False)
    print("\n" + df.to_string(index=False, float_format="%.3f"))
    print(f"\nMean ARI {df['ARI'].mean():.3f} | mean purity {df['Purity'].mean():.3f} "
          f"vs random {df['Purity_random'].mean():.3f}")


if __name__ == "__main__":
    main()
