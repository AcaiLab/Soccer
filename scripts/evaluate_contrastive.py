"""
Post-Contrastive Clustering Evaluation
=======================================
Compares K-Means + t-SNE on the projected embeddings with the original DINOv2
mean-pooled embeddings, separately for:

  held-out games - games the head never saw; the result that counts
  train games    - games whose labels the head was trained on; shows fit only

The two are never pooled into one number: a pooled score averages a near-perfect
fit on training games with whatever happens on unseen ones, which is how a
"leak-free" run could still look like a gain (docs/leak_audit.md, B2).

Produces (in --out-dir):
  - before_vs_after_labels.png, before_vs_after_games.png (held-out games)
  - contrastive_3panel.png (held-out games: label | game | k-means)
  - metrics_comparison.png, contrastive_metrics.csv (both groups)

Usage:
    python scripts/evaluate_contrastive.py
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from soccer.config import data_path, dropped_labels, load_config  # noqa: E402
from soccer.features import load_temporal, mean_pool  # noqa: E402
from soccer.metrics import clustering_scores  # noqa: E402


def score(features, labels, seed):
    k = len(set(labels))
    cluster_ids = KMeans(n_clusters=k, random_state=seed, n_init=15).fit_predict(features)
    return cluster_ids, {**clustering_scores(labels, cluster_ids, seed=seed),
                         "Silhouette": silhouette_score(features, cluster_ids,
                                                        sample_size=min(2000, len(features)),
                                                        random_state=seed)}


def tsne(features, seed):
    return TSNE(n_components=2, perplexity=min(30, len(features) - 1), max_iter=1000,
                random_state=seed).fit_transform(features)


def scatter_pair(maps, colors, cmap, vmax, titles, suptitle, path, handles=None):
    fig, axes = plt.subplots(1, len(maps), figsize=(8 * len(maps), 6))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    for ax, xy, c, t in zip(axes, maps, colors, titles, strict=True):
        ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, vmin=0, vmax=vmax, s=25, alpha=0.8, linewidths=0)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    if handles:
        axes[0].legend(handles=handles, fontsize=6, loc="lower left", ncol=2, framealpha=0.7, handlelength=1)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main(argv=None):
    cfg = load_config()
    prefix = cfg["features"]["model"].split("/")[-1]
    features_root = data_path(cfg, "features")
    parser = argparse.ArgumentParser(description="Evaluate the contrastive head on held-out and train games.")
    parser.add_argument("--dino-features", default=str(features_root / f"temporal_{prefix}"))
    parser.add_argument("--contrastive-features", default=str(features_root / "contrastive"))
    parser.add_argument("--out-dir", default=str(data_path(cfg, "results") / "contrastive"))
    parser.add_argument("--keep-all-labels", action="store_true",
                        help="must match the setting used for contrastive_training.py")
    args = parser.parse_args(argv)

    seed = cfg["seed"]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = load_temporal(args.dino_features)
    if not args.keep_all_labels:
        tf = tf.drop_labels(dropped_labels(cfg))
    cont = np.load(pathlib.Path(args.contrastive_features) / "contrastive_embeddings.npz")
    if "train_games" not in cont.files:
        raise SystemExit("These embeddings do not record their training games (made by the old, leaky "
                         "contrastive_training.py). Retrain with the current version.")
    for key, ours in [("games", tf.games), ("halves", tf.halves), ("labels", tf.labels), ("event_tags", tf.event_tags)]:
        if not np.array_equal(cont[key].astype(str), ours):
            raise SystemExit(f"Event order differs from the DINOv2 features ({key}); check --keep-all-labels "
                             "and that both come from the same feature run.")

    before = mean_pool(tf.features, tf.masks)
    after = cont["features"].astype(np.float32)
    train_games = set(cont["train_games"].astype(str))
    groups = {"held-out games": ~np.isin(tf.games, list(train_games)),
              "train games": np.isin(tf.games, list(train_games))}
    if not groups["held-out games"].any():
        raise SystemExit("Every game was used for training; there is nothing honest to evaluate.")

    rows, cache = [], {}
    for group, m in groups.items():
        print(f"\n[{group}] {m.sum()} events from {len(set(tf.games[m]))} games")
        for rep, feats in [("DINOv2 (before)", before), ("Contrastive (after)", after)]:
            cluster_ids, metrics = score(feats[m], tf.labels[m], seed)
            cache[(group, rep)] = cluster_ids
            rows.append({"Group": group, "Representation": rep, "Events": int(m.sum()), **metrics})
            print(f"  {rep:<20s} ARI={metrics['ARI']:.3f}  Purity={metrics['Purity']:.3f} "
                  f"(random {metrics['Purity_random']:.3f})  Silhouette={metrics['Silhouette']:.3f}")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "contrastive_metrics.csv", index=False)

    # Plots: held-out games only.
    m = groups["held-out games"]
    labels, games = tf.labels[m], tf.games[m]
    classes, game_names = sorted(set(labels)), sorted(set(games))
    lab_ids = np.array([classes.index(x) for x in labels])
    game_ids = np.array([game_names.index(g) for g in games])
    lab_cmap = plt.get_cmap("tab20" if len(classes) <= 20 else "turbo", len(classes))
    game_cmap = plt.get_cmap("tab10" if len(game_names) <= 10 else "tab20", len(game_names))
    lab_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=lab_cmap(i), markersize=7, label=c)
                   for i, c in enumerate(classes)]
    xy_before, xy_after = tsne(before[m], seed), tsne(after[m], seed)
    r = {rep: metrics_df[(metrics_df.Group == "held-out games") & (metrics_df.Representation == rep)].iloc[0]
         for rep in ("DINOv2 (before)", "Contrastive (after)")}
    titles = [f"{rep}\nARI={r[rep].ARI:.3f}  Purity={r[rep].Purity:.3f} (random {r[rep].Purity_random:.3f})"
              for rep in r]

    scatter_pair([xy_before, xy_after], [lab_ids, lab_ids], lab_cmap, len(classes) - 1, titles,
                 f"Held-out games ({', '.join(game_names)}): before vs after, coloured by event label",
                 out_dir / "before_vs_after_labels.png", lab_handles)
    scatter_pair([xy_before, xy_after], [game_ids, game_ids], game_cmap, len(game_names) - 1,
                 ["DINOv2 (before)", "Contrastive (after)"],
                 "Held-out games: before vs after, coloured by game", out_dir / "before_vs_after_games.png")
    km = cache[("held-out games", "Contrastive (after)")]
    scatter_pair([xy_after] * 3, [lab_ids, game_ids, km], "tab20", max(len(classes), len(game_names)) - 1,
                 ["Colored by Event Label", "Colored by Game", f"K-Means (k={len(classes)})"],
                 f"Post-contrastive embeddings, held-out games only ({m.sum()} events)",
                 out_dir / "contrastive_3panel.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, group in zip(axes, groups, strict=True):
        sub = metrics_df[metrics_df.Group == group]
        x = np.arange(3)
        for j, rep in enumerate(("DINOv2 (before)", "Contrastive (after)")):
            vals = sub[sub.Representation == rep][["ARI", "Purity", "Silhouette"]].iloc[0].values
            bars = ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, label=rep, alpha=0.85)
            ax.bar_label(bars, fmt="%.3f", fontsize=9)
        ax.set_xticks(x, ["ARI", "Purity", "Silhouette"])
        ax.set_title(group, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle("Clustering before vs after contrastive learning", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n" + metrics_df.to_string(index=False, float_format="%.4f"))
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
