"""
K-Means + t-SNE Analysis of Soccer Event Embeddings
====================================================
Representations analysed:

  (A) Individual frames  - one embedding per frame (taken from the per-event features)
  (B) Event mean-pool    - mask-aware mean over each event's frames
  (C) Event concat+PCA   - all frames of an event concatenated, reduced with PCA
  (F) Per-game centred   - (B) with each game's mean subtracted

For each: K-Means with k = number of event labels, scored against the event
labels (purity next to its shuffled-cluster baseline, ARI, NMI) and against
game identity (Game_ARI, Game_NMI), plus silhouette and t-SNE maps coloured by
label, k-means cluster, game and half. ARI is the number to lead with: it is
already corrected for chance, purity is not. Game_ARI well above ARI means the
clusters follow the match more than the event.

Junk and non-visual labels (configs/pipeline.yaml) are dropped first.

Outputs (in --out-dir, default results/<model name>)
--------------------------------------------------------
  A_individual_frames.png, B_event_meanpool.png, C_event_concat_pca.png,
  D_colorings_game_half.png, F_normalized_event_meanpool.png,
  F_normalized_colorings_game_half.png, per_game/game_XX.png,
  clustering_metrics.csv

Usage
-----
    python scripts/cluster_analysis.py
    python scripts/cluster_analysis.py --model openai/clip-vit-base-patch32
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from soccer.config import data_path, dropped_labels, load_config  # noqa: E402
from soccer.features import center_per_group, l2_normalize, load_temporal, mean_pool  # noqa: E402
from soccer.manifest import (
    manifest_games,  # noqa: E402
    normalize_game_id,  # noqa: E402
)
from soccer.metrics import clustering_scores, game_vs_event_scores  # noqa: E402

N_TSNE_SAMPLE = 3000  # individual frames are subsampled for t-SNE (speed)


def palette(n: int):
    """A colormap with n distinguishable entries, also beyond 10 or 20."""
    return plt.get_cmap("tab10" if n <= 10 else "tab20" if n <= 20 else "turbo", n)


def legend(names, cmap, prefix=""):
    return [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i),
                       markersize=7, label=f"{prefix}{n}") for i, n in enumerate(names)]


class Analysis:
    def __init__(self, labels_all, out_dir, model_name, seed):
        self.classes = sorted(set(labels_all))
        self.idx = {c: i for i, c in enumerate(self.classes)}
        self.k = len(self.classes)
        self.cmap = palette(self.k)
        self.handles = legend(self.classes, self.cmap)
        self.out_dir = out_dir
        self.model_name = model_name
        self.seed = seed
        self.rows = []

    def ids(self, labels):
        return np.array([self.idx[x] for x in labels])

    def kmeans(self, features, labels, tag, games):
        print(f"\n[K-Means] {tag}  n={len(features)}  k={self.k}")
        cluster_ids = KMeans(n_clusters=self.k, random_state=self.seed, n_init=15).fit_predict(features)
        row = {"Representation": tag, **clustering_scores(labels, cluster_ids, seed=self.seed),
               **game_vs_event_scores(labels, games, cluster_ids),
               "Silhouette": silhouette_score(features, cluster_ids, sample_size=min(2000, len(features)),
                                              random_state=self.seed)}
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "Representation"))
        self.rows.append(row)
        return cluster_ids, row

    def tsne(self, features):
        perplexity = min(30, len(features) - 1)
        return TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
                    random_state=self.seed).fit_transform(features)

    def title(self, letter, name, n, row):
        return (f"({letter}) {name}  --  {self.model_name}\n{n:,} items  |  ARI={row['ARI']:.3f}  "
                f"Purity={row['Purity']:.3f} (random {row['Purity_random']:.3f})  "
                f"Silhouette={row['Silhouette']:.3f}")

    def label_vs_cluster(self, xy, true_ids, cluster_ids, title, filename):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=12, fontweight="bold")
        for ax, c, t in [(axes[0], true_ids, "t-SNE  (true label)"), (axes[1], cluster_ids, "t-SNE  (k-means cluster)")]:
            ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=self.cmap, vmin=0, vmax=self.k - 1, s=10, alpha=0.65, linewidths=0)
            ax.set_title(t, fontsize=10)
            ax.axis("off")
        axes[0].legend(handles=self.handles, fontsize=6, loc="lower left", ncol=2, framealpha=0.7, handlelength=1)
        self.save(fig, filename)

    def colorings(self, xy, labels, games, halves, title, filename):
        g_names, h_names = sorted(set(games)), sorted(set(halves))
        g_cmap, h_cmap = palette(len(g_names)), palette(len(h_names))
        panels = [
            (self.ids(labels), self.cmap, self.k, self.handles, "Colored by Event Label"),
            (np.array([g_names.index(g) for g in games]), g_cmap, len(g_names),
             legend(g_names, g_cmap, "Game ") if len(g_names) <= 30 else [], "Colored by Game"),
            (np.array([h_names.index(h) for h in halves]), h_cmap, len(h_names),
             legend(h_names, h_cmap, "Half "), "Colored by Half"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(title, fontsize=12, fontweight="bold")
        for ax, (c, cmap, n, handles, t) in zip(axes, panels, strict=True):
            ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, vmin=0, vmax=max(n - 1, 1), s=18, alpha=0.8, linewidths=0)
            ax.set_title(t, fontsize=11)
            ax.axis("off")
            if handles:
                ax.legend(handles=handles, fontsize=7, loc="lower left", ncol=2 if len(handles) > 6 else 1,
                          framealpha=0.7, handlelength=1)
        self.save(fig, filename)

    def save(self, fig, filename):
        path = self.out_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {path}")


def main(argv=None):
    cfg = load_config()
    model = cfg["features"]["model"]
    parser = argparse.ArgumentParser(description="K-Means + t-SNE clustering of soccer event embeddings.")
    parser.add_argument("--model", default=model, help="selects the default feature and output folders")
    parser.add_argument("--features-dir", default=None, help="default: features/temporal_<model name>")
    parser.add_argument("--flat-features", default=None,
                        help="per-frame features from extract_frame_features.py (default: the frames inside "
                             "the per-event features, so no second encoding pass is needed)")
    parser.add_argument("--flat-meta", default=None, help="default: metadata.csv next to --flat-features")
    parser.add_argument("--out-dir", default=None, help="default: results/<model name>")
    parser.add_argument("--keep-all-labels", action="store_true", help="do not drop junk / non-visual labels")
    args = parser.parse_args(argv)

    prefix = args.model.split("/")[-1]
    features_root = data_path(cfg, "features")
    features_dir = pathlib.Path(args.features_dir or features_root / f"temporal_{prefix}")
    flat_path = pathlib.Path(args.flat_features) if args.flat_features else None
    meta_path = pathlib.Path(args.flat_meta) if args.flat_meta else (flat_path.with_name("metadata.csv")
                                                                       if flat_path else None)
    out_dir = pathlib.Path(args.out_dir or data_path(cfg, "results") / prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    drop = set() if args.keep_all_labels else dropped_labels(cfg)

    tf = load_temporal(features_dir, games=manifest_games(cfg))
    n_before = len(tf)
    tf = tf.drop_labels(drop)
    print(f"Events: {len(tf)} ({n_before - len(tf)} with dropped labels removed), "
          f"games: {len(set(tf.games))}, frames tensor {tf.features.shape}")

    have_flat = True
    if args.flat_features:
        # Per-frame features from extract_frame_features.py (e.g. another model's).
        flat = np.load(flat_path)
        meta = pd.read_csv(meta_path, dtype={"game": str})
        flat_labels = meta["label"].astype(str).values
        flat_games = np.array([normalize_game_id(g) for g in meta["game"]])
        keep = ~np.isin(flat_labels, list(drop)) & np.isin(flat_games, manifest_games(cfg))
        flat, flat_labels, flat_games = flat[keep], flat_labels[keep], flat_games[keep]
    else:
        # The per-event tensors already hold every frame's embedding: no second encoding pass.
        flat = tf.features[tf.masks]
        n_valid = tf.masks.sum(axis=1)
        flat_labels, flat_games = np.repeat(tf.labels, n_valid), np.repeat(tf.games, n_valid)

    a = Analysis(np.concatenate([tf.labels, flat_labels]), out_dir, prefix, seed)
    true_ids = a.ids(tf.labels)

    # (A) individual frames
    if have_flat:
        cluster_A, row = a.kmeans(flat, flat_labels, "Individual frames", flat_games)
        idx = np.random.default_rng(seed).choice(len(flat), size=min(N_TSNE_SAMPLE, len(flat)), replace=False)
        a.label_vs_cluster(a.tsne(flat[idx]), a.ids(flat_labels[idx]), cluster_A[idx],
                           a.title("A", "Individual Frames", len(flat), row), "A_individual_frames.png")

    # (B) event mean-pool
    feat_mean = mean_pool(tf.features, tf.masks)
    cluster_B, row = a.kmeans(feat_mean, tf.labels, "Event mean-pool", tf.games)
    tsne_B = a.tsne(feat_mean)
    a.label_vs_cluster(tsne_B, true_ids, cluster_B, a.title("B", "Event Mean-Pool", len(tf), row),
                       "B_event_meanpool.png")

    # (C) event concat + PCA
    flat_events = tf.features.reshape(len(tf), -1)
    n_comp = min(512, len(tf) - 1, flat_events.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed, svd_solver="randomized")
    feat_pca = l2_normalize(pca.fit_transform(flat_events))
    print(f"  PCA to {n_comp} dims, variance explained {pca.explained_variance_ratio_.sum():.1%}")
    cluster_C, row = a.kmeans(feat_pca, tf.labels, "Event concat+PCA", tf.games)
    a.label_vs_cluster(a.tsne(feat_pca), true_ids, cluster_C,
                       a.title("C", f"Event Concat + PCA ({n_comp})", len(tf), row), "C_event_concat_pca.png")

    # (D) the mean-pool map coloured by label, game and half
    a.colorings(tsne_B, tf.labels, tf.games, tf.halves,
                f"(D) Event Mean-Pool t-SNE -- {prefix}\n{len(tf)} events", "D_colorings_game_half.png")

    # (E) each game highlighted on the global map
    for g in sorted(set(tf.games)):
        fg = tf.games == g
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(tsne_B[~fg, 0], tsne_B[~fg, 1], c="lightgrey", s=8, alpha=0.3, linewidths=0)
        ax.scatter(tsne_B[fg, 0], tsne_B[fg, 1], c=true_ids[fg], cmap=a.cmap, vmin=0, vmax=a.k - 1,
                   s=55, alpha=0.95, linewidths=0.4, edgecolors="white")
        ax.set_title(f"Game {g} -- {fg.sum()} events on the global t-SNE (colour = label)", fontsize=11)
        ax.axis("off")
        a.save(fig, f"per_game/game_{g}.png")

    # (F) per-game centring
    feat_centred = center_per_group(feat_mean, tf.games)
    cluster_F, row = a.kmeans(feat_centred, tf.labels, "Normalized mean-pool", tf.games)
    tsne_F = a.tsne(feat_centred)
    a.label_vs_cluster(tsne_F, true_ids, cluster_F, a.title("F", "Per-Game Centred Mean-Pool", len(tf), row),
                       "F_normalized_event_meanpool.png")
    a.colorings(tsne_F, tf.labels, tf.games, tf.halves,
                f"(F) Per-Game Centred Mean-Pool t-SNE -- {prefix}\n{len(tf)} events",
                "F_normalized_colorings_game_half.png")

    metrics = pd.DataFrame(a.rows)
    metrics.to_csv(out_dir / "clustering_metrics.csv", index=False)
    print("\n" + metrics.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved -> {out_dir / 'clustering_metrics.csv'}")


if __name__ == "__main__":
    main()
