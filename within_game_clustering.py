"""
Within-Game Clustering Analysis
================================
Clusters events separately within each game to see how events group
when game-level visual bias is naturally removed.

Produces per-game t-SNE + K-Means plots and a summary metrics CSV.

Usage:
    python within_game_clustering.py
    python within_game_clustering.py --features-dir features/temporal_dinov2-base
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--features-dir", default="features/temporal_dinov2-base")
parser.add_argument("--out-dir", default="clustering/within_game")
args = parser.parse_args()

features_dir = pathlib.Path(args.features_dir)
out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load features
# ---------------------------------------------------------------------------
print("Loading features ...")
npz = np.load(features_dir / "clip_temporal_padded.npz", allow_pickle=True)
feat_padded = npz["features"].astype(np.float32)   # (627, 60, 768)
masks = npz["masks"]                                 # (627, 60)
labels = npz["labels"].astype(str)                   # (627,)
games = npz["games"].astype(str)                     # (627,)

# Mean-pool per event
n_valid = masks.sum(axis=1, keepdims=True).clip(min=1)
feat_mean = (feat_padded * masks[:, :, np.newaxis]).sum(axis=1) / n_valid
feat_mean = feat_mean / np.linalg.norm(feat_mean, axis=1, keepdims=True)

print(f"  Events: {len(labels)}")
print(f"  Unique games: {sorted(set(games))}")

# ---------------------------------------------------------------------------
# Global label setup
# ---------------------------------------------------------------------------
all_labels = sorted(set(labels))
le = LabelEncoder().fit(all_labels)
n_total_labels = len(all_labels)
cmap = plt.get_cmap("tab20", n_total_labels)

legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap(i), markersize=7, label=all_labels[i])
    for i in range(n_total_labels)
]


def cluster_purity(true_labels, cluster_ids, le_, n_classes):
    correct = 0
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        counts = np.bincount(le_.transform(true_labels[mask]), minlength=n_classes)
        correct += counts.max()
    return correct / len(true_labels)


# ---------------------------------------------------------------------------
# Per-game clustering
# ---------------------------------------------------------------------------
unique_games = sorted(set(games))
all_metrics = []

for g in unique_games:
    mask_g = games == g
    feats_g = feat_mean[mask_g]
    labels_g = labels[mask_g]

    unique_labels_g = sorted(set(labels_g))
    n_labels_g = len(unique_labels_g)
    n_events_g = len(labels_g)

    print(f"\n--- Game {g}: {n_events_g} events, {n_labels_g} label types ---")

    if n_events_g < 4 or n_labels_g < 2:
        print("  Too few events/labels, skipping.")
        continue

    k = min(n_labels_g, n_events_g - 1)  # can't have more clusters than events

    # K-Means
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=15)
    cluster_ids = km.fit_predict(feats_g)

    ari = adjusted_rand_score(labels_g, cluster_ids)
    sil = silhouette_score(feats_g, cluster_ids,
                           random_state=RANDOM_STATE) if k > 1 and k < n_events_g else 0.0
    pur = cluster_purity(labels_g, cluster_ids, le, n_total_labels)

    print(f"  k={k}  Purity={pur:.3f}  ARI={ari:.3f}  Silhouette={sil:.3f}")
    all_metrics.append({
        "Game": g, "Events": n_events_g, "Labels": n_labels_g,
        "k": k, "Purity": pur, "ARI": ari, "Silhouette": sil,
    })

    # t-SNE
    perp = min(30, n_events_g - 1)
    tsne_xy = TSNE(n_components=2, perplexity=perp, max_iter=1000,
                   random_state=RANDOM_STATE).fit_transform(feats_g)

    # Plot: true labels vs K-Means
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Game {g} -- Within-Game Clustering  ({n_events_g} events, {n_labels_g} labels)\n"
                 f"Purity={pur:.3f}  ARI={ari:.3f}  Silhouette={sil:.3f}",
                 fontsize=12, fontweight="bold")

    true_ids = le.transform(labels_g)
    for ax, color_ids, title in [
        (axes[0], true_ids, "Colored by Event Label"),
        (axes[1], cluster_ids, f"K-Means (k={k})"),
    ]:
        ax.scatter(tsne_xy[:, 0], tsne_xy[:, 1],
                   c=color_ids, cmap=cmap, vmin=0, vmax=n_total_labels - 1,
                   s=55, alpha=0.85, linewidths=0.4, edgecolors="white")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Legend with only labels present in this game
    present_handles = [legend_handles[i] for i in range(n_total_labels)
                       if all_labels[i] in unique_labels_g]
    axes[0].legend(handles=present_handles, fontsize=7, loc="lower left",
                   ncol=2, framealpha=0.7, handlelength=1)

    plt.tight_layout()
    fig.savefig(out_dir / f"game_{g}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_dir / f'game_{g}.png'}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame(all_metrics)
metrics_path = out_dir / "within_game_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\n" + "=" * 60)
print("WITHIN-GAME CLUSTERING SUMMARY")
print("=" * 60)
print(metrics_df.to_string(index=False, float_format="%.3f"))
print(f"\nMean Purity:     {metrics_df['Purity'].mean():.3f}")
print(f"Mean ARI:        {metrics_df['ARI'].mean():.3f}")
print(f"Mean Silhouette: {metrics_df['Silhouette'].mean():.3f}")
print(f"\nSaved -> {metrics_path}")
