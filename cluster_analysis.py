"""
K-Means + t-SNE Analysis of CLIP Soccer Event Embeddings
=========================================================
Three representations are analysed:

  (A) Individual frames  -  (N_frames, 512)
  (B) Event mean-pool    -  (N_events, 512)   mask-aware mean across frames
  (C) Event concat+PCA   -  (N_events, 512)   PCA on flattened (60 x 512)

For each representation:
  * K-Means (k = n_label_types) -> cluster purity, ARI, silhouette
  * t-SNE 2-D -> scatter coloured by true label vs. k-means cluster

Outputs
-------
  clustering/A_individual_frames.png
  clustering/B_event_meanpool.png
  clustering/C_event_concat_pca.png
  clustering/clustering_metrics.csv

Usage
-----
    python cluster_analysis.py
"""

import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(
    description="K-Means + t-SNE clustering of soccer event embeddings."
)
_ap.add_argument(
    "--features-dir",
    default="features/temporal",
    help="Directory containing clip_temporal_padded.npz "
         "(default: features/temporal). Use features/temporal_dino for DINOv2.",
)
_ap.add_argument(
    "--flat-features",
    default="features/clip_features.npy",
    help="Path to flat per-frame feature matrix (default: features/clip_features.npy)",
)
_ap.add_argument(
    "--flat-meta",
    default="features/metadata.csv",
    help="Path to flat feature metadata CSV (default: features/metadata.csv)",
)
_ap.add_argument(
    "--out-dir",
    default="clustering",
    help="Output directory for clustering results (default: clustering)",
)
_args = _ap.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_FLAT = pathlib.Path(_args.flat_features)
META_FLAT     = pathlib.Path(_args.flat_meta)
PKL_TEMPORAL  = pathlib.Path(_args.features_dir) / "clip_temporal.pkl"
NPZ_TEMPORAL  = pathlib.Path(_args.features_dir) / "clip_temporal_padded.npz"
OUT_DIR       = pathlib.Path(_args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE   = 42
N_TSNE_SAMPLE  = 3000   # subsample individual frames for t-SNE (speed)

# ---------------------------------------------------------------------------
# Load features
# ---------------------------------------------------------------------------
print("Loading features ...")

feat_flat    = np.load(FEATURES_FLAT)                         # (18501, 512)
labels_flat  = pd.read_csv(META_FLAT)["label"].values

npz          = np.load(NPZ_TEMPORAL, allow_pickle=True)
feat_padded  = npz["features"].astype(np.float32)             # (627, 60, 512)
masks        = npz["masks"]                                   # (627, 60)
labels_event = npz["labels"].astype(str)                      # (627,)
games_event  = npz["games"].astype(int)                       # (627,)
halves_event = npz["halves"].astype(int)                      # (627,)

print(f"  Frames  : {feat_flat.shape}")
print(f"  Events  : {feat_padded.shape}")

# ---------------------------------------------------------------------------
# Build event-level representations
# ---------------------------------------------------------------------------

# (B) Mask-aware mean pool -> (627, 512), then re-L2-normalise
n_valid   = masks.sum(axis=1, keepdims=True).clip(min=1)
feat_mean = (feat_padded * masks[:, :, np.newaxis]).sum(axis=1) / n_valid
feat_mean = feat_mean / np.linalg.norm(feat_mean, axis=1, keepdims=True)

# (C) Flatten (627, 60x512=30720) -> PCA -> (627, 512)
print("Fitting PCA on concatenated event features ...")
feat_concat_raw = feat_padded.reshape(len(feat_padded), -1)
pca = PCA(n_components=512, random_state=RANDOM_STATE, svd_solver="randomized")
feat_concat_pca = pca.fit_transform(feat_concat_raw)
feat_concat_pca = feat_concat_pca / np.linalg.norm(feat_concat_pca, axis=1, keepdims=True)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# ---------------------------------------------------------------------------
# Label encoder  (shared across all representations)
# ---------------------------------------------------------------------------
all_labels = sorted(set(labels_flat) | set(labels_event))
le         = LabelEncoder().fit(all_labels)
n_clusters = len(all_labels)
cmap       = plt.get_cmap("tab20", n_clusters)

legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap(i), markersize=7, label=all_labels[i])
    for i in range(n_clusters)
]

print(f"\nLabel types ({n_clusters}): {all_labels}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cluster_purity(true_labels, cluster_ids):
    """Fraction of samples assigned to their cluster's majority true label."""
    correct = 0
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        counts = np.bincount(le.transform(true_labels[mask]), minlength=n_clusters)
        correct += counts.max()
    return correct / len(true_labels)


def normalize_per_game(feat_2d: np.ndarray, games: np.ndarray) -> np.ndarray:
    """
    Remove the per-game centroid from each event embedding, then re-L2-normalize.

    Motivation: CLIP/DINOv2 embeddings cluster by broadcast visual style
    (stadium lighting, kit colours) rather than event semantics.  Subtracting
    each game's mean removes this low-rank bias so remaining variance better
    reflects event content.
    """
    out = feat_2d.copy().astype(np.float32)
    for g in np.unique(games):
        mask = games == g
        out[mask] -= out[mask].mean(axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-8)
    return out / norms


def run_kmeans(features, labels, tag):
    print(f"\n[K-Means] {tag}  n={len(features)}  k={n_clusters}")
    km          = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=15)
    cluster_ids = km.fit_predict(features)
    ari  = adjusted_rand_score(labels, cluster_ids)
    sil  = silhouette_score(features, cluster_ids,
                             sample_size=min(2000, len(features)),
                             random_state=RANDOM_STATE)
    pur  = cluster_purity(labels, cluster_ids)
    print(f"  Purity={pur:.3f}   ARI={ari:.3f}   Silhouette={sil:.3f}")
    return cluster_ids, {"Representation": tag, "Purity": pur, "ARI": ari, "Silhouette": sil}


def run_tsne(features, tag):
    print(f"[t-SNE]   {tag}  n={len(features)}")
    return TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=RANDOM_STATE).fit_transform(features)


def make_figure(tsne_xy, true_ids, cluster_ids, suptitle, filename, left_title="t-SNE  (true label)", right_title="t-SNE  (k-means cluster)"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for ax, color_ids, title in [
        (axes[0], true_ids,    left_title),
        (axes[1], cluster_ids, right_title),
    ]:
        sc = ax.scatter(tsne_xy[:, 0], tsne_xy[:, 1],
                        c=color_ids, cmap=cmap, vmin=0, vmax=n_clusters - 1,
                        s=10, alpha=0.65, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    axes[0].legend(handles=legend_handles, fontsize=6, loc="lower left",
                   ncol=2, framealpha=0.7, handlelength=1)

    plt.tight_layout()
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# (A) Individual frames
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("(A) Individual frames")
print("-" * 60)

# k-means on full set
cluster_A, metrics_A = run_kmeans(feat_flat, labels_flat, "Individual frames")

# t-SNE on a random subsample (speed)
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(feat_flat), size=min(N_TSNE_SAMPLE, len(feat_flat)), replace=False)
tsne_A = run_tsne(feat_flat[idx], "Individual frames (subsample)")

make_figure(
    tsne_A,
    true_ids    = le.transform(labels_flat[idx]),
    cluster_ids = cluster_A[idx],
    suptitle    = f"(A) Individual Frames  --  CLIP 512-dim\n"
                  f"t-SNE subsample: {len(idx):,} / {len(feat_flat):,}  |  "
                  f"Purity={metrics_A['Purity']:.3f}  ARI={metrics_A['ARI']:.3f}  "
                  f"Silhouette={metrics_A['Silhouette']:.3f}",
    filename    = "A_individual_frames.png",
)

# ---------------------------------------------------------------------------
# (B) Event mean-pool
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("(B) Event mean-pool")
print("-" * 60)

cluster_B, metrics_B = run_kmeans(feat_mean, labels_event, "Event mean-pool")
tsne_B = run_tsne(feat_mean, "Event mean-pool")

make_figure(
    tsne_B,
    true_ids    = le.transform(labels_event),
    cluster_ids = cluster_B,
    suptitle    = f"(B) Event Mean-Pool  --  CLIP 512-dim\n"
                  f"{len(feat_mean)} events  |  "
                  f"Purity={metrics_B['Purity']:.3f}  ARI={metrics_B['ARI']:.3f}  "
                  f"Silhouette={metrics_B['Silhouette']:.3f}",
    filename    = "B_event_meanpool.png",
)

# ---------------------------------------------------------------------------
# (C) Event concat + PCA
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("(C) Event concat + PCA  (60 x 512 -> PCA -> 512)")
print("-" * 60)

cluster_C, metrics_C = run_kmeans(feat_concat_pca, labels_event, "Event concat+PCA")
tsne_C = run_tsne(feat_concat_pca, "Event concat+PCA")

make_figure(
    tsne_C,
    true_ids    = le.transform(labels_event),
    cluster_ids = cluster_C,
    suptitle    = f"(C) Event Concat + PCA  --  60x512 -> PCA->512\n"
                  f"{len(feat_concat_pca)} events  |  "
                  f"Purity={metrics_C['Purity']:.3f}  ARI={metrics_C['ARI']:.3f}  "
                  f"Silhouette={metrics_C['Silhouette']:.3f}",
    filename    = "C_event_concat_pca.png",
)

# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame([metrics_A, metrics_B, metrics_C])
metrics_path = OUT_DIR / "clustering_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\n" + "=" * 60)
print("METRICS SUMMARY")
print("=" * 60)
print(metrics_df.to_string(index=False, float_format="%.4f"))
print(f"\nSaved -> {metrics_path}")

# ---------------------------------------------------------------------------
# (D) Re-color the event mean-pool t-SNE by game and by half
#     Reuses tsne_B coordinates already computed above
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("(D) Alternative colorings: game and half")
print("-" * 60)

unique_games  = sorted(np.unique(games_event))   # [1..10]
unique_halves = sorted(np.unique(halves_event))  # [1, 2]
n_games       = len(unique_games)

cmap_game  = plt.get_cmap("tab10", n_games)
cmap_half  = plt.get_cmap("Set1",  2)

# Map game/half values to 0-based colour indices
game_ids  = np.array([unique_games.index(g)  for g in games_event])
half_ids  = np.array([unique_halves.index(h) for h in halves_event])

game_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap_game(i), markersize=8, label=f"Game {unique_games[i]}")
    for i in range(n_games)
]
half_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap_half(i), markersize=8, label=f"Half {unique_halves[i]}")
    for i in range(2)
]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("(D) Event Mean-Pool t-SNE -- Alternative Colorings\n627 events", fontsize=12, fontweight="bold")

panels = [
    (axes[0], le.transform(labels_event), cmap,      n_clusters - 1, legend_handles, "Colored by Event Label"),
    (axes[1], game_ids,                   cmap_game, n_games - 1,    game_handles,   "Colored by Game"),
    (axes[2], half_ids,                   cmap_half, 1,              half_handles,   "Colored by Half"),
]
for ax, color_ids, cm, vmax, handles, title in panels:
    ax.scatter(tsne_B[:, 0], tsne_B[:, 1],
               c=color_ids, cmap=cm, vmin=0, vmax=vmax,
               s=18, alpha=0.8, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    ax.legend(handles=handles, fontsize=7, loc="lower left",
              ncol=2 if len(handles) > 6 else 1,
              framealpha=0.7, handlelength=1)

plt.tight_layout()
path_D = OUT_DIR / "D_colorings_game_half.png"
fig.savefig(path_D, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {path_D}")

# ---------------------------------------------------------------------------
# (E) Per-game highlight plots
#     Global t-SNE (tsne_B) in grey; one game highlighted per figure.
#     Left panel:  highlight coloured by event label
#     Right panel: highlight coloured by half
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("(E) Per-game highlight plots")
print("-" * 60)

per_game_dir = OUT_DIR / "per_game"
per_game_dir.mkdir(exist_ok=True)

for g in unique_games:
    fg_mask = games_event == g          # foreground: this game
    bg_mask = ~fg_mask                  # background: all other games

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Game {g:02d}  --  {fg_mask.sum()} events highlighted on global t-SNE",
                 fontsize=12, fontweight="bold")

    panels_game = [
        (axes[0], le.transform(labels_event[fg_mask]),  cmap,      n_clusters - 1, legend_handles,  "Colored by Event Label"),
        (axes[1], half_ids[fg_mask],                    cmap_half, 1,              half_handles,    "Colored by Half"),
    ]

    for ax, fg_colors, cm, vmax, handles, title in panels_game:
        # Grey background (all other games)
        ax.scatter(tsne_B[bg_mask, 0], tsne_B[bg_mask, 1],
                   c="lightgrey", s=8, alpha=0.3, linewidths=0, zorder=1)
        # Coloured foreground (this game)
        ax.scatter(tsne_B[fg_mask, 0], tsne_B[fg_mask, 1],
                   c=fg_colors, cmap=cm, vmin=0, vmax=vmax,
                   s=55, alpha=0.95, linewidths=0.4,
                   edgecolors="white", zorder=2)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        ax.legend(handles=handles, fontsize=7, loc="lower left",
                  ncol=2 if len(handles) > 6 else 1,
                  framealpha=0.7, handlelength=1)

    plt.tight_layout()
    path_game = per_game_dir / f"game_{g:02d}.png"
    fig.savefig(path_game, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path_game}")

print("\nDone.")

# ---------------------------------------------------------------------------
# (F) Normalized event mean-pool  (per-game mean subtraction)
#     Recomputes k-means + t-SNE after removing game-level visual bias.
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("(F) Normalized event mean-pool  (per-game centering)")
print("-" * 60)

feat_mean_norm = normalize_per_game(feat_mean, games_event)

cluster_F, metrics_F = run_kmeans(feat_mean_norm, labels_event, "Normalized mean-pool")
tsne_F = run_tsne(feat_mean_norm, "Normalized mean-pool")

make_figure(
    tsne_F,
    true_ids    = le.transform(labels_event),
    cluster_ids = cluster_F,
    suptitle    = f"(F) Normalized Event Mean-Pool  --  per-game centering\n"
                  f"{len(feat_mean_norm)} events  |  "
                  f"Purity={metrics_F['Purity']:.3f}  ARI={metrics_F['ARI']:.3f}  "
                  f"Silhouette={metrics_F['Silhouette']:.3f}",
    filename    = "F_normalized_event_meanpool.png",
)

# 3-panel coloring: label | game | half  (same layout as section D)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    "(F) Normalized Event Mean-Pool t-SNE -- Alternative Colorings\n"
    f"{len(feat_mean_norm)} events  (per-game centroid subtracted)",
    fontsize=12, fontweight="bold",
)

panels_F = [
    (axes[0], le.transform(labels_event), cmap,      n_clusters - 1, legend_handles, "Colored by Event Label"),
    (axes[1], game_ids,                   cmap_game, n_games - 1,    game_handles,   "Colored by Game"),
    (axes[2], half_ids,                   cmap_half, 1,              half_handles,   "Colored by Half"),
]
for ax, color_ids, cm, vmax, handles, title in panels_F:
    ax.scatter(tsne_F[:, 0], tsne_F[:, 1],
               c=color_ids, cmap=cm, vmin=0, vmax=vmax,
               s=18, alpha=0.8, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    ax.legend(handles=handles, fontsize=7, loc="lower left",
              ncol=2 if len(handles) > 6 else 1,
              framealpha=0.7, handlelength=1)

plt.tight_layout()
path_F2 = OUT_DIR / "F_normalized_colorings_game_half.png"
fig.savefig(path_F2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {path_F2}")

# Update metrics CSV with section F row
metrics_df = pd.DataFrame([metrics_A, metrics_B, metrics_C, metrics_F])
metrics_df.to_csv(metrics_path, index=False)

print("\n" + "=" * 60)
print("UPDATED METRICS SUMMARY  (A/B/C/F)")
print("=" * 60)
print(metrics_df.to_string(index=False, float_format="%.4f"))
print(f"\nSaved -> {metrics_path}")
