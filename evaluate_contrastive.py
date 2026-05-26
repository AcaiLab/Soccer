"""
Post-Contrastive Clustering Evaluation
=======================================
Loads the projected embeddings from contrastive training and runs
K-Means + t-SNE analysis, comparing against the original DINOv2
embeddings (before contrastive learning).

Produces:
  - Side-by-side t-SNE: before vs after contrastive learning
  - Metrics comparison table & bar chart
  - Colorings by event label, game, half

Usage:
    python evaluate_contrastive.py
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
parser.add_argument("--dino-features", default="features/temporal_dinov2-base")
parser.add_argument("--contrastive-features", default="features/contrastive")
parser.add_argument("--out-dir", default="clustering/contrastive")
args = parser.parse_args()

dino_dir = pathlib.Path(args.dino_features)
contrastive_dir = pathlib.Path(args.contrastive_features)
out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load features
# ---------------------------------------------------------------------------
print("Loading features ...")

# Original DINOv2 mean-pooled
npz_dino = np.load(dino_dir / "clip_temporal_padded.npz", allow_pickle=True)
feat_padded = npz_dino["features"].astype(np.float32)
masks = npz_dino["masks"]
labels = npz_dino["labels"].astype(str)
games = npz_dino["games"].astype(str)
halves = npz_dino["halves"].astype(str)

n_valid = masks.sum(axis=1, keepdims=True).clip(min=1)
feat_dino = (feat_padded * masks[:, :, np.newaxis]).sum(axis=1) / n_valid
feat_dino = feat_dino / np.linalg.norm(feat_dino, axis=1, keepdims=True)

# Contrastive projected
npz_cont = np.load(contrastive_dir / "contrastive_embeddings.npz", allow_pickle=True)
feat_cont = npz_cont["features"].astype(np.float32)  # (627, 256)

print(f"  DINOv2 original:   {feat_dino.shape}")
print(f"  Contrastive proj:  {feat_cont.shape}")
print(f"  Events: {len(labels)}")

# ---------------------------------------------------------------------------
# Label setup
# ---------------------------------------------------------------------------
all_labels = sorted(set(labels))
le = LabelEncoder().fit(all_labels)
n_clusters = len(all_labels)
true_ids = le.transform(labels)

cmap = plt.get_cmap("tab20", n_clusters)
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap(i), markersize=7, label=all_labels[i])
    for i in range(n_clusters)
]

unique_games = sorted(set(games))
n_games = len(unique_games)
cmap_game = plt.get_cmap("tab10", n_games)
game_ids = np.array([unique_games.index(g) for g in games])
game_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap_game(i), markersize=8, label=f"Game {unique_games[i]}")
    for i in range(n_games)
]


def cluster_purity(true_labels, cluster_ids):
    correct = 0
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        counts = np.bincount(le.transform(true_labels[mask]), minlength=n_clusters)
        correct += counts.max()
    return correct / len(true_labels)


def run_analysis(features, tag):
    """K-Means + t-SNE on a feature set, return metrics + t-SNE coords."""
    print(f"\n[{tag}]")

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=15)
    cluster_ids = km.fit_predict(features)

    ari = adjusted_rand_score(labels, cluster_ids)
    sil = silhouette_score(features, cluster_ids,
                           sample_size=min(2000, len(features)),
                           random_state=RANDOM_STATE)
    pur = cluster_purity(labels, cluster_ids)
    print(f"  Purity={pur:.3f}  ARI={ari:.3f}  Silhouette={sil:.3f}")

    tsne_xy = TSNE(n_components=2, perplexity=30, max_iter=1000,
                   random_state=RANDOM_STATE).fit_transform(features)

    return {
        "Representation": tag,
        "Purity": pur, "ARI": ari, "Silhouette": sil,
        "cluster_ids": cluster_ids, "tsne": tsne_xy,
    }


# ---------------------------------------------------------------------------
# Run analysis on both
# ---------------------------------------------------------------------------
res_dino = run_analysis(feat_dino, "DINOv2 (before)")
res_cont = run_analysis(feat_cont, "Contrastive (after)")

# ---------------------------------------------------------------------------
# Plot 1: Before vs After t-SNE (true labels)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Contrastive Learning: Before vs After\nt-SNE colored by Event Label",
             fontsize=13, fontweight="bold")

for ax, res, title in [
    (axes[0], res_dino, f"DINOv2 Mean-Pool (before)\nPurity={res_dino['Purity']:.3f}  ARI={res_dino['ARI']:.3f}"),
    (axes[1], res_cont, f"After Triplet Loss\nPurity={res_cont['Purity']:.3f}  ARI={res_cont['ARI']:.3f}"),
]:
    ax.scatter(res["tsne"][:, 0], res["tsne"][:, 1],
               c=true_ids, cmap=cmap, vmin=0, vmax=n_clusters - 1,
               s=25, alpha=0.8, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

axes[0].legend(handles=legend_handles, fontsize=6, loc="lower left",
               ncol=2, framealpha=0.7, handlelength=1)

plt.tight_layout()
fig.savefig(out_dir / "before_vs_after_labels.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_dir / 'before_vs_after_labels.png'}")

# ---------------------------------------------------------------------------
# Plot 2: Before vs After t-SNE (game coloring)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Contrastive Learning: Before vs After\nt-SNE colored by Game",
             fontsize=13, fontweight="bold")

for ax, res, title in [
    (axes[0], res_dino, "DINOv2 Mean-Pool (before)"),
    (axes[1], res_cont, "After Triplet Loss"),
]:
    ax.scatter(res["tsne"][:, 0], res["tsne"][:, 1],
               c=game_ids, cmap=cmap_game, vmin=0, vmax=n_games - 1,
               s=25, alpha=0.8, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

axes[0].legend(handles=game_handles, fontsize=7, loc="lower left",
               ncol=1, framealpha=0.7, handlelength=1)

plt.tight_layout()
fig.savefig(out_dir / "before_vs_after_games.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_dir / 'before_vs_after_games.png'}")

# ---------------------------------------------------------------------------
# Plot 3: After contrastive — 3-panel (label | game | k-means)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Post-Contrastive Embeddings (Triplet Loss)\n"
             f"{len(labels)} events  |  "
             f"Purity={res_cont['Purity']:.3f}  ARI={res_cont['ARI']:.3f}  "
             f"Silhouette={res_cont['Silhouette']:.3f}",
             fontsize=12, fontweight="bold")

panels = [
    (axes[0], true_ids,              cmap,      n_clusters - 1, legend_handles, "Colored by Event Label"),
    (axes[1], game_ids,              cmap_game, n_games - 1,    game_handles,   "Colored by Game"),
    (axes[2], res_cont["cluster_ids"], cmap,    n_clusters - 1, [],             f"K-Means (k={n_clusters})"),
]
for ax, color_ids, cm, vmax, handles, title in panels:
    ax.scatter(res_cont["tsne"][:, 0], res_cont["tsne"][:, 1],
               c=color_ids, cmap=cm, vmin=0, vmax=vmax,
               s=25, alpha=0.8, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    if handles:
        ax.legend(handles=handles, fontsize=6, loc="lower left",
                  ncol=2 if len(handles) > 6 else 1,
                  framealpha=0.7, handlelength=1)

plt.tight_layout()
fig.savefig(out_dir / "contrastive_3panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_dir / 'contrastive_3panel.png'}")

# ---------------------------------------------------------------------------
# Plot 4: Metrics comparison bar chart
# ---------------------------------------------------------------------------
metrics_data = [
    {"Representation": res_dino["Representation"],
     "Purity": res_dino["Purity"], "ARI": res_dino["ARI"],
     "Silhouette": res_dino["Silhouette"]},
    {"Representation": res_cont["Representation"],
     "Purity": res_cont["Purity"], "ARI": res_cont["ARI"],
     "Silhouette": res_cont["Silhouette"]},
]
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(out_dir / "contrastive_metrics.csv", index=False)

x = np.arange(3)
width = 0.35
metric_names = ["Purity", "ARI", "Silhouette"]
before_vals = [res_dino["Purity"], res_dino["ARI"], res_dino["Silhouette"]]
after_vals = [res_cont["Purity"], res_cont["ARI"], res_cont["Silhouette"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, before_vals, width, label="DINOv2 (before)", color="#1f77b4", alpha=0.8)
bars2 = ax.bar(x + width/2, after_vals, width, label="Contrastive (after)", color="#2ca02c", alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Clustering Metrics: Before vs After Contrastive Learning",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(out_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_dir / 'metrics_comparison.png'}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONTRASTIVE LEARNING EVALUATION SUMMARY")
print("=" * 60)
print(metrics_df.to_string(index=False, float_format="%.4f"))

for metric in metric_names:
    before = metrics_data[0][metric]
    after = metrics_data[1][metric]
    pct = (after - before) / before * 100 if before != 0 else 0
    direction = "+" if pct > 0 else ""
    print(f"\n  {metric}: {before:.4f} -> {after:.4f}  ({direction}{pct:.1f}%)")

print(f"\nAll outputs saved to {out_dir}/")
print("Done.")
