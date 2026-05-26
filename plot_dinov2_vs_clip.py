"""
Comparison plot: DINOv2 vs CLIP clustering metrics across 4 representations.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
representations = [
    "Individual\nframes",
    "Event\nmean-pool",
    "Event\nconcat+PCA",
    "Normalized\nmean-pool",
]

# CLIP baseline (from original clustering)
clip_purity = [0.213, 0.250, 0.284, 0.314]
clip_ari = [0.025, 0.026, 0.051, 0.065]
clip_silhouette = [0.086, 0.184, 0.063, 0.034]

# DINOv2 results
dinov2_purity = [0.199, 0.271, 0.324, 0.289]
dinov2_ari = [0.017, 0.036, 0.072, 0.047]
dinov2_silhouette = [0.065, 0.059, 0.034, 0.048]

x = np.arange(len(representations))
width = 0.35

# Create 3 subplots (one for each metric)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("DINOv2 vs CLIP: Clustering Performance Across Representations",
             fontsize=14, fontweight="bold", y=1.02)

# --- Purity ---
ax = axes[0]
bars1 = ax.bar(x - width/2, clip_purity, width, label="CLIP", color="#1f77b4", alpha=0.8)
bars2 = ax.bar(x + width/2, dinov2_purity, width, label="DINOv2", color="#ff7f0e", alpha=0.8)
ax.set_ylabel("Purity", fontsize=11, fontweight="bold")
ax.set_title("Purity (higher is better)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(representations, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.35)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# --- ARI ---
ax = axes[1]
bars1 = ax.bar(x - width/2, clip_ari, width, label="CLIP", color="#1f77b4", alpha=0.8)
bars2 = ax.bar(x + width/2, dinov2_ari, width, label="DINOv2", color="#ff7f0e", alpha=0.8)
ax.set_ylabel("Adjusted Rand Index", fontsize=11, fontweight="bold")
ax.set_title("ARI (higher is better)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(representations, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.075)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

# --- Silhouette ---
ax = axes[2]
bars1 = ax.bar(x - width/2, clip_silhouette, width, label="CLIP", color="#1f77b4", alpha=0.8)
bars2 = ax.bar(x + width/2, dinov2_silhouette, width, label="DINOv2", color="#ff7f0e", alpha=0.8)
ax.set_ylabel("Silhouette Score", fontsize=11, fontweight="bold")
ax.set_title("Silhouette (higher is better)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(representations, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.20)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("clustering/dinov2_vs_clip_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: clustering/dinov2_vs_clip_comparison.png")
plt.close()

# --- Improvement heatmap ---
fig, ax = plt.subplots(figsize=(10, 4))

improvements = np.array([
    [(dinov2_purity[i] - clip_purity[i]) / clip_purity[i] * 100 for i in range(4)],
    [(dinov2_ari[i] - clip_ari[i]) / clip_ari[i] * 100 for i in range(4)],
    [(dinov2_silhouette[i] - clip_silhouette[i]) / clip_silhouette[i] * 100 for i in range(4)],
]).T

metrics = ["Purity", "ARI", "Silhouette"]
im = ax.imshow(improvements, cmap="RdYlGn", aspect="auto", vmin=-40, vmax=40)

ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(representations)))
ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
ax.set_yticklabels(representations, fontsize=11)

# Add percentage labels
for i in range(len(representations)):
    for j in range(len(metrics)):
        val = improvements[i, j]
        color = "white" if abs(val) > 20 else "black"
        text = ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                      color=color, fontsize=10, fontweight="bold")

ax.set_title("DINOv2 vs CLIP: % Improvement (+green, -red)",
             fontsize=13, fontweight="bold", pad=15)
cbar = plt.colorbar(im, ax=ax, label="Improvement (%)")
cbar.ax.set_ylabel("Improvement (%)", fontsize=10)

plt.tight_layout()
plt.savefig("clustering/dinov2_improvement_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: clustering/dinov2_improvement_heatmap.png")
plt.close()

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print("\nPurity Improvements (DINOv2 vs CLIP):")
for i, rep in enumerate(representations):
    pct = (dinov2_purity[i] - clip_purity[i]) / clip_purity[i] * 100
    print(f"  {rep.replace(chr(10), ' '):<25s} {pct:+6.1f}%")

print("\nARI Improvements (DINOv2 vs CLIP):")
for i, rep in enumerate(representations):
    pct = (dinov2_ari[i] - clip_ari[i]) / clip_ari[i] * 100
    print(f"  {rep.replace(chr(10), ' '):<25s} {pct:+6.1f}%")

print("\n✓ Best DINOv2 performer: Event concat+PCA")
print(f"  - Purity: {dinov2_purity[2]:.4f} (+14.1% vs CLIP)")
print(f"  - ARI: {dinov2_ari[2]:.4f} (+41.5% vs CLIP)")
