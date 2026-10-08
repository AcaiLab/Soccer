"""
Comparison plot: DINOv2 vs CLIP clustering metrics across representations.

Reads the clustering_metrics.csv that cluster_analysis.py wrote for each model,
instead of numbers typed into this file, so the figure always matches the run.

Usage:
    python cluster_analysis.py --model openai/clip-vit-base-patch32
    python cluster_analysis.py --model facebook/dinov2-base
    python plot_dinov2_vs_clip.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from soccer.config import data_path, load_config  # noqa: E402

METRICS = ["ARI", "Purity", "Silhouette"]


def main(argv=None):
    cfg = load_config()
    results = data_path(cfg, "results")
    parser = argparse.ArgumentParser(description="Compare clustering metrics of two models.")
    parser.add_argument("--clip", default=str(results / "clip-vit-base-patch32" / "clustering_metrics.csv"))
    parser.add_argument("--dino", default=str(results / "dinov2-base" / "clustering_metrics.csv"))
    parser.add_argument("--out", default=str(results / "dinov2_vs_clip_comparison.png"))
    args = parser.parse_args(argv)

    clip = pd.read_csv(args.clip).set_index("Representation")
    dino = pd.read_csv(args.dino).set_index("Representation")
    reps = [r for r in dino.index if r in clip.index]
    x, width = np.arange(len(reps)), 0.35

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 5))
    fig.suptitle("DINOv2 vs CLIP: clustering against event labels", fontsize=14, fontweight="bold", y=1.02)
    for ax, metric in zip(axes, METRICS, strict=True):
        for offset, (name, df, color) in zip((-width / 2, width / 2),
                                             (("CLIP", clip, "#1f77b4"), ("DINOv2", dino, "#ff7f0e")), strict=True):
            bars = ax.bar(x + offset, df.loc[reps, metric], width, label=name, color=color, alpha=0.8)
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        if metric == "Purity" and "Purity_random" in dino:
            ax.scatter(x, dino.loc[reps, "Purity_random"], marker="_", s=400, color="black",
                       label="random purity (DINOv2)")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xticks(x, [r.replace(" ", "\n", 1) for r in reps], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
