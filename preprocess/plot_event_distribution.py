"""Event-type distribution over the manifest's games, from the annotation JSONs.

Counts every timestamped first- and second-half comment by sanitised label
(the same labels the frame folders use) and saves a bar chart.

Usage:
    python preprocess/plot_event_distribution.py
    python preprocess/plot_event_distribution.py --split test --out test_distribution.png
"""

import argparse
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from soccer.annotations import events_for_half, load_comments  # noqa: E402
from soccer.config import data_path, load_config, repo_path  # noqa: E402
from soccer.manifest import load_manifest  # noqa: E402
from soccer.pipeline import annotation_for  # noqa: E402


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Plot the event label distribution of manifest games.")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="default: all games")
    parser.add_argument("--out", default="preprocess/event_distribution.png")
    args = parser.parse_args(argv)

    games = [g for g in load_manifest(repo_path(cfg, "manifest")) if args.split in (None, g.split)]
    counts = Counter()
    for g in games:
        comments = load_comments(annotation_for(data_path(cfg, "annotations"), g))
        counts.update(e.label for half in (1, 2) for e in events_for_half(comments, half))

    labels = [label for label, _ in counts.most_common()]
    values = [counts[label] for label in labels]
    print(f"{'Event Type':<30s}  {'Count':>5s}\n" + "-" * 40)
    for label, n in zip(labels, values, strict=True):
        print(f"{label:<30s}  {n:>5d}")
    print("-" * 40 + f"\n{'TOTAL':<30s}  {sum(values):>5d}")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), values, color="#3498db")
    ax.bar_label(bars, fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("# Events")
    ax.set_title(f"Event distribution by label - {len(games)} games ({args.split or 'all splits'})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
