"""Events and frames per label in the extracted frames folder.

Reads <frames root>/game_XX_half_N/frames/<label>/ for every game and saves a
two-panel bar chart; useful to spot labels with few events, or events with
fewer frames than the window should give.

Usage:
    python preprocess/plot_labels.py
"""

import argparse
import pathlib
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from soccer.config import data_path, load_config  # noqa: E402
from soccer.frames import scan_events  # noqa: E402


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Count events and frames per label.")
    parser.add_argument("--frames-root", default=str(data_path(cfg, "frames")))
    parser.add_argument("--out", default="preprocess/labels_frames.png")
    args = parser.parse_args(argv)

    events = scan_events(pathlib.Path(args.frames_root))
    if not events:
        raise SystemExit(f"No frames found under {args.frames_root}")
    n_events = Counter(k[2] for k in events)
    n_frames = Counter()
    for k, frames in events.items():
        n_frames[k[2]] += len(frames)
    labels = sorted(n_events, key=n_events.get, reverse=True)

    print(f"{'Label':<26s} {'Events':>7s} {'Frames':>7s} {'Frames/event':>13s}")
    for label in labels:
        print(f"{label:<26s} {n_events[label]:>7d} {n_frames[label]:>7d} {n_frames[label] / n_events[label]:>13.1f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, counts, title in [(axes[0], n_events, "Events per label"), (axes[1], n_frames, "Frames per label")]:
        ax.bar(labels, [counts[label] for label in labels])
        ax.set_title(title, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
