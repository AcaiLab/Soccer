"""
Event Classification Pipeline
==============================
Evaluates event classification accuracy using DINOv2 temporal features
with several pooling strategies and classifiers.

Compares:
  - Pooling strategies: mean, max, first frame, last frame
  - Classifiers: Linear SVM, Logistic Regression, MLP
  - Embeddings: raw DINOv2 vs contrastive-projected (if available)

Protocol: games are split train / val / test by the manifest. Every
configuration is fit on train and ranked on val; only the chosen one is
scored on test, together with macro-F1 and the majority-class baseline.

Usage:
    python scripts/event_classification.py
    python scripts/event_classification.py --test-games 25 26 27 28 29 30
"""

import argparse
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix  # noqa: E402

from soccer.config import data_path, dropped_labels, load_config, repo_path  # noqa: E402
from soccer.evaluation import format_test_summary, select_on_validation, split_by_game  # noqa: E402
from soccer.features import POOLING, load_temporal  # noqa: E402
from soccer.manifest import assert_disjoint, load_manifest  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def load_contrastive(path: pathlib.Path, tf, held_out_games):
    """Projected embeddings aligned to tf, refusing heads trained on held-out games."""
    if not path.is_file():
        return None
    cont = np.load(path)
    if "train_games" not in cont.files:
        print(f"  Ignoring {path}: no record of its training games (old, leaky training run)")
        return None
    assert_disjoint(head_training=set(cont["train_games"].astype(str)), held_out=held_out_games)
    if not np.array_equal(cont["event_tags"].astype(str), tf.event_tags) or \
            not np.array_equal(cont["games"].astype(str), tf.games):
        raise SystemExit("Contrastive embeddings do not line up with the DINOv2 events; regenerate them.")
    return cont["features"].astype(np.float32)


def main(argv=None):
    cfg = load_config()
    prefix = cfg["features"]["model"].split("/")[-1]
    features_root = data_path(cfg, "features")
    parser = argparse.ArgumentParser(description="Event classification with pooled frozen features.")
    parser.add_argument("--dino-features", default=str(features_root / f"temporal_{prefix}"))
    parser.add_argument("--contrastive-features", default=str(features_root / "contrastive"))
    parser.add_argument("--out-dir", default=str(data_path(cfg, "results") / "classification"))
    parser.add_argument("--val-games", nargs="+", default=None, help="default: manifest val split")
    parser.add_argument("--test-games", nargs="+", default=None, help="default: manifest test split")
    parser.add_argument("--keep-all-labels", action="store_true")
    args = parser.parse_args(argv)

    seed = cfg["seed"]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = load_temporal(args.dino_features)
    if not args.keep_all_labels:
        tf = tf.drop_labels(dropped_labels(cfg))
    split = split_by_game(tf.games, tf.labels, load_manifest(repo_path(cfg, "manifest")),
                          args.val_games, args.test_games)
    print(f"Events: train={split.train.sum()} val={split.val.sum()} test={split.test.sum()}")

    feature_sets = {f"DINOv2 {name}": fn(tf.features, tf.masks) for name, fn in POOLING.items()}
    cont = load_contrastive(pathlib.Path(args.contrastive_features) / "contrastive_embeddings.npz", tf,
                            set(tf.games[split.val | split.test]))
    if cont is not None:
        feature_sets["Contrastive (triplet head)"] = cont

    rows, chosen = select_on_validation(feature_sets, tf.labels, split, seed, games=tf.games)
    pd.DataFrame(rows).sort_values("Val_accuracy", ascending=False).to_csv(
        out_dir / "classification_results.csv", index=False)

    y_true, y_pred = chosen.pop("y_true"), chosen.pop("y_pred")
    summary = format_test_summary(chosen, split, len(y_true), "events")
    present = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=present, zero_division=0)
    (out_dir / "classification_report.txt").write_text(summary + "\n" + report)
    print("\n" + summary + "\n" + report)

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(14, 6))
    best_val = df.groupby("Features")["Val_accuracy"].max().reindex(list(feature_sets))
    bars = ax.bar(range(len(best_val)), best_val.values,
                  color=["#2ca02c" if f.startswith("Contrastive") else "#1f77b4" for f in best_val.index], alpha=0.8)
    ax.bar_label(bars, labels=[f"{v * 100:.1f}%" for v in best_val.values], fontsize=9)
    ax.axhline(chosen["Test_Majority_baseline"], color="red", linestyle="--", label="majority class (test)")
    ax.scatter([list(feature_sets).index(chosen["Features"])], [chosen["Test_Accuracy"]], color="black",
               zorder=3, label=f"chosen config, test accuracy {chosen['Test_Accuracy'] * 100:.1f}%")
    ax.set_xticks(range(len(best_val)), [f.replace(" ", "\n", 1) for f in best_val.index], fontsize=9)
    ax.set_ylabel("Validation accuracy (best classifier)")
    ax.set_title("Event classification by representation (bars: validation; dot: test of the chosen config)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "classification_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=present), display_labels=present).plot(
        ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix (test) - {chosen['Features']} + {chosen['Classifier']}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"All results in {out_dir}/")


if __name__ == "__main__":
    main()
