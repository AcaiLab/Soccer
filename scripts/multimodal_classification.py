"""
Multimodal Event Classifier
============================
Combines DINOv2 visual features (mean-pooled over the event window) with
annotation-only game flow features (temporal context) for event classification.

Compares three feature sets:
  (1) Visual only     - DINOv2 mean-pooled
  (2) Annotation only - game flow context (soccer/game_flow.py)
  (3) Multimodal      - both, concatenated

Events are matched across the two sources on (match, half, minute); events
without a partner are left out, so the test set is smaller than in
event_classification.py and the two are not directly comparable.

Protocol: train / val / test games from the manifest; every configuration is
ranked on validation and only the chosen one is scored on test. Game-flow
categories are encoded with an encoder fit on training events only.
Annotation features use true previous labels, so they are an upper bound.

Usage:
    python scripts/game_flow_features.py
    python scripts/multimodal_classification.py
"""

import argparse
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

from soccer.annotations import match_minute, tag_to_seconds  # noqa: E402
from soccer.config import data_path, dropped_labels, load_config, repo_path  # noqa: E402
from soccer.evaluation import (  # noqa: E402
    format_test_summary,
    save_test_predictions,
    select_on_validation,
    split_by_game,
)
from soccer.features import load_temporal, mean_pool  # noqa: E402
from soccer.game_flow import CATEGORICAL, FEATURES, encoder  # noqa: E402
from soccer.manifest import (
    load_manifest,  # noqa: E402
    manifest_games,  # noqa: E402
)

warnings.filterwarnings("ignore", category=UserWarning)


def main(argv=None):
    cfg = load_config()
    prefix = cfg["features"]["model"].split("/")[-1]
    features_root = data_path(cfg, "features")
    parser = argparse.ArgumentParser(description="Visual + annotation event classification.")
    parser.add_argument("--dino-features", default=str(features_root / f"temporal_{prefix}"))
    parser.add_argument("--game-flow-features", default=str(features_root / "game_flow" / "game_flow_features.csv"))
    parser.add_argument("--out-dir", default=str(data_path(cfg, "results") / "classification"))
    parser.add_argument("--val-games", nargs="+", default=None)
    parser.add_argument("--test-games", nargs="+", default=None)
    parser.add_argument("--keep-all-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg["seed"], help="seed for the classifiers (MLP initialisation)")
    args = parser.parse_args(argv)

    seed = args.seed
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(repo_path(cfg, "manifest"))
    match_of = {g.game_num: g.match_id for g in manifest}

    tf = load_temporal(args.dino_features, games=manifest_games(cfg))
    if not args.keep_all_labels:
        tf = tf.drop_labels(dropped_labels(cfg))
    gf = pd.read_csv(args.game_flow_features, keep_default_na=False)
    for c in CATEGORICAL:
        gf[c] = gf[c].astype(str)
    # The label is part of the key: different events often share a minute
    # (e.g. corner and lead_to_corner), and keying on time alone dropped one of them.
    gf_index = {}
    for i, r in enumerate(gf.itertuples()):
        gf_index.setdefault((r.match_id, int(r.half), round(float(r.minute), 2), r.label), i)

    pairs = []
    for i in range(len(tf)):
        half = int(tf.halves[i])
        key = (match_of.get(tf.games[i]), half, match_minute(tag_to_seconds(tf.event_tags[i]), half), tf.labels[i])
        j = gf_index.get(key)
        if j is not None:
            pairs.append((i, j))
    if not pairs:
        raise SystemExit("No events matched between the visual and annotation features")
    vis_idx, gf_idx = map(np.array, zip(*pairs, strict=True))
    print(f"Matched {len(pairs)} / {len(tf)} visual events to annotation events")

    labels, games = tf.labels[vis_idx], tf.games[vis_idx]
    split = split_by_game(games, labels, manifest, args.val_games, args.test_games)
    X_vis = mean_pool(tf.features, tf.masks)[vis_idx]
    enc = encoder(FEATURES, one_hot=True).fit(gf.loc[gf_idx[split.train], FEATURES])
    X_gf = enc.transform(gf.loc[gf_idx, FEATURES])
    X_gf = X_gf.toarray() if hasattr(X_gf, "toarray") else X_gf
    feature_sets = {
        "Visual only (DINOv2 mean)": X_vis,
        "Annotation only (game flow)": X_gf.astype(np.float32),
        "Multimodal (visual + annot)": np.hstack([X_vis, X_gf]).astype(np.float32),
    }
    print(f"Events: train={split.train.sum()} val={split.val.sum()} test={split.test.sum()}")

    rows, chosen = select_on_validation(feature_sets, labels, split, seed, games=games)
    pd.DataFrame(rows).sort_values("Val_accuracy", ascending=False).to_csv(
        out_dir / "multimodal_results.csv", index=False)
    y_true, y_pred = chosen.pop("y_true"), chosen.pop("y_pred")
    test_idx = vis_idx[split.test]
    save_test_predictions(out_dir / "test_predictions.csv", tf.games[test_idx], tf.halves[test_idx],
                          tf.event_tags[test_idx], y_true, y_pred, seed=seed)
    summary = format_test_summary(chosen, split, len(y_true), "matched events")
    report = classification_report(y_true, y_pred, zero_division=0)
    (out_dir / "multimodal_classification_report.txt").write_text(summary + "\n" + report)
    print("\n" + summary + "\n" + report)

    best_val = pd.DataFrame(rows).groupby("Features")["Val_accuracy"].max().reindex(list(feature_sets))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(best_val)), best_val.values, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.85)
    ax.bar_label(bars, labels=[f"{v * 100:.1f}%" for v in best_val.values], fontsize=11)
    ax.axhline(chosen["Test_Majority_baseline"], color="red", linestyle="--", label="majority class (test)")
    ax.set_xticks(range(len(best_val)), [n.replace(" (", "\n(") for n in best_val.index], fontsize=10)
    ax.set_ylabel("Validation accuracy (best classifier)")
    ax.set_title(f"Multimodal vs unimodal (chosen: {chosen['Features']}, test accuracy "
                 f"{chosen['Test_Accuracy'] * 100:.1f}%)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "multimodal_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Done. All results in {out_dir}/")


if __name__ == "__main__":
    main()
