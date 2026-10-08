"""
Temporal Event Classifier (Annotation-Only)
=============================================
Trains classifiers on game flow features from JSON annotations (no video).

Compares feature subsets:
  (1) Baseline: half + minute
  (2) + previous 3 events
  (3) + rolling 5-minute counts
  (4) Full: all temporal features

Protocol: one league-season is held out for testing. Within the remaining
leagues, 20% of matches (grouped by match) form a validation set used to pick
the feature subset and classifier; the chosen configuration is retrained on
all training matches and scored once on the held-out league.

Previous-event features use true labels (teacher forcing), so the accuracy
is an upper bound for a deployed system (docs/leak_audit.md, B9).

--cross-league repeats the test for every league-season with the chosen
configuration and writes cross_league_results.csv.

Usage:
    python scripts/game_flow_features.py --all-leagues
    python scripts/temporal_event_classifier.py
    python scripts/temporal_event_classifier.py --cross-league
"""

import argparse
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from soccer.config import data_path, dropped_labels, load_config  # noqa: E402
from soccer.game_flow import CATEGORICAL, FEATURES, NUMERIC, encoder  # noqa: E402
from soccer.metrics import classification_scores  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

PREV = ["prev_1", "prev_2", "prev_3", "prev_1_cat", "prev_2_cat", "prev_3_cat"]
ROLLING = [c for c in NUMERIC if c.startswith("recent_")]
SUBSETS = {
    "Baseline (half+minute)": ["half", "minute"],
    "+ Previous events": ["half", "minute", *PREV],
    "+ Rolling counts": ["half", "minute", *ROLLING],
    "Full temporal (all features)": FEATURES,
}


def make_model(name: str, columns: list[str], seed: int) -> Pipeline:
    if name == "LogReg":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    elif name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6, random_state=seed, learning_rate=0.1)
    return Pipeline([("encode", encoder(columns, one_hot=name == "LogReg")), ("clf", clf)])


CLASSIFIERS = ("LogReg", "RandomForest", "HistGradBoost")


def fit_score(df, train, test, subset, clf_name, seed):
    cols = SUBSETS[subset]
    model = make_model(clf_name, cols, seed).fit(df.loc[train, cols], df.loc[train, "label"])
    known = test & df["label"].isin(set(df.loc[train, "label"]))
    pred = model.predict(df.loc[known, cols])
    return model, known, pred, classification_scores(df.loc[known, "label"], pred, df.loc[train, "label"])


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Annotation-only event classification across leagues.")
    parser.add_argument("--features", default=str(data_path(cfg, "features") / "game_flow" / "game_flow_features.csv"))
    parser.add_argument("--out-dir", default=str(data_path(cfg, "results") / "classification_annotation_only"))
    parser.add_argument("--test-league", default="england_epl_2021-2022")
    parser.add_argument("--cross-league", action="store_true", help="also hold out every league-season in turn")
    parser.add_argument("--keep-all-labels", action="store_true")
    args = parser.parse_args(argv)

    seed = cfg["seed"]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features, keep_default_na=False)
    if not args.keep_all_labels:
        df = df[~df["label"].isin(dropped_labels(cfg))].reset_index(drop=True)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    print(f"Events: {len(df)}  matches: {df['match_id'].nunique()}  leagues: {df['league'].nunique()}")

    test = (df["league"] == args.test_league).values
    if not test.any():
        raise SystemExit(f"No events for test league {args.test_league!r}")
    pool = np.flatnonzero(~test)
    tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                      .split(pool, groups=df.loc[pool, "match_id"]))
    train, val = np.zeros(len(df), bool), np.zeros(len(df), bool)
    train[pool[tr_i]], val[pool[va_i]] = True, True
    print(f"Train {train.sum()} / val {val.sum()} events (val = 20% of training matches); "
          f"test {test.sum()} events from {args.test_league}")

    rows = []
    for subset in SUBSETS:
        for clf_name in CLASSIFIERS:
            _, _, _, s = fit_score(df, train, val, subset, clf_name, seed)
            rows.append({"Features": subset, "Classifier": clf_name,
                         "Val_accuracy": s["Accuracy"], "Val_macro_F1": s["Macro_F1"]})
            print(f"  {subset:<30s} {clf_name:<14s} val acc={s['Accuracy']:.3f}  macro-F1={s['Macro_F1']:.3f}")
    val_df = pd.DataFrame(rows).sort_values("Val_accuracy", ascending=False)
    val_df.to_csv(out_dir / "temporal_classification_results.csv", index=False)
    best = val_df.iloc[0]

    model, known, pred, s = fit_score(df, train | val, test, best["Features"], best["Classifier"], seed)
    summary = (f"Chosen on validation: {best['Features']} + {best['Classifier']}\n"
               f"Test league: {args.test_league} ({known.sum()} events)\n"
               f"Test accuracy {s['Accuracy']:.4f}  macro-F1 {s['Macro_F1']:.4f}  "
               f"majority-class baseline {s['Majority_baseline']:.4f}\n"
               "Upper bound: previous-event features use true labels (teacher forcing).\n")
    report = classification_report(df.loc[known, "label"], pred, zero_division=0)
    (out_dir / "temporal_classification_report.txt").write_text(summary + "\n" + report)
    print("\n" + summary)

    if best["Classifier"] == "RandomForest":
        names = model.named_steps["encode"].get_feature_names_out()
        imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        imp.plot.bar(ax=ax, color="#1f77b4", alpha=0.8)
        ax.set_title("Feature importance (chosen Random Forest)", fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    best_per = val_df.groupby("Features")["Val_accuracy"].max().reindex(list(SUBSETS))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(best_per)), best_per.values, color="#1f77b4", alpha=0.85)
    ax.bar_label(bars, labels=[f"{v * 100:.1f}%" for v in best_per.values])
    ax.set_xticks(range(len(best_per)), [n.replace(" (", "\n(") for n in best_per.index], fontsize=9)
    ax.set_ylabel("Validation accuracy (best classifier)")
    ax.set_title(f"Annotation-only classification by feature subset\n(chosen config: test accuracy "
                 f"{s['Accuracy'] * 100:.1f}% on {args.test_league}, majority {s['Majority_baseline'] * 100:.1f}%)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "temporal_feature_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if args.cross_league:
        cross = []
        for league in sorted(df["league"].unique()):
            held = (df["league"] == league).values
            _, known_l, _, sl = fit_score(df, ~held, held, best["Features"], best["Classifier"], seed)
            cross.append({"league": league, "events": int(known_l.sum()), "accuracy": sl["Accuracy"],
                          "macro_F1": sl["Macro_F1"], "majority_baseline": sl["Majority_baseline"]})
            print(f"  held out {league:<38s} acc={sl['Accuracy']:.3f}  majority={sl['Majority_baseline']:.3f}")
        pd.DataFrame(cross).to_csv(out_dir / "cross_league_results.csv", index=False)
    print(f"\nAll results in {out_dir}/")


if __name__ == "__main__":
    main()
