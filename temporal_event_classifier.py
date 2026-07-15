"""
Temporal Event Classifier (Annotation-Only)
=============================================
Trains classifiers on game flow features extracted from JSON annotations.
No video required.

Compares:
  (1) Baseline: current event features only (half + minute)
  (2) + Previous events (prev_1, prev_2, prev_3)
  (3) + Rolling counts (corners, shots, fouls, cards in last 5 min)
  (4) Full: all temporal features combined

Train/test split: leave-N-leagues-out OR leave-N-games-out.

Usage:
    python temporal_event_classifier.py
    python temporal_event_classifier.py --features-dir features/game_flow
"""

import argparse
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--features-dir", default="features/game_flow")
parser.add_argument("--out-dir", default="clustering/classification")
parser.add_argument("--test-league", default="england_epl_2021-2022",
                    help="League to hold out for testing")
args = parser.parse_args()

out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load features
# ---------------------------------------------------------------------------
print("Loading features ...")
npz = np.load(pathlib.Path(args.features_dir) / "game_flow_features.npz",
              allow_pickle=True)

X_all        = npz["X"]
y_all        = npz["y"]
game_ids     = npz["game_ids"]
leagues      = npz["leagues"]
labels_raw   = npz["labels"]
feature_names = npz["feature_names"]
label_classes = npz["label_classes"]

print(f"  Total events:  {len(X_all)}")
print(f"  Features:      {len(feature_names)}")
print(f"  Label classes: {len(label_classes)}")
print(f"  Leagues:       {sorted(set(leagues))[:5]} ...")

# ---------------------------------------------------------------------------
# Feature subsets to compare
# ---------------------------------------------------------------------------
feat_idx = {name: i for i, name in enumerate(feature_names)}

SUBSETS = {
    "Baseline\n(half+minute)": [
        feat_idx["half"], feat_idx["minute"]
    ],
    "+ Previous\nevents": [
        feat_idx["half"], feat_idx["minute"],
        feat_idx["prev_1_enc"], feat_idx["prev_2_enc"], feat_idx["prev_3_enc"],
        feat_idx["prev_1_cat_enc"], feat_idx["prev_2_cat_enc"], feat_idx["prev_3_cat_enc"],
    ],
    "+ Rolling\ncounts": [
        feat_idx["half"], feat_idx["minute"],
        feat_idx["recent_corners"], feat_idx["recent_shots"],
        feat_idx["recent_fouls"], feat_idx["recent_cards"],
        feat_idx["recent_attacks"], feat_idx["recent_stoppages"],
    ],
    "Full temporal\n(all features)": list(range(len(feature_names))),
}

# ---------------------------------------------------------------------------
# Train / test split — leave test league out
# ---------------------------------------------------------------------------
test_mask  = leagues == args.test_league
train_mask = ~test_mask

print(f"\nTrain: {train_mask.sum()} events from {len(set(leagues[train_mask]))} leagues")
print(f"Test:  {test_mask.sum()} events from league '{args.test_league}'")

# Filter out labels in test not seen in train
known_labels = set(y_all[train_mask])
known_mask   = np.isin(y_all, list(known_labels))
test_mask    = test_mask & known_mask

y_train = y_all[train_mask]
y_test  = y_all[test_mask]

print(f"  Test events after filtering unseen labels: {test_mask.sum()}")

# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def make_classifiers():
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                       random_state=RANDOM_STATE,
                                       solver="lbfgs")),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "HistGradBoost": HistGradientBoostingClassifier(
            max_iter=200, max_depth=6,
            random_state=RANDOM_STATE, learning_rate=0.1,
        ),
    }

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
all_results = []

print("\n" + "=" * 70)
print(f"TEMPORAL CLASSIFICATION RESULTS")
print(f"(train: all leagues except '{args.test_league}')")
print("=" * 70)

for subset_name, col_idx in SUBSETS.items():
    X_train = X_all[train_mask][:, col_idx]
    X_test  = X_all[test_mask][:, col_idx]

    for clf_name, clf in make_classifiers().items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        tag = subset_name.replace("\n", " ")
        print(f"  {tag:<30s} + {clf_name:<20s}  Acc={acc*100:.1f}%")
        all_results.append({
            "Features": subset_name,
            "Classifier": clf_name,
            "Accuracy": acc,
            "n_features": len(col_idx),
        })

results_df = pd.DataFrame(all_results).sort_values("Accuracy", ascending=False)
results_df.to_csv(out_dir / "temporal_classification_results.csv", index=False)

print("\n" + "=" * 70)
print("TOP RESULTS")
print("=" * 70)
print(results_df.head(6).to_string(index=False, float_format="%.4f"))

# ---------------------------------------------------------------------------
# Per-class breakdown for best model
# ---------------------------------------------------------------------------
best = results_df.iloc[0]
col_idx = SUBSETS[best["Features"]]
X_train = X_all[train_mask][:, col_idx]
X_test  = X_all[test_mask][:, col_idx]

best_clf = make_classifiers()[best["Classifier"]]
best_clf.fit(X_train, y_train)
y_pred_best = best_clf.predict(X_test)

present = sorted(set(y_test))
report = classification_report(
    y_test, y_pred_best,
    labels=present,
    target_names=label_classes[present],
    zero_division=0,
)
tag = best["Features"].replace("\n", " ")
print(f"\nPer-class report (best: {tag} + {best['Classifier']}):")
print(report)

report_path = out_dir / "temporal_classification_report.txt"
with open(report_path, "w") as f:
    f.write(f"Best model: {tag} + {best['Classifier']}\n")
    f.write(f"Test league: {args.test_league}\n\n")
    f.write(report)

# ---------------------------------------------------------------------------
# Feature importance (RandomForest on full features)
# ---------------------------------------------------------------------------
rf_full_idx = SUBSETS["Full temporal\n(all features)"]
X_train_full = X_all[train_mask][:, rf_full_idx]
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                            random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_full, y_train)
importances = rf.feature_importances_

fig, ax = plt.subplots(figsize=(10, 5))
sorted_idx = np.argsort(importances)[::-1]
ax.bar(range(len(rf_full_idx)),
       importances[sorted_idx], color="#1f77b4", alpha=0.8)
ax.set_xticks(range(len(rf_full_idx)))
ax.set_xticklabels([feature_names[i] for i in sorted_idx],
                   rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Importance", fontsize=11)
ax.set_title("Feature Importance — Random Forest (Full Temporal Features)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_dir / 'feature_importance.png'}")

# ---------------------------------------------------------------------------
# Comparison bar chart: accuracy by feature subset
# ---------------------------------------------------------------------------
best_per_subset = results_df.groupby("Features")["Accuracy"].max().reset_index()
# Preserve ordering
subset_order = list(SUBSETS.keys())
best_per_subset["order"] = best_per_subset["Features"].map(
    {k: i for i, k in enumerate(subset_order)}
)
best_per_subset = best_per_subset.sort_values("order")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(best_per_subset)), best_per_subset["Accuracy"],
              color=["#aec7e8", "#1f77b4", "#aec7e8", "#ff7f0e"], alpha=0.9)
ax.set_xticks(range(len(best_per_subset)))
ax.set_xticklabels(best_per_subset["Features"], fontsize=10)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title(f"Annotation-Only Classification: Impact of Temporal Features\n"
             f"(tested on {args.test_league})",
             fontsize=12, fontweight="bold")

# Reference lines
ax.axhline(0.1375, color="red",    linestyle="--", linewidth=1.5,
           label="Yunting baseline 13.75%")
ax.axhline(0.3054, color="orange", linestyle="--", linewidth=1.5,
           label="Yunting temporal 30.54%")

for bar, acc in zip(bars, best_per_subset["Accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=11,
            fontweight="bold")

ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(best_per_subset["Accuracy"].max() * 1.2, 0.45))
plt.tight_layout()
fig.savefig(out_dir / "temporal_feature_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_dir / 'temporal_feature_comparison.png'}")
print(f"\nDone. All results in {out_dir}/")
