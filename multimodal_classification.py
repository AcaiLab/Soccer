"""
Multimodal Event Classifier
============================
Combines DINOv2 visual features (temporal pooling) with annotation-only
game flow features (temporal context) for event classification.

Compares three feature sets:
  (1) Visual only    — DINOv2 attention-pooled (768-dim)
  (2) Annotation only — game flow temporal context (14-dim)
  (3) Multimodal     — concatenation of (1) + (2) (782-dim)

Train/test split: leave-2-games-out (train: games 01-08, test: 09-10).

Usage:
    python multimodal_classification.py
    python multimodal_classification.py --test-games 09 10
"""

import argparse
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dino-features",
                    default="features/temporal_dinov2-base")
parser.add_argument("--game-flow-features",
                    default="features/game_flow/game_flow_features.csv")
parser.add_argument("--out-dir", default="clustering/classification")
parser.add_argument("--test-games", nargs="+", default=["09", "10"])
args = parser.parse_args()

out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Game number -> game_id mapping (order matches process_10_games.py)
# ---------------------------------------------------------------------------
GAME_MAP = {
    "01": "2021-08-22_arsenal-fc-chelsea-fc-premier-league-2021-2022",
    "02": "2021-08-28_manchester-city-arsenal-fc-premier-league-2021-2022",
    "03": "2021-09-11_manchester-united-newcastle-united-premier-league-2021-2022",
    "04": "2021-09-19_tottenham-hotspur-chelsea-fc-premier-league-2021-2022",
    "05": "2021-09-25_manchester-united-aston-villa-premier-league-2021-2022",
    "06": "2021-10-03_liverpool-fc-manchester-city-premier-league-2021-2022",
    "07": "2021-10-16_leicester-city-manchester-united-premier-league-2021-2022",
    "08": "2021-10-24_manchester-united-liverpool-fc-premier-league-2021-2022",
    "09": "2021-10-24_west-ham-united-tottenham-hotspur-premier-league-2021-2022",
    "10": "2021-11-06_manchester-united-manchester-city-premier-league-2021-2022",
    "11": "2021-08-13_brentford-fc-arsenal-fc",
    "12": "2021-08-14_manchester-united-leeds-united-premier-league-2021-2022",
    "13": "2021-08-15_newcastle-united-west-ham-united-premier-league-2021-2022",
    "14": "2021-09-18_manchester-city-southampton-fc-premier-league-2021-2022",
    "15": "2021-09-19_west-ham-united-manchester-united-premier-league-2021-2022",
    "16": "2021-09-25_leeds-united-west-ham-united-premier-league-2021-2022",
    "17": "2021-10-23_everton-fc-watford-fc-premier-league",
    "18": "2021-10-23_leeds-united-wolverhampton-wanderers",
    "19": "2021-10-30_tottenham-hotspur-manchester-united-premier-league-2021-2022",
    "20": "2021-10-31_aston-villa-west-ham-united-premier-league",
    "21": "2021-11-07_west-ham-united-liverpool-fc-premier-league-2021-2022",
    "22": "2021-11-20_liverpool-fc-arsenal-fc-premier-league-2021-2022",
    "23": "2021-11-20_newcastle-united-brentford-fc",
    "24": "2021-12-01_everton-fc-liverpool-fc-premier-league-2021-2022",
    "25": "2021-12-04_west-ham-united-chelsea-fc-premier-league-2021-2022",
    "26": "2021-12-14_manchester-city-leeds-united-premier-league-2021-2022",
    "27": "2021-12-19_tottenham-hotspur-liverpool-fc-premier-league-2021-2022",
    "28": "2022-01-02_chelsea-fc-liverpool-fc-premier-league-2021-2022",
    "29": "2022-01-15_manchester-city-chelsea-fc-premier-league-2021-2022",
    "30": "2022-01-22_brentford-fc-wolverhampton-wanderers",
}

# ---------------------------------------------------------------------------
# Load DINOv2 temporal features
# ---------------------------------------------------------------------------
print("Loading DINOv2 temporal features ...")
npz = np.load(pathlib.Path(args.dino_features) / "clip_temporal_padded.npz",
              allow_pickle=True)
feat_padded = npz["features"].astype(np.float32)   # (627, 60, 768)
masks       = npz["masks"]                          # (627, 60)  bool
labels_raw  = npz["labels"].astype(str)             # (627,)
games_raw   = npz["games"].astype(str)              # (627,) e.g. "01"
halves_raw  = npz["halves"].astype(str)             # (627,)
event_tags  = npz["event_tags"].astype(str)         # (627,) "MM-SS.sss"

print(f"  DINOv2 events: {len(labels_raw)}")


def tag_to_minute(tag: str, half: str) -> float:
    parts = tag.split("-")
    mins  = int(parts[0])
    secs  = float(parts[1]) if len(parts) > 1 else 0.0
    minute = mins + secs / 60.0
    if int(half) == 2:
        minute += 45.0
    return round(minute, 2)


# Build a key for each DINOv2 event: (game_id, half, minute)
dino_minutes = np.array([tag_to_minute(et, h)
                         for et, h in zip(event_tags, halves_raw)])
dino_game_ids = np.array([GAME_MAP[g] for g in games_raw])

# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------
def attention_pool(padded, masks):
    norms = np.linalg.norm(padded, axis=2)          # (E, T)
    norms[~masks] = -np.inf
    weights = np.exp(norms - norms.max(axis=1, keepdims=True))
    weights[~masks] = 0.0
    weights = weights / weights.sum(axis=1, keepdims=True).clip(min=1e-8)
    feat = (padded * weights[:, :, np.newaxis]).sum(axis=1)
    norms_out = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
    return feat / norms_out

def mean_pool(padded, masks):
    n_valid = masks.sum(axis=1, keepdims=True).clip(min=1)
    feat = (padded * masks[:, :, np.newaxis]).sum(axis=1) / n_valid
    norms = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
    return feat / norms

# Use attention pooling for visual features (best single pooling in event_classification.py)
visual_feat = attention_pool(feat_padded, masks.copy())   # (627, 768)

# ---------------------------------------------------------------------------
# Load game flow features for these 10 games
# ---------------------------------------------------------------------------
print("Loading game flow features ...")
df_gf = pd.read_csv(args.game_flow_features)
df_gf_10 = df_gf[df_gf["game_id"].isin(GAME_MAP.values())].copy()
print(f"  Game flow events for 10 games: {len(df_gf_10)}")

# Numeric game-flow feature columns (same as used in temporal_event_classifier.py)
GF_FEATURE_COLS = [
    "half", "minute",
    "prev_1_enc", "prev_2_enc", "prev_3_enc",
    "prev_1_cat_enc", "prev_2_cat_enc", "prev_3_cat_enc",
    "recent_corners", "recent_shots", "recent_fouls",
    "recent_cards", "recent_attacks", "recent_stoppages",
]

# Encode categorical columns (fit on ALL 10 games consistently)
from sklearn.preprocessing import LabelEncoder as _LE
for col in ["label", "category", "prev_1", "prev_2", "prev_3",
            "prev_1_cat", "prev_2_cat", "prev_3_cat"]:
    le = _LE()
    df_gf_10[f"{col}_enc"] = le.fit_transform(df_gf_10[col].astype(str))

# ---------------------------------------------------------------------------
# Align DINOv2 events with game flow events (match on game_id, half, minute)
# ---------------------------------------------------------------------------
print("Aligning DINOv2 and game flow events ...")

# Build lookup: (game_id, half, minute) -> row index in df_gf_10
df_gf_10 = df_gf_10.reset_index(drop=True)
gf_key_to_idx = {}
for idx, row in df_gf_10.iterrows():
    key = (row["game_id"], int(row["half"]), row["minute"])
    gf_key_to_idx[key] = idx

matched_dino_idx = []
matched_gf_idx   = []

for i in range(len(labels_raw)):
    key = (dino_game_ids[i], int(halves_raw[i]), dino_minutes[i])
    if key in gf_key_to_idx:
        matched_dino_idx.append(i)
        matched_gf_idx.append(gf_key_to_idx[key])

matched_dino_idx = np.array(matched_dino_idx)
matched_gf_idx   = np.array(matched_gf_idx)

print(f"  Matched {len(matched_dino_idx)} / {len(labels_raw)} events")

# Extract aligned feature arrays
X_visual   = visual_feat[matched_dino_idx]                             # (N, 768)
X_gf       = df_gf_10.loc[matched_gf_idx, GF_FEATURE_COLS].values.astype(np.float32)  # (N, 14)
X_multimod = np.concatenate([X_visual, X_gf], axis=1)                 # (N, 782)
y_labels   = labels_raw[matched_dino_idx]
games_matched = games_raw[matched_dino_idx]

print(f"  Visual dim: {X_visual.shape[1]}")
print(f"  Annotation dim: {X_gf.shape[1]}")
print(f"  Multimodal dim: {X_multimod.shape[1]}")

# Sanity check — labels should match between sources
gf_labels_matched = df_gf_10.loc[matched_gf_idx, "label"].values
mismatches = np.sum(y_labels != gf_labels_matched)
if mismatches > 0:
    print(f"  WARNING: {mismatches} label mismatches between sources")
else:
    print("  Label alignment: OK")

# ---------------------------------------------------------------------------
# Train / test split — leave-N-games-out
# ---------------------------------------------------------------------------
test_mask  = np.isin(games_matched, args.test_games)
train_mask = ~test_mask

print(f"\nTrain: {train_mask.sum()} events from games "
      f"{sorted(set(games_matched[train_mask]))}")
print(f"Test:  {test_mask.sum()} events from games {args.test_games}")

le_label = LabelEncoder()
le_label.fit(y_labels[train_mask])
known_mask = np.isin(y_labels, le_label.classes_)
test_mask  = test_mask & known_mask

y_train = le_label.transform(y_labels[train_mask])
y_test  = le_label.transform(y_labels[test_mask])
print(f"  Test events after filtering unseen labels: {test_mask.sum()}")

# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def make_classifiers():
    return {
        "LinearSVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=5000, random_state=RANDOM_STATE, C=1.0)),
        ]),
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                       C=1.0, solver="lbfgs")),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                                  random_state=RANDOM_STATE, early_stopping=True,
                                  validation_fraction=0.1)),
        ]),
    }

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "Visual only\n(DINOv2 attn)":    (X_visual,   "768-dim"),
    "Annotation only\n(game flow)":  (X_gf,       "14-dim"),
    "Multimodal\n(visual + annot)":  (X_multimod, "782-dim"),
}

all_results = []

print("\n" + "=" * 70)
print(f"MULTIMODAL CLASSIFICATION RESULTS (test games: {args.test_games})")
print("=" * 70)

for feat_name, (X_all, dim_str) in FEATURE_SETS.items():
    X_tr = X_all[train_mask]
    X_te = X_all[test_mask]
    tag  = feat_name.replace("\n", " ")
    for clf_name, clf in make_classifiers().items():
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        print(f"  {tag:<30s}  {clf_name:<12s}  Acc={acc*100:.1f}%  [{dim_str}]")
        all_results.append({
            "Features":   feat_name,
            "Classifier": clf_name,
            "Accuracy":   acc,
            "Dim":        dim_str,
        })

results_df = pd.DataFrame(all_results).sort_values("Accuracy", ascending=False)
results_df.to_csv(out_dir / "multimodal_results.csv", index=False)

print("\n" + "=" * 70)
print("TOP RESULTS")
print("=" * 70)
print(results_df.head(6).to_string(index=False, float_format="%.4f"))
print(f"\n  Yunting baseline (pretrained + ML):  13.75%")
print(f"  Yunting temporal (30s window):       30.54%")
best_row = results_df.iloc[0]
tag = best_row["Features"].replace("\n", " ")
print(f"  Our best:  {best_row['Accuracy']*100:.2f}%  "
      f"({tag} + {best_row['Classifier']})")

# ---------------------------------------------------------------------------
# Per-class report for best model
# ---------------------------------------------------------------------------
best = results_df.iloc[0]
X_all_best = FEATURE_SETS[best["Features"]][0]
X_tr = X_all_best[train_mask]
X_te = X_all_best[test_mask]
best_clf = make_classifiers()[best["Classifier"]]
best_clf.fit(X_tr, y_train)
y_pred_best = best_clf.predict(X_te)

present = sorted(set(y_test))
report = classification_report(
    y_test, y_pred_best,
    labels=present,
    target_names=le_label.classes_[present],
    zero_division=0,
)
tag = best["Features"].replace("\n", " ")
print(f"\nPer-class report (best: {tag} + {best['Classifier']}):")
print(report)

with open(out_dir / "multimodal_classification_report.txt", "w") as f:
    f.write(f"Best model: {tag} + {best['Classifier']}\n")
    f.write(f"Test games: {args.test_games}\n\n")
    f.write(report)

# ---------------------------------------------------------------------------
# Bar chart: accuracy by feature set
# ---------------------------------------------------------------------------
best_per_set = results_df.groupby("Features")["Accuracy"].max().reset_index()
order = list(FEATURE_SETS.keys())
best_per_set["order"] = best_per_set["Features"].map(
    {k: i for i, k in enumerate(order)}
)
best_per_set = best_per_set.sort_values("order")

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(best_per_set)), best_per_set["Accuracy"],
              color=colors[:len(best_per_set)], alpha=0.85)
ax.set_xticks(range(len(best_per_set)))
ax.set_xticklabels(best_per_set["Features"], fontsize=11)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title(f"Multimodal vs. Unimodal Event Classification\n"
             f"(trained on games 01-08, tested on games {args.test_games})",
             fontsize=12, fontweight="bold")

ax.axhline(0.1375, color="red",    linestyle="--", linewidth=1.5,
           label="Yunting baseline 13.75%")
ax.axhline(0.3054, color="orange", linestyle="--", linewidth=1.5,
           label="Yunting temporal 30.54%")

for bar, acc in zip(bars, best_per_set["Accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{acc*100:.1f}%", ha="center", va="bottom",
            fontsize=12, fontweight="bold")

ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(best_per_set["Accuracy"].max() * 1.2, 0.45))
plt.tight_layout()
fig.savefig(out_dir / "multimodal_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_dir / 'multimodal_comparison.png'}")
print(f"Done. All results in {out_dir}/")
