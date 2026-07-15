"""
Event Classification Pipeline
==============================
Evaluates event classification accuracy using DINOv2 temporal features
with multiple pooling strategies and classifiers.

Compares:
  - Pooling strategies: mean, max, first frame, last frame, attention
  - Classifiers: Linear SVM, Logistic Regression, MLP
  - Embeddings: raw DINOv2 vs contrastive-projected

Train/test split: leave-2-games-out (trains on 8 games, tests on 2).
This tests generalization to unseen games — the realistic setting.

Usage:
    python event_classification.py
    python event_classification.py --test-games 09 10
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
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dino-features", default="features/temporal_dinov2-base")
parser.add_argument("--contrastive-features", default="features/contrastive")
parser.add_argument("--out-dir", default="clustering/classification")
parser.add_argument("--test-games", nargs="+", default=["09", "10"],
                    help="Games held out for testing (default: 09 10)")
args = parser.parse_args()

out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load features
# ---------------------------------------------------------------------------
print("Loading features ...")
npz = np.load(pathlib.Path(args.dino_features) / "clip_temporal_padded.npz",
              allow_pickle=True)
feat_padded = npz["features"].astype(np.float32)   # (627, 60, 768)
masks = npz["masks"]                                 # (627, 60)
labels_raw = npz["labels"].astype(str)
games = npz["games"].astype(str)

# Contrastive embeddings
cont_path = pathlib.Path(args.contrastive_features) / "contrastive_embeddings.npz"
has_contrastive = cont_path.exists()
if has_contrastive:
    feat_cont = np.load(cont_path)["features"].astype(np.float32)  # (627, 256)
    print(f"  Contrastive embeddings: {feat_cont.shape}")

print(f"  Events: {len(labels_raw)},  Games: {sorted(set(games))}")
print(f"  Test games: {args.test_games}")

# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------
def mean_pool(padded, masks):
    n_valid = masks.sum(axis=1, keepdims=True).clip(min=1)
    feat = (padded * masks[:, :, np.newaxis]).sum(axis=1) / n_valid
    return feat / np.linalg.norm(feat, axis=1, keepdims=True)

def max_pool(padded, masks):
    # Set padding to -inf before taking max
    padded_masked = padded.copy()
    padded_masked[~masks] = -np.inf
    feat = padded_masked.max(axis=1)
    # Handle events where all frames are masked (shouldn't happen)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
    return feat / norms

def first_frame(padded, masks):
    feat = padded[:, 0, :]
    norms = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
    return feat / norms

def last_frame(padded, masks):
    # Get index of last valid frame per event
    last_idx = (masks.sum(axis=1) - 1).clip(min=0).astype(int)
    feat = padded[np.arange(len(padded)), last_idx, :]
    norms = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
    return feat / norms

def attention_pool(padded, masks):
    """Simple self-attention: weight frames by their L2 norm (salience proxy)."""
    norms = np.linalg.norm(padded, axis=2)          # (E, T)
    norms[~masks] = -np.inf
    weights = np.exp(norms - norms.max(axis=1, keepdims=True))
    weights[~masks] = 0.0
    weights = weights / weights.sum(axis=1, keepdims=True).clip(min=1e-8)
    feat = (padded * weights[:, :, np.newaxis]).sum(axis=1)
    norms_out = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
    return feat / norms_out

# ---------------------------------------------------------------------------
# Train/test split (leave-N-games-out)
# ---------------------------------------------------------------------------
test_mask = np.isin(games, args.test_games)
train_mask = ~test_mask

print(f"\nTrain: {train_mask.sum()} events from games "
      f"{sorted(set(games[train_mask]))}")
print(f"Test:  {test_mask.sum()} events from games "
      f"{sorted(set(games[test_mask]))}")

# Label encoder — fit on train labels only, handle unseen test labels
le = LabelEncoder()
le.fit(labels_raw[train_mask])

# Filter test events whose labels don't appear in train
known_mask = np.isin(labels_raw, le.classes_)
test_mask = test_mask & known_mask

y_train = le.transform(labels_raw[train_mask])
y_test = le.transform(labels_raw[test_mask])

print(f"  Train labels: {len(set(y_train))},  Test labels: {len(set(y_test))}")
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
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128),
                                  max_iter=500, random_state=RANDOM_STATE,
                                  early_stopping=True, validation_fraction=0.1)),
        ]),
    }

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
pooling_fns = {
    "mean_pool":      mean_pool,
    "max_pool":       max_pool,
    "first_frame":    first_frame,
    "last_frame":     last_frame,
    "attention_pool": attention_pool,
}

all_results = []

print("\n" + "=" * 70)
print("CLASSIFICATION RESULTS (test games:", args.test_games, ")")
print("=" * 70)

# DINOv2 pooling experiments
for pool_name, pool_fn in pooling_fns.items():
    feat_all = pool_fn(feat_padded, masks)
    X_train = feat_all[train_mask]
    X_test = feat_all[test_mask]

    for clf_name, clf in make_classifiers().items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  DINOv2 {pool_name:<18s} + {clf_name:<10s}  Acc={acc:.4f}  ({acc*100:.1f}%)")
        all_results.append({
            "Embedding": "DINOv2",
            "Pooling": pool_name,
            "Classifier": clf_name,
            "Accuracy": acc,
        })

# Contrastive embeddings
if has_contrastive:
    print()
    X_train_cont = feat_cont[train_mask]
    X_test_cont = feat_cont[test_mask]

    for clf_name, clf in make_classifiers().items():
        clf.fit(X_train_cont, y_train)
        y_pred = clf.predict(X_test_cont)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Contrastive (256-dim)      + {clf_name:<10s}  Acc={acc:.4f}  ({acc*100:.1f}%)")
        all_results.append({
            "Embedding": "Contrastive",
            "Pooling": "triplet_projected",
            "Classifier": clf_name,
            "Accuracy": acc,
        })

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(all_results).sort_values("Accuracy", ascending=False)
results_df.to_csv(out_dir / "classification_results.csv", index=False)

print("\n" + "=" * 70)
print("TOP 10 RESULTS")
print("=" * 70)
print(results_df.head(10).to_string(index=False, float_format="%.4f"))

# Professor's baseline for comparison
print(f"\n  Yunting's baseline (pretrained embeddings + ML):  13.75%")
print(f"  Yunting's temporal (30s window):                  30.54%")
print(f"  Our best:  {results_df['Accuracy'].max()*100:.2f}%  "
      f"({results_df.iloc[0]['Embedding']} {results_df.iloc[0]['Pooling']} + "
      f"{results_df.iloc[0]['Classifier']})")

# ---------------------------------------------------------------------------
# Plot: accuracy comparison bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

# Group by embedding+pooling, show best classifier per group
best_per_config = results_df.groupby(["Embedding", "Pooling"])["Accuracy"].max().reset_index()
labels_plot = [f"{r['Embedding']}\n{r['Pooling']}" for _, r in best_per_config.iterrows()]
colors = ["#1f77b4" if r["Embedding"] == "DINOv2" else "#2ca02c"
          for _, r in best_per_config.iterrows()]

bars = ax.bar(range(len(best_per_config)), best_per_config["Accuracy"],
              color=colors, alpha=0.8)
ax.set_xticks(range(len(best_per_config)))
ax.set_xticklabels(labels_plot, fontsize=9)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Event Classification Accuracy by Pooling Strategy\n"
             f"(trained on games {sorted(set(games[train_mask]))}, "
             f"tested on games {args.test_games})",
             fontsize=12, fontweight="bold")

# Reference lines
ax.axhline(0.1375, color="red", linestyle="--", linewidth=1.5,
           label="Yunting baseline (13.75%)")
ax.axhline(0.3054, color="orange", linestyle="--", linewidth=1.5,
           label="Yunting temporal (30.54%)")

for bar, acc in zip(bars, best_per_config["Accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=9)

ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(best_per_config["Accuracy"].max() * 1.15, 0.40))

# Legend for colors
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#1f77b4", alpha=0.8, label="DINOv2"),
    Patch(color="#2ca02c", alpha=0.8, label="Contrastive"),
    plt.Line2D([0], [0], color="red", linestyle="--", label="Yunting baseline 13.75%"),
    plt.Line2D([0], [0], color="orange", linestyle="--", label="Yunting temporal 30.54%"),
], fontsize=9)

plt.tight_layout()
fig.savefig(out_dir / "classification_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_dir / 'classification_accuracy.png'}")

# ---------------------------------------------------------------------------
# Confusion matrix for best model
# ---------------------------------------------------------------------------
best_row = results_df.iloc[0]
pool_fn = pooling_fns.get(best_row["Pooling"], None)

if pool_fn:
    feat_all = pool_fn(feat_padded, masks)
else:
    feat_all = feat_cont  # contrastive

X_train = feat_all[train_mask]
X_test = feat_all[test_mask]

best_clf_name = best_row["Classifier"]
best_clf = list(make_classifiers().items())
best_clf = {k: v for k, v in make_classifiers().items()}[best_clf_name]
best_clf.fit(X_train, y_train)
y_pred_best = best_clf.predict(X_test)

# Per-class report
report = classification_report(y_test, y_pred_best,
                                target_names=le.classes_[sorted(set(y_test))],
                                zero_division=0)
print(f"\nPer-class report (best model: {best_row['Embedding']} "
      f"{best_row['Pooling']} + {best_clf_name}):")
print(report)

with open(out_dir / "classification_report.txt", "w") as f:
    f.write(f"Best model: {best_row['Embedding']} {best_row['Pooling']} + {best_clf_name}\n")
    f.write(f"Test games: {args.test_games}\n\n")
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(14, 12))
present_labels = sorted(set(y_test))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_[present_labels]
)
disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_row['Embedding']} {best_row['Pooling']} + {best_clf_name}\n"
             f"Test games {args.test_games}  |  Accuracy={best_row['Accuracy']*100:.1f}%",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out_dir / 'confusion_matrix.png'}")
print(f"\nDone. All results in {out_dir}/")
