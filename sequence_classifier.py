"""
Sequence models over per-frame DINOv2 features for event classification.

Instead of pooling the 60-frame window into a single vector, these models
consume the full (60, 768) sequence, using temporal ordering:

    1. MeanPool-MLP  : masked mean over frames -> MLP        (baseline)
    2. BiLSTM        : bidirectional LSTM -> masked mean of hidden states
    3. TemporalCNN   : stacked 1D convolutions over time -> masked mean

Protocol (leak-free, by game):
    train = games 01-20,  val = games 21-24 (early stopping),  test = 25-30

Usage:
    python sequence_classifier.py \
        --features features/temporal_dinov2-base_30games/clip_temporal_padded.npz \
        --out-dir clustering/classification
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--features",
                    default="features/temporal_dinov2-base_30games/clip_temporal_padded.npz")
parser.add_argument("--out-dir", default="clustering/classification")
parser.add_argument("--val-games", nargs="+", default=["21", "22", "23", "24"])
parser.add_argument("--test-games", nargs="+", default=["25", "26", "27", "28", "29", "30"])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--hidden-dim", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default=None)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
npz = np.load(args.features, allow_pickle=True)
feats = npz["features"].astype(np.float32)   # (N, T, D)
masks = npz["masks"].astype(np.float32)      # (N, T)
labels = npz["labels"].astype(str)
games = npz["games"].astype(str)

classes = sorted(set(labels))
cls_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([cls_to_idx[l] for l in labels])

val_set, test_set = set(args.val_games), set(args.test_games)
tr = np.array([g not in val_set and g not in test_set for g in games])
va = np.array([g in val_set for g in games])
te = np.array([g in test_set for g in games])
print(f"Events: train={tr.sum()}  val={va.sum()}  test={te.sum()}  "
      f"classes={len(classes)}  dim={feats.shape[2]}")

# Standardize per-dim using train frames only (valid frames)
train_frames = feats[tr][masks[tr].astype(bool)]
mu, sd = train_frames.mean(0), train_frames.std(0) + 1e-6
feats = (feats - mu) / sd
feats *= masks[:, :, None]  # keep padding at zero

X = torch.tensor(feats)
M = torch.tensor(masks)
Y = torch.tensor(y, dtype=torch.long)


def loader(idx_mask, shuffle):
    ds = torch.utils.data.TensorDataset(X[idx_mask], M[idx_mask], Y[idx_mask])
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def masked_mean(h, m):
    """h: (B, T, D), m: (B, T) -> (B, D)"""
    return (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


class MeanPoolMLP(nn.Module):
    def __init__(self, d_in, d_h, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d_h, n_cls))

    def forward(self, x, m):
        return self.net(masked_mean(x, m))


class BiLSTM(nn.Module):
    def __init__(self, d_in, d_h, n_cls):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.lstm = nn.LSTM(d_h, d_h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d_h, n_cls))

    def forward(self, x, m):
        h, _ = self.lstm(torch.relu(self.proj(x)))
        return self.head(masked_mean(h, m))


class TemporalCNN(nn.Module):
    def __init__(self, d_in, d_h, n_cls):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, d_h, 5, padding=2), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(d_h, d_h, 5, padding=2), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(d_h, n_cls)

    def forward(self, x, m):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d_h)
        return self.head(masked_mean(h, m))


class TransformerClassifier(nn.Module):
    def __init__(self, d_in, d_h, n_cls, n_heads=4, n_layers=2, max_len=60):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_h) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_h, nhead=n_heads, dim_feedforward=d_h * 4,
            dropout=0.3, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_h, n_cls))

    def forward(self, x, m):
        h = self.proj(x) + self.pos[:, :x.size(1)]
        pad_mask = ~m.bool()  # True = ignore
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(masked_mean(h, m))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def run_model(name, model):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    tr_loader = loader(tr, shuffle=True)
    best_val, best_state, wait = 0.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, mb, yb in tr_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb, mb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for xb, mb, _ in loader(va, shuffle=False):
                preds.append(model(xb.to(device), mb.to(device)).argmax(1).cpu())
            val_acc = accuracy_score(y[va], torch.cat(preds).numpy())

        if val_acc > best_val:
            best_val, wait = val_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        for xb, mb, _ in loader(te, shuffle=False):
            preds.append(model(xb.to(device), mb.to(device)).argmax(1).cpu())
        test_pred = torch.cat(preds).numpy()
    test_acc = accuracy_score(y[te], test_pred)
    print(f"{name:14s} stopped@{epoch:3d}  val={best_val:.1%}  test={test_acc:.1%}")
    return best_val, test_acc, test_pred


d_in, n_cls = feats.shape[2], len(classes)
results = {}
for name, model in [
    ("MeanPool-MLP",  MeanPoolMLP(d_in, args.hidden_dim, n_cls)),
    ("BiLSTM",        BiLSTM(d_in, args.hidden_dim, n_cls)),
    ("TemporalCNN",   TemporalCNN(d_in, args.hidden_dim, n_cls)),
    ("Transformer",   TransformerClassifier(d_in, args.hidden_dim, n_cls)),
]:
    results[name] = run_model(name, model)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
rows = [{"model": k, "val_acc": v[0], "test_acc": v[1]} for k, v in results.items()]
df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
df.to_csv(out_dir / "sequence_results.csv", index=False)
print("\n" + df.to_string(index=False))

best_name = df.iloc[0]["model"]
best_pred = results[best_name][2]
report = classification_report(
    y[te], best_pred, labels=range(n_cls), target_names=classes, zero_division=0)
(out_dir / "sequence_classification_report.txt").write_text(
    f"Best model: {best_name}\n\n{report}")
print(f"\nPer-class report (best: {best_name}):\n{report}")
print(f"Saved -> {out_dir / 'sequence_results.csv'}")
