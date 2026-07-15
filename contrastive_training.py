"""
Contrastive Learning with Triplet Loss for Soccer Event Embeddings
===================================================================
Trains a projection head on top of frozen DINOv2 event embeddings
to disentangle event semantics from game-level visual bias.

Triplet construction:
    Anchor:   Event_type_A, Game_X
    Positive: Event_type_A, Game_Y   (same event, different game)
    Negative: Event_type_B, Game_X   (different event, same game)

Goal: distance(anchor, positive) < distance(anchor, negative)

After training, the learned projection maps DINOv2 embeddings into a
space where same-type events cluster together regardless of which game
they came from.

Usage:
    python contrastive_training.py
    python contrastive_training.py --epochs 100 --lr 1e-3 --margin 0.5
    python contrastive_training.py --features-dir features/temporal_dinov2-base
"""

import argparse
import pathlib
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--features-dir", default="features/temporal_dinov2-base")
parser.add_argument("--out-dir", default="features/contrastive")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--margin", type=float, default=0.5)
parser.add_argument("--hidden-dim", type=int, default=512)
parser.add_argument("--output-dim", type=int, default=256)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train-games", nargs="+", default=None,
                    help="Game IDs to mine triplets from (e.g. 01 02 ... 24). "
                         "All events are still projected/saved. "
                         "Default: all games (leaky for downstream eval).")
parser.add_argument("--device", default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load DINOv2 event-level features (mean-pooled)
# ---------------------------------------------------------------------------
print("Loading features ...")
features_dir = pathlib.Path(args.features_dir)
npz = np.load(features_dir / "clip_temporal_padded.npz", allow_pickle=True)
feat_padded = npz["features"].astype(np.float32)   # (627, 60, 768)
masks = npz["masks"]                                 # (627, 60)
labels = npz["labels"].astype(str)                   # (627,)
games = npz["games"].astype(str)                     # (627,)

# Mean-pool per event -> (627, 768)
n_valid = masks.sum(axis=1, keepdims=True).clip(min=1)
feat_mean = (feat_padded * masks[:, :, np.newaxis]).sum(axis=1) / n_valid
feat_mean = feat_mean / np.linalg.norm(feat_mean, axis=1, keepdims=True)

input_dim = feat_mean.shape[1]
n_events = len(labels)
print(f"  Events: {n_events},  Embedding dim: {input_dim}")
print(f"  Unique labels: {len(set(labels))},  Unique games: {len(set(games))}")

# ---------------------------------------------------------------------------
# Build lookup indices for triplet mining
# ---------------------------------------------------------------------------
# label -> list of event indices
label_to_indices = defaultdict(list)
# game -> list of event indices
game_to_indices = defaultdict(list)
# (label, game) -> list of event indices
label_game_to_indices = defaultdict(list)

if args.train_games is not None:
    train_games = set(args.train_games)
    train_indices = [i for i in range(n_events) if games[i] in train_games]
    print(f"  Triplet mining restricted to {len(train_games)} train games "
          f"({len(train_indices)} / {n_events} events)")
else:
    train_indices = list(range(n_events))

for i in train_indices:
    lab, gam = labels[i], games[i]
    label_to_indices[lab].append(i)
    game_to_indices[gam].append(i)
    label_game_to_indices[(lab, gam)].append(i)

# Find labels that appear in 2+ games (required for positive pairs)
eligible_labels = set()
for lab in label_to_indices:
    games_with_label = set(games[i] for i in label_to_indices[lab])
    if len(games_with_label) >= 2:
        eligible_labels.add(lab)

eligible_indices = [i for i in train_indices if labels[i] in eligible_labels]
print(f"  Eligible labels (in 2+ games): {len(eligible_labels)}")
print(f"  Eligible events: {len(eligible_indices)} / {n_events}")


# ---------------------------------------------------------------------------
# Triplet Dataset with online mining
# ---------------------------------------------------------------------------
class TripletDataset(Dataset):
    """
    For each eligible anchor, dynamically samples a positive and negative.

    Anchor:   event i with label L in game G
    Positive: random event with label L in game != G
    Negative: random event with label != L in game G
    """

    def __init__(self, features, labels, games, eligible_indices,
                 label_to_indices, game_to_indices, label_game_to_indices):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = labels
        self.games = games
        self.eligible = eligible_indices
        self.label_to_idx = label_to_indices
        self.game_to_idx = game_to_indices
        self.lg_to_idx = label_game_to_indices

    def __len__(self):
        return len(self.eligible)

    def __getitem__(self, idx):
        anchor_idx = self.eligible[idx]
        anchor_label = self.labels[anchor_idx]
        anchor_game = self.games[anchor_idx]

        # Positive: same label, different game
        pos_candidates = [i for i in self.label_to_idx[anchor_label]
                          if self.games[i] != anchor_game]
        pos_idx = random.choice(pos_candidates)

        # Negative: different label, same game
        neg_candidates = [i for i in self.game_to_idx[anchor_game]
                          if self.labels[i] != anchor_label]
        if not neg_candidates:
            # Fallback: different label, any train game
            neg_candidates = [i for lst in self.label_to_idx.values()
                              for i in lst if self.labels[i] != anchor_label]
        neg_idx = random.choice(neg_candidates)

        return (self.features[anchor_idx],
                self.features[pos_idx],
                self.features[neg_idx])


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """MLP projection: input_dim -> hidden -> output_dim, L2-normalized."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
dataset = TripletDataset(feat_mean, labels, games, eligible_indices,
                         label_to_indices, game_to_indices, label_game_to_indices)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

model = ProjectionHead(input_dim, args.hidden_dim, args.output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)

print(f"\nProjection head: {input_dim} -> {args.hidden_dim} -> {args.output_dim}")
print(f"Device: {device}")
print(f"Margin: {args.margin},  LR: {args.lr},  Epochs: {args.epochs}")
print(f"Batches per epoch: {len(loader)}\n")

history = {"epoch": [], "loss": [], "pos_dist": [], "neg_dist": [], "triplet_acc": []}

for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_loss = 0.0
    epoch_pos_dist = 0.0
    epoch_neg_dist = 0.0
    epoch_correct = 0
    epoch_total = 0

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        a_emb = model(anchor)
        p_emb = model(positive)
        n_emb = model(negative)

        loss = triplet_loss_fn(a_emb, p_emb, n_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        with torch.no_grad():
            d_pos = F.pairwise_distance(a_emb, p_emb)
            d_neg = F.pairwise_distance(a_emb, n_emb)
            correct = (d_pos < d_neg).sum().item()

        batch_size = anchor.size(0)
        epoch_loss += loss.item() * batch_size
        epoch_pos_dist += d_pos.sum().item()
        epoch_neg_dist += d_neg.sum().item()
        epoch_correct += correct
        epoch_total += batch_size

    scheduler.step()

    avg_loss = epoch_loss / epoch_total
    avg_pos = epoch_pos_dist / epoch_total
    avg_neg = epoch_neg_dist / epoch_total
    acc = epoch_correct / epoch_total

    history["epoch"].append(epoch)
    history["loss"].append(avg_loss)
    history["pos_dist"].append(avg_pos)
    history["neg_dist"].append(avg_neg)
    history["triplet_acc"].append(acc)

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"Loss={avg_loss:.4f}  "
              f"d(a,p)={avg_pos:.4f}  d(a,n)={avg_neg:.4f}  "
              f"TripletAcc={acc:.3f}")

# ---------------------------------------------------------------------------
# Save model + projected embeddings
# ---------------------------------------------------------------------------
print("\nTraining complete. Saving outputs ...")

# Save model weights
model_path = out_dir / "projection_head.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": input_dim,
    "hidden_dim": args.hidden_dim,
    "output_dim": args.output_dim,
    "margin": args.margin,
    "epochs": args.epochs,
}, model_path)
print(f"  Model -> {model_path}")

# Project all event embeddings through the trained head
model.eval()
with torch.no_grad():
    all_feats = torch.tensor(feat_mean, dtype=torch.float32).to(device)
    projected = model(all_feats).cpu().numpy()  # (627, output_dim)

print(f"  Projected embeddings shape: {projected.shape}")

# Save projected embeddings
np.savez_compressed(
    out_dir / "contrastive_embeddings.npz",
    features=projected,
    labels=labels,
    games=games,
)
print(f"  Embeddings -> {out_dir / 'contrastive_embeddings.npz'}")

# Save training history
hist_df = pd.DataFrame(history) if 'pd' in dir() else None
import pandas as pd
hist_df = pd.DataFrame(history)
hist_df.to_csv(out_dir / "training_history.csv", index=False)

# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Contrastive Learning Training Curves (Triplet Loss)",
             fontsize=13, fontweight="bold")

# Loss
axes[0].plot(history["epoch"], history["loss"], color="tab:red", linewidth=1.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Triplet Loss")
axes[0].set_title("Loss")
axes[0].grid(alpha=0.3)

# Distances
axes[1].plot(history["epoch"], history["pos_dist"], label="d(anchor, positive)",
             color="tab:green", linewidth=1.5)
axes[1].plot(history["epoch"], history["neg_dist"], label="d(anchor, negative)",
             color="tab:red", linewidth=1.5)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("L2 Distance")
axes[1].set_title("Positive vs Negative Distance")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# Triplet accuracy
axes[2].plot(history["epoch"], history["triplet_acc"], color="tab:blue", linewidth=1.5)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
axes[2].set_title("Triplet Accuracy (d_pos < d_neg)")
axes[2].set_ylim(0, 1.05)
axes[2].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Training curves -> {out_dir / 'training_curves.png'}")

print(f"\nFinal metrics:")
print(f"  Loss:        {history['loss'][-1]:.4f}")
print(f"  d(a,p):      {history['pos_dist'][-1]:.4f}")
print(f"  d(a,n):      {history['neg_dist'][-1]:.4f}")
print(f"  Triplet Acc: {history['triplet_acc'][-1]:.3f}")
print("\nDone.")
