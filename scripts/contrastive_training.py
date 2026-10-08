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

Triplets are built from event labels, so the head is a supervised model: it
may only see training games. Triplets are mined from the manifest's train
split by default, a game list overlapping the test split is refused, and the
games used are saved with the model and the embeddings so evaluation can keep
them apart (docs/leak_audit.md, B1).

Every event is still projected and saved, so evaluate_contrastive.py can
score held-out games.

Usage:
    python scripts/contrastive_training.py
    python scripts/contrastive_training.py --epochs 100 --lr 1e-3 --margin 0.5
    python scripts/contrastive_training.py --train-games 01 02 03 ... 20
"""

import argparse
import json
import pathlib
import random
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from soccer.config import data_path, dropped_labels, load_config, repo_path  # noqa: E402
from soccer.encoding import pick_device  # noqa: E402
from soccer.features import load_temporal, mean_pool  # noqa: E402
from soccer.manifest import (  # noqa: E402
    assert_disjoint,
    game_nums,
    load_manifest,
    manifest_games,  # noqa: E402
    normalize_game_id,
)


class TripletDataset(Dataset):
    """
    For each eligible anchor, dynamically samples a positive and negative.

    Anchor:   event i with label L in game G
    Positive: random event with label L in game != G
    Negative: random event with label != L in game G (any training game if none)
    """

    def __init__(self, features, labels, games, train_indices):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels, self.games = labels, games
        self.by_label, self.by_game = defaultdict(list), defaultdict(list)
        for i in train_indices:
            self.by_label[labels[i]].append(i)
            self.by_game[games[i]].append(i)
        # A positive needs the same label in another game.
        eligible = {lab for lab, idx in self.by_label.items() if len({games[i] for i in idx}) >= 2}
        self.anchors = [i for i in train_indices if labels[i] in eligible]
        self.train_indices = list(train_indices)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        a = self.anchors[idx]
        label, game = self.labels[a], self.games[a]
        pos = random.choice([i for i in self.by_label[label] if self.games[i] != game])
        neg_pool = [i for i in self.by_game[game] if self.labels[i] != label] or \
                   [i for i in self.train_indices if self.labels[i] != label]
        neg = random.choice(neg_pool)
        return self.features[a], self.features[pos], self.features[neg]


class ProjectionHead(nn.Module):
    """MLP projection: input_dim -> hidden -> hidden -> output_dim, L2-normalized."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def plot_history(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Contrastive Learning Training Curves (Triplet Loss, training games)",
                 fontsize=13, fontweight="bold")
    axes[0].plot(history["epoch"], history["loss"], color="tab:red")
    axes[0].set(xlabel="Epoch", ylabel="Triplet Loss", title="Loss")
    axes[1].plot(history["epoch"], history["pos_dist"], label="d(anchor, positive)", color="tab:green")
    axes[1].plot(history["epoch"], history["neg_dist"], label="d(anchor, negative)", color="tab:red")
    axes[1].set(xlabel="Epoch", ylabel="L2 Distance", title="Positive vs Negative Distance")
    axes[1].legend(fontsize=9)
    axes[2].plot(history["epoch"], history["triplet_acc"], color="tab:blue")
    axes[2].set(xlabel="Epoch", ylabel="Accuracy", title="Triplet Accuracy (d_pos < d_neg)", ylim=(0, 1.05))
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    cfg = load_config()
    prefix = cfg["features"]["model"].split("/")[-1]
    parser = argparse.ArgumentParser(description="Train a triplet-loss projection head on training games.")
    parser.add_argument("--features-dir", default=str(data_path(cfg, "features") / f"temporal_{prefix}"))
    parser.add_argument("--out-dir", default=str(data_path(cfg, "features") / "contrastive"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--train-games", nargs="+", default=None,
                        help="games to mine triplets from (default: the manifest's train split)")
    parser.add_argument("--keep-all-labels", action="store_true")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto (GPU only if it has free memory)")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    games_manifest = load_manifest(repo_path(cfg, "manifest"))
    test_games = game_nums(games_manifest, "test")
    train_games = sorted({normalize_game_id(g) for g in args.train_games} if args.train_games
                         else game_nums(games_manifest, "train"))
    assert_disjoint(train=train_games, test=test_games)

    tf = load_temporal(args.features_dir, games=manifest_games(cfg))
    if not args.keep_all_labels:
        tf = tf.drop_labels(dropped_labels(cfg))
    feat = mean_pool(tf.features, tf.masks)
    train_idx = np.flatnonzero(np.isin(tf.games, train_games))
    missing = sorted(set(train_games) - set(tf.games))
    print(f"Events: {len(tf)}, dim {feat.shape[1]}; training games {len(train_games)} "
          f"({len(train_idx)} events){'; missing from features: ' + ' '.join(missing) if missing else ''}")

    dataset = TripletDataset(feat, tf.labels, tf.games, train_idx)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    print(f"Anchors (labels present in 2+ training games): {len(dataset)}")

    model = ProjectionHead(feat.shape[1], args.hidden_dim, args.output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)

    history = {"epoch": [], "loss": [], "pos_dist": [], "neg_dist": [], "triplet_acc": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = {"loss": 0.0, "pos": 0.0, "neg": 0.0, "correct": 0, "n": 0}
        for a, p, n in loader:
            a_emb, p_emb, n_emb = (model(x.to(device)) for x in (a, p, n))
            loss = loss_fn(a_emb, p_emb, n_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                d_pos = F.pairwise_distance(a_emb, p_emb)
                d_neg = F.pairwise_distance(a_emb, n_emb)
            b = a.size(0)
            tot["loss"] += loss.item() * b
            tot["pos"] += d_pos.sum().item()
            tot["neg"] += d_neg.sum().item()
            tot["correct"] += (d_pos < d_neg).sum().item()
            tot["n"] += b
        scheduler.step()
        history["epoch"].append(epoch)
        history["loss"].append(tot["loss"] / tot["n"])
        history["pos_dist"].append(tot["pos"] / tot["n"])
        history["neg_dist"].append(tot["neg"] / tot["n"])
        history["triplet_acc"].append(tot["correct"] / tot["n"])
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  Loss={history['loss'][-1]:.4f}  "
                  f"d(a,p)={history['pos_dist'][-1]:.4f}  d(a,n)={history['neg_dist'][-1]:.4f}  "
                  f"TripletAcc={history['triplet_acc'][-1]:.3f}")

    torch.save({"model_state_dict": model.state_dict(), "input_dim": feat.shape[1],
                "hidden_dim": args.hidden_dim, "output_dim": args.output_dim, "margin": args.margin,
                "epochs": args.epochs, "train_games": train_games}, out_dir / "projection_head.pt")

    model.eval()
    with torch.no_grad():
        projected = model(torch.tensor(feat, dtype=torch.float32, device=device)).cpu().numpy()
    np.savez_compressed(out_dir / "contrastive_embeddings.npz", features=projected, labels=tf.labels,
                        games=tf.games, halves=tf.halves, event_tags=tf.event_tags,
                        train_games=np.array(train_games))
    (out_dir / "run_info.json").write_text(json.dumps({**vars(args), "train_games": train_games},
                                                      indent=2, default=str))
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    plot_history(history, out_dir / "training_curves.png")
    print(f"\nSaved head, embeddings ({projected.shape}) and history to {out_dir}")
    print(f"Final training-game triplet accuracy: {history['triplet_acc'][-1]:.3f} "
          "(training data; see evaluate_contrastive.py for held-out games)")


if __name__ == "__main__":
    main()
