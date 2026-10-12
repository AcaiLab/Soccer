"""
Sequence models over per-frame DINOv2 features for event classification.

Instead of pooling the 60-frame window into a single vector, these models
consume the full (60, 768) sequence, using temporal ordering:

    1. MeanPool-MLP  : masked mean over frames -> MLP        (baseline)
    2. BiLSTM        : bidirectional LSTM -> masked mean of hidden states
    3. TemporalCNN   : stacked 1D convolutions over time -> masked mean
    4. Transformer   : self-attention over time with a padding mask

Protocol (leak-free, by game, from the manifest):
    train games fit the weights, val games pick the stopping epoch and the
    model, and only the chosen model is scored on the test games. With
    --seeds, every model is trained once per seed, the model with the best
    mean validation accuracy is chosen, and its test accuracy is reported as
    mean and standard deviation over seeds plus a 95% interval from
    resampling test games.
Per-dimension standardisation uses training frames only.

Usage:
    python scripts/sequence_classifier.py
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from soccer.config import data_path, dropped_labels, load_config, repo_path
from soccer.encoding import pick_device
from soccer.evaluation import prediction_rows, split_by_game
from soccer.features import load_temporal
from soccer.manifest import load_manifest, manifest_games
from soccer.metrics import bootstrap_ci, classification_scores


def masked_mean(h, m):
    """h: (B, T, D), m: (B, T) -> (B, D)"""
    return (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


class MeanPoolMLP(nn.Module):
    def __init__(self, d_in, d_h, n_cls, max_len):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_h, n_cls))

    def forward(self, x, m):
        return self.net(masked_mean(x, m))


class BiLSTM(nn.Module):
    def __init__(self, d_in, d_h, n_cls, max_len):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.lstm = nn.LSTM(d_h, d_h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d_h, n_cls))

    def forward(self, x, m):
        # Pack so the backward direction starts at each event's last real frame, not in its
        # padding (real frames come first; events at the start of a half have fewer).
        lengths = m.sum(1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(torch.relu(self.proj(x)), lengths, batch_first=True,
                                                   enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        return self.head(masked_mean(h, m))


class TemporalCNN(nn.Module):
    def __init__(self, d_in, d_h, n_cls, max_len):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, d_h, 5, padding=2), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(d_h, d_h, 5, padding=2), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(d_h, n_cls)

    def forward(self, x, m):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d_h)
        return self.head(masked_mean(h, m))


class TransformerClassifier(nn.Module):
    def __init__(self, d_in, d_h, n_cls, max_len, n_heads=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_h) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_h, nhead=n_heads, dim_feedforward=d_h * 4,
                                           dropout=0.3, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_h, n_cls))

    def forward(self, x, m):
        h = self.proj(x) + self.pos[:, :x.size(1)]
        h = self.encoder(h, src_key_padding_mask=~m.bool())
        return self.head(masked_mean(h, m))


MODELS = {"MeanPool-MLP": MeanPoolMLP, "BiLSTM": BiLSTM, "TemporalCNN": TemporalCNN,
          "Transformer": TransformerClassifier}


def main(argv=None):
    cfg = load_config()
    prefix = cfg["features"]["model"].split("/")[-1]
    parser = argparse.ArgumentParser(description="Sequence classifiers over per-frame features.")
    parser.add_argument("--features", default=str(data_path(cfg, "features") / f"temporal_{prefix}"))
    parser.add_argument("--out-dir", default=str(data_path(cfg, "results") / "classification"))
    parser.add_argument("--val-games", nargs="+", default=None, help="default: manifest val split")
    parser.add_argument("--test-games", nargs="+", default=None, help="default: manifest test split")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[cfg["seed"]])
    parser.add_argument("--keep-all-labels", action="store_true")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto (GPU only if it has free memory)")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = load_temporal(args.features, games=manifest_games(cfg))
    if not args.keep_all_labels:
        tf = tf.drop_labels(dropped_labels(cfg))
    split = split_by_game(tf.games, tf.labels, load_manifest(repo_path(cfg, "manifest")),
                          args.val_games, args.test_games)
    classes = sorted(set(tf.labels[split.train]))
    cls_idx = {c: i for i, c in enumerate(classes)}
    keep = split.train | split.val | split.test
    y = np.array([cls_idx.get(label, -1) for label in tf.labels])
    print(f"Events: train={split.train.sum()} val={split.val.sum()} test={split.test.sum()}  "
          f"classes={len(classes)}  dim={tf.features.shape[2]}")

    masks = tf.masks.astype(np.float32)
    train_frames = tf.features[split.train][tf.masks[split.train]]
    mu, sd = train_frames.mean(0), train_frames.std(0) + 1e-6
    feats = ((tf.features - mu) / sd) * masks[:, :, None]  # keep padding at zero
    X, M, Y = torch.tensor(feats), torch.tensor(masks), torch.tensor(y, dtype=torch.long)

    def loader(mask, shuffle, generator=None):
        ds = torch.utils.data.TensorDataset(X[mask & keep], M[mask & keep], Y[mask & keep])
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, generator=generator)

    def predict(model, mask):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xb.to(device), mb.to(device)).argmax(1).cpu()
                              for xb, mb, _ in loader(mask, False)]).numpy()

    def train(cls, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = cls(tf.features.shape[2], args.hidden_dim, len(classes), tf.features.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        best_val, best_state, wait, stopped = -1.0, None, 0, 0
        # One generator per run, so each epoch gets a new (but reproducible) order.
        order = torch.Generator().manual_seed(seed)
        for epoch in range(1, args.epochs + 1):
            stopped = epoch
            model.train()
            for xb, mb, yb in loader(split.train, True, order):
                opt.zero_grad()
                crit(model(xb.to(device), mb.to(device)), yb.to(device)).backward()
                opt.step()
            val_acc = float(np.mean(predict(model, split.val) == y[split.val & keep]))
            if val_acc > best_val:
                best_val, wait = val_acc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= args.patience:
                    break
        model.load_state_dict(best_state)
        return model, best_val, stopped

    y_test, y_train = y[split.test & keep], y[split.train & keep]
    test_games = tf.games[split.test & keep]
    rows, runs = [], {}
    for name, cls in MODELS.items():
        for seed in args.seeds:
            model, val_acc, stopped = train(cls, seed)
            runs[(name, seed)] = model
            rows.append({"model": name, "seed": seed, "val_acc": val_acc, "stopped_epoch": stopped})
            print(f"{name:14s} seed {seed}  stopped@{stopped:3d}  best val accuracy={val_acc:.1%}")
    df = pd.DataFrame(rows)
    by_model = df.groupby("model")["val_acc"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    chosen = by_model.index[0]

    # Test only the chosen model, once per seed.
    test_rows, preds = [], []
    for seed in args.seeds:
        pred = predict(runs[(chosen, seed)], split.test)
        preds.append(pred)
        test_rows.append({"seed": seed, **classification_scores(y_test, pred, y_train)})
    test_df = pd.DataFrame(test_rows)
    in_test = split.test & keep
    pd.concat([prediction_rows(test_games, tf.halves[in_test], tf.event_tags[in_test], y_test, pred, seed=seed)
               for seed, pred in zip(args.seeds, preds, strict=True)]).to_csv(
        out_dir / "test_predictions.csv", index=False)
    ci = bootstrap_ci(np.tile(y_test, len(preds)), np.concatenate(preds), np.tile(test_games, len(preds)),
                      seed=args.seeds[0])

    df.to_csv(out_dir / "sequence_results.csv", index=False)
    test_df.to_csv(out_dir / "sequence_test_by_seed.csv", index=False)
    acc, f1 = test_df["Accuracy"], test_df["Macro_F1"]
    spread = (lambda v: f" +/- {v.std(ddof=1):.4f} (sd over {len(v)} seeds)") if len(args.seeds) > 1 else (lambda v: "")
    summary = (f"Chosen on validation: {chosen} (mean val accuracy {by_model.loc[chosen, 'mean']:.4f} "
               f"over {len(args.seeds)} seed(s))\n"
               f"Test games: {' '.join(split.test_games)}  ({len(y_test)} events)\n"
               f"Test accuracy {acc.mean():.4f}{spread(acc)}  macro-F1 {f1.mean():.4f}{spread(f1)}  "
               f"majority-class baseline {test_df['Majority_baseline'].iloc[0]:.4f}\n"
               f"95% interval, resampling test games (all seeds pooled): accuracy "
               f"{ci['Accuracy_CI'][0]:.3f}-{ci['Accuracy_CI'][1]:.3f}, macro-F1 "
               f"{ci['Macro_F1_CI'][0]:.3f}-{ci['Macro_F1_CI'][1]:.3f}\n")
    present = sorted(set(y_test) | set(preds[0]))
    report = classification_report(y_test, preds[0], labels=present,
                                   target_names=[classes[i] for i in present], zero_division=0)
    details = (f"\nValidation accuracy by model (mean, sd over seeds):\n{by_model.to_string()}\n\n"
               f"Per-class report (seed {args.seeds[0]}):\n{report}")
    (out_dir / "sequence_classification_report.txt").write_text(summary + details)
    print("\n" + summary + details)


if __name__ == "__main__":
    main()
