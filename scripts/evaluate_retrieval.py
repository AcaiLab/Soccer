import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import ensure_dir, repo_root, write_json


def average_precision(relevance: np.ndarray) -> float:
    hits = 0
    precision_sum = 0.0
    for i, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            precision_sum += hits / i
    return precision_sum / hits if hits else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, default=Path("retrieval/memory_mobilenet_v3_small_mean_std.npz"))
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = np.load(args.memory, allow_pickle=True)
    features = data["features"].astype(np.float32)
    labels = data["event_labels"].astype(str)
    games = data["game_labels"].astype(str)
    window_ids = data["window_ids"].astype(str)
    sims = features @ features.T
    np.fill_diagonal(sims, -np.inf)
    max_k = min(max(args.k), len(features) - 1)
    order = np.argsort(-sims, axis=1)[:, :max_k]

    rows = []
    for i in range(len(features)):
        neighbor_labels = labels[order[i]]
        neighbor_games = games[order[i]]
        rel = neighbor_labels == labels[i]
        row = {
            "window_id": window_ids[i],
            "event_label": labels[i],
            "game_label": games[i],
            "ap_at_max_k": average_precision(rel),
            "cross_game_rate_at_max_k": float(np.mean(neighbor_games != games[i])),
        }
        for k in args.k:
            kk = min(k, max_k)
            row[f"recall_at_{k}"] = float(np.any(rel[:kk]))
            row[f"purity_at_{k}"] = float(np.mean(rel[:kk]))
            row[f"cross_game_rate_at_{k}"] = float(np.mean(neighbor_games[:kk] != games[i]))
        rows.append(row)

    df = pd.DataFrame(rows)
    reports = ensure_dir(repo_root() / "reports")
    out_csv = args.out_csv or reports / f"{args.memory.stem}_retrieval_eval.csv"
    out_summary = args.out_summary or reports / f"{args.memory.stem}_retrieval_summary.json"
    df.to_csv(out_csv, index=False)
    metric_cols = [c for c in df.columns if c.startswith(("recall", "purity", "cross_game", "ap_"))]
    summary = {
        "memory": str(args.memory),
        "windows": int(len(df)),
        "labels": int(pd.Series(labels).nunique()),
        "out_csv": str(out_csv),
        "metrics": {col: float(df[col].mean()) for col in metric_cols},
        "by_label": df.groupby("event_label")[metric_cols].mean().round(4).to_dict(orient="index"),
    }
    write_json(out_summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
