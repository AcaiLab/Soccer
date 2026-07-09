import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import ensure_dir, l2_normalize, load_config, repo_root, resolve_path, write_json


def make_window_features(window_embeddings: np.ndarray, mode: str) -> np.ndarray:
    x = window_embeddings.astype(np.float32)
    if mode == "mean":
        return x.mean(axis=1)
    if mode == "mean_std":
        return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1)
    if mode == "concat":
        return x.reshape(x.shape[0], -1)
    raise ValueError(f"Unknown retrieval feature mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--feature-mode", default="mean_std", choices=["mean", "mean_std", "concat"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    base = repo_root()
    output_root = resolve_path(cfg["project"]["output_root"], base) or base
    retrieval_dir = ensure_dir(output_root / "retrieval")
    embeddings_path = args.embeddings or output_root / "features" / f"visual_{cfg['embeddings']['visual_model']}_local.npz"
    out = args.out or retrieval_dir / f"memory_{cfg['embeddings']['visual_model']}_{args.feature_mode}.npz"

    data = np.load(embeddings_path, allow_pickle=True)
    features = make_window_features(data["window_embeddings"], args.feature_mode)
    features = l2_normalize(features.astype(np.float32))
    np.savez_compressed(
        out,
        features=features,
        window_ids=data["window_ids"],
        event_labels=data["event_labels"],
        game_labels=data["game_labels"],
        feature_mode=args.feature_mode,
    )
    catalog = pd.DataFrame(
        {
            "row": np.arange(len(features)),
            "window_id": data["window_ids"],
            "event_label": data["event_labels"],
            "game_label": data["game_labels"],
        }
    )
    catalog.to_csv(out.with_suffix(".catalog.csv"), index=False)
    meta = {
        "embeddings": str(embeddings_path),
        "output": str(out),
        "feature_mode": args.feature_mode,
        "windows": int(len(features)),
        "feature_dim": int(features.shape[1]),
    }
    write_json(out.with_suffix(".metadata.json"), meta)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
