"""
Model-Agnostic Frame Feature Extraction
========================================
Extracts one embedding per frame (no grouping into events) with any vision
model supported by model_encoders.py. Used for the individual-frame clustering
in cluster_analysis.py.

Outputs (in --out, default features/<model name>_flat)
-------------------------------------------------------
  <model name>_features.npy   - (N, D) frame embeddings
  metadata.csv                - one row per embedding: game, half, label, filepath

Usage:
    python extract_frame_features.py
    python extract_frame_features.py --model openai/clip-vit-base-patch32
    python extract_frame_features.py --model facebook/dinov2-large --batch-size 64 --device cuda
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from soccer.config import data_path, load_config
from soccer.frames import scan_events
from temporal_clip_features import load_safe


def flat_output_dir(cfg: dict, model: str) -> pathlib.Path:
    return data_path(cfg, "features") / f"{model.split('/')[-1]}_flat"


def main(argv=None):
    cfg = load_config()
    fcfg = cfg["features"]
    parser = argparse.ArgumentParser(description="Extract per-frame image features with any model.")
    parser.add_argument("--frames-root", default=str(data_path(cfg, "frames")))
    parser.add_argument("--out", default=None, help="default: features/<model name>_flat")
    parser.add_argument("--model", default=fcfg["model"])
    parser.add_argument("--batch-size", type=int, default=fcfg["batch_size"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    import torch

    from model_encoders import build_encoder

    out_dir = pathlib.Path(args.out or flat_output_dir(cfg, args.model))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prefix = args.model.split("/")[-1]

    records = [{"game": g, "half": h, "label": label, "filepath": path}
               for (g, h, label, _), frames in sorted(scan_events(pathlib.Path(args.frames_root)).items())
               for _, path in sorted(frames)]
    if not records:
        raise SystemExit(f"No frames found under {args.frames_root}")
    print(f"Found {len(records)} frames across {len({r['label'] for r in records})} labels")

    encoder = build_encoder(args.model, device)
    feats, kept = [], []
    for start in tqdm(range(0, len(records), args.batch_size), unit="batch"):
        batch = [(r, load_safe(r["filepath"])) for r in records[start:start + args.batch_size]]
        batch = [(r, img) for r, img in batch if img is not None]
        if batch:
            feats.append(encoder.encode_batch([img for _, img in batch]))
            kept += [r for r, _ in batch]

    features = np.vstack(feats)
    np.save(out_dir / f"{prefix}_features.npy", features)
    df = pd.DataFrame(kept)
    df["feature_idx"] = range(len(df))
    df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"Saved {out_dir / f'{prefix}_features.npy'}  {features.shape}  "
          f"({len(records) - len(kept)} unreadable frames skipped)")


if __name__ == "__main__":
    main()
