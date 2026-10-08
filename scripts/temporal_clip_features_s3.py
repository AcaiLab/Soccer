"""
Temporal Feature Extraction - S3 Streaming Version
====================================================
Downloads one game_XX_half_N frame folder at a time from S3, encodes it with
the same code as temporal_clip_features.py, saves an .npz part per folder,
deletes the local frames and moves on. Disk use stays at one half's frames;
progress survives restarts (Colab, spot instances).

AWS credentials come from the standard chain (instance or task role,
AWS_PROFILE); none are read from code.

Usage:
    python scripts/temporal_clip_features_s3.py --frames-uri s3://<bucket>/frames/
"""

import argparse
import json
import pathlib
import shutil
import tempfile

import numpy as np

from soccer.config import data_path, load_config, repo_path
from soccer.encoding import encode_events, save_outputs
from soccer.frames import parse_half_dir, scan_events
from soccer.manifest import load_manifest
from soccer.storage import download_prefix, list_subdirs


def main(argv=None):
    cfg = load_config()
    fcfg = cfg["features"]
    parser = argparse.ArgumentParser(description="Encode frames streamed from S3, one half at a time.")
    parser.add_argument("--frames-uri", required=True, help="s3://bucket/prefix holding game_XX_half_N/")
    parser.add_argument("--out", default=None, help="default: features/temporal_<model name>_s3")
    parser.add_argument("--model", default=fcfg["model"])
    parser.add_argument("--max-frames", type=int, default=fcfg["max_frames"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tmp-dir", default=None, help="scratch space (default: system temp)")
    args = parser.parse_args(argv)

    import torch

    from soccer.encoders import build_encoder

    out_dir = pathlib.Path(args.out or data_path(cfg, "features") / f"temporal_{args.model.split('/')[-1]}_s3")
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = pathlib.Path(args.tmp_dir or tempfile.gettempdir()) / "soccer_frames"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    progress_file = out_dir / "s3_progress.json"
    done = set(json.loads(progress_file.read_text())["done"]) if progress_file.exists() else set()
    half_dirs = [d for d in list_subdirs(args.frames_uri) if parse_half_dir(d)]
    remaining = [d for d in half_dirs if d not in done]
    print(f"Half folders: {len(half_dirs)}, done: {len(done)}, remaining: {len(remaining)}")

    if remaining:
        print(f"Loading encoder {args.model} on {device} ...")
        encoder = build_encoder(args.model, device)
    for i, name in enumerate(remaining, 1):
        work = tmp_root / f"{name}.work"  # holds only this half, even after a crash
        shutil.rmtree(work, ignore_errors=True)
        try:
            download_prefix(f"{args.frames_uri.rstrip('/')}/{name}/", work / name)
            events = scan_events(work)
            if events:
                padded, masks, keys, n_frames = encode_events(encoder, events, args.max_frames,
                                                              args.batch_size, progress=False)
                games, halves, labels, tags = (np.array(c) for c in zip(*keys, strict=True))
                np.savez_compressed(parts_dir / f"{name}.npz", features=padded, masks=masks,
                                    labels=labels, games=games, halves=halves, event_tags=tags,
                                    n_frames=np.array(n_frames))
            print(f"[{i}/{len(remaining)}] {name}: {len(events)} events", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        done.add(name)
        progress_file.write_text(json.dumps({"done": sorted(done)}))

    parts = sorted(parts_dir.glob("*.npz"))
    if not parts:
        raise SystemExit("No part files found")
    loaded = [np.load(p) for p in parts]
    padded = np.concatenate([d["features"] for d in loaded])
    masks = np.concatenate([d["masks"] for d in loaded])
    keys = [k for d in loaded for k in zip(d["games"], d["halves"], d["labels"], d["event_tags"], strict=True)]
    n_frames = np.concatenate([d["n_frames"] for d in loaded]).tolist()
    df = save_outputs(out_dir, padded, masks, keys, n_frames, load_manifest(repo_path(cfg, "manifest")))
    print(f"\nSaved {out_dir / 'clip_temporal_padded.npz'}  features {padded.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
