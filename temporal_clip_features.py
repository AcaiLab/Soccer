"""
Temporal Feature Extraction for Soccer Events
==============================================
Groups extracted frames by event (game x half x label x event_tag), encodes
each frame with a frozen vision model (DINOv2 or CLIP, see model_encoders.py)
and saves per-event feature tensors of shape (n_frames, embed_dim).

With a +/-30 s window at 1 fps there are up to 60 frames per event.

Filename pattern expected:
    {label}_frame_{event_tag}_t{frame_tag}.jpg
    e.g.  corner_frame_09-00.000_t08-45.000.jpg

Outputs (in --out, default features/temporal_<model name>)
----------------------------------------------------------
  clip_temporal_padded.npz    - padded features (E, T, D) + masks (E, T) + labels,
                                games, halves, event_tags (name kept for compatibility)
  temporal_metadata.csv       - one row per event, with match_id and split
  run_info.json               - model, frame root and frame counts

Usage
-----
    python temporal_clip_features.py
    python temporal_clip_features.py --model facebook/dinov2-base --batch-size 64 --device cuda
"""

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from soccer.annotations import tag_to_seconds
from soccer.config import data_path, load_config, repo_path
from soccer.extract import FRAME_RE
from soccer.manifest import load_manifest, normalize_game_id


def parse_half_dir(name: str) -> tuple[str, str] | None:
    """"game_07_half_2" -> ("07", "2"); None for anything else."""
    parts = name.split("_")
    if len(parts) == 4 and parts[0] == "game" and parts[2] == "half":
        return normalize_game_id(parts[1]), parts[3]
    return None


def scan_events(frames_root: pathlib.Path) -> dict:
    """(game, half, label, event_tag) -> [(frame_seconds, path), ...]"""
    events = defaultdict(list)
    for half_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        ids = parse_half_dir(half_dir.name)
        frames_dir = half_dir / "frames"
        if ids is None or not frames_dir.is_dir():
            continue
        for label_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
            for img in sorted(label_dir.iterdir()):
                m = FRAME_RE.search(img.name)
                if m:
                    events[(*ids, label_dir.name, m.group(1))].append((tag_to_seconds(m.group(2)), str(img)))
    return events


def select_frames(frames: list, max_frames: int) -> list:
    """Time-ordered frames, uniformly subsampled to at most max_frames."""
    frames = sorted(frames)
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]
    return frames


def load_safe(path: str):
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception as e:  # corrupt or truncated file: skip the frame, keep the event
        print(f"  [WARN] Skipping {path}: {e}")
        return None


def encode_events(encoder, events: dict, max_frames: int, batch_size: int, progress: bool = True):
    """Encode every frame; return (padded (E,T,D), masks (E,T), keys, n_frames per event)."""
    keys = sorted(events)
    frame_lists = [select_frames(events[k], max_frames) for k in keys]
    flat = [(i, p) for i, frames in enumerate(frame_lists) for _, p in frames]

    padded = np.zeros((len(keys), max_frames, encoder.embed_dim), dtype=np.float32)
    masks = np.zeros((len(keys), max_frames), dtype=bool)
    filled = np.zeros(len(keys), dtype=int)
    starts = range(0, len(flat), batch_size)
    for start in tqdm(starts, unit="batch", disable=not progress):
        batch = [(i, load_safe(p)) for i, p in flat[start:start + batch_size]]
        batch = [(i, img) for i, img in batch if img is not None]
        if not batch:
            continue
        feats = encoder.encode_batch([img for _, img in batch])
        for (i, _), feat in zip(batch, feats, strict=True):
            padded[i, filled[i]] = feat
            masks[i, filled[i]] = True
            filled[i] += 1
    return padded, masks, keys, [len(f) for f in frame_lists]


def save_outputs(out_dir: pathlib.Path, padded, masks, keys, n_frames, manifest_games=None) -> pd.DataFrame:
    games, halves, labels, tags = (np.array(c) for c in zip(*keys, strict=True)) if keys else ([],) * 4
    np.savez_compressed(out_dir / "clip_temporal_padded.npz", features=padded, masks=masks,
                        labels=labels, games=games, halves=halves, event_tags=tags)
    by_num = {g.game_num: g for g in manifest_games or []}
    df = pd.DataFrame({
        "game": games, "half": halves, "label": labels, "event_tag": tags,
        "n_frames": n_frames, "n_valid": masks.sum(axis=1),
        "match_id": [by_num[g].match_id if g in by_num else "" for g in games],
        "split": [by_num[g].split if g in by_num else "" for g in games],
    })
    df.to_csv(out_dir / "temporal_metadata.csv", index=False)
    return df


def main(argv=None):
    cfg = load_config()
    fcfg = cfg["features"]
    parser = argparse.ArgumentParser(description="Extract per-event temporal features from soccer frames.")
    parser.add_argument("--frames-root", default=str(data_path(cfg, "frames")),
                        help="Root dir containing game_XX_half_N/frames/<label>/*.jpg")
    parser.add_argument("--out", default=None, help="default: features/temporal_<model name>")
    parser.add_argument("--model", default=fcfg["model"], help="HuggingFace model name (DINOv2 or CLIP)")
    parser.add_argument("--max-frames", type=int, default=fcfg["max_frames"],
                        help="Max frames kept per event (uniform subsample if exceeded)")
    parser.add_argument("--batch-size", type=int, default=fcfg["batch_size"])
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    args = parser.parse_args(argv)

    import torch

    from model_encoders import build_encoder

    frames_root = pathlib.Path(args.frames_root)
    out_dir = pathlib.Path(args.out or data_path(cfg, "features") / f"temporal_{args.model.split('/')[-1]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Model: {args.model}  Max frames: {args.max_frames}")
    print(f"Input: {frames_root}\nOutput: {out_dir}")

    events = scan_events(frames_root)
    if not events:
        raise SystemExit(f"No frames found under {frames_root}")
    n_per = [len(v) for v in events.values()]
    print(f"Found {len(events)} events across {len({k[2] for k in events})} labels; "
          f"frames per event min={min(n_per)} max={max(n_per)} mean={np.mean(n_per):.1f}")

    encoder = build_encoder(args.model, device)
    padded, masks, keys, n_frames = encode_events(encoder, events, args.max_frames, args.batch_size)
    games = load_manifest(repo_path(cfg, "manifest"))
    df = save_outputs(out_dir, padded, masks, keys, n_frames, games)
    (out_dir / "run_info.json").write_text(json.dumps({
        "model": args.model, "frames_root": str(frames_root), "max_frames": args.max_frames,
        "events": len(keys), "frames_encoded": int(masks.sum()), "frames_found": int(sum(n_per)),
    }, indent=2))

    print(f"\nSaved {out_dir / 'clip_temporal_padded.npz'}  features {padded.shape}")
    print(f"Frames encoded: {int(masks.sum())} / {sum(min(n, args.max_frames) for n in n_per)}")
    print("\nLabel distribution (events):")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
