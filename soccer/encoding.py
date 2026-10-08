"""Encoding extracted frames into padded per-event feature tensors.

Torch-free: the encoder object is passed in (soccer/encoders.py builds one),
so this module can be imported and tested without a GPU stack.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from soccer.frames import select_frames


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
