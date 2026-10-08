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

from soccer.frames import half_marker, select_frames


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


def _fingerprint(meta: dict, frames: dict) -> str:
    """What a cached half depends on: encoder settings, the exact frame files, and
    the half's extraction marker (window settings, source video, annotation file),
    so re-extracting with other settings invalidates the cache even if names match."""
    import hashlib
    import json

    names = sorted(f"{k[2]}/{pathlib.Path(p).name}" for k, fs in frames.items() for _, p in fs)
    first = next(p for fs in frames.values() for _, p in fs)
    return json.dumps({**meta, "frames": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                       "extraction": hashlib.sha256(half_marker(first).encode()).hexdigest()}, sort_keys=True)


def encode_cached(get_encoder, events: dict, max_frames: int, batch_size: int,
                  cache_dir: pathlib.Path, meta: dict):
    """encode_events, cached per game half under cache_dir.

    A half is re-encoded only if its frames or `meta` (model, max_frames) changed,
    so adding games encodes just the new ones. `get_encoder` is called at most
    once, and only if something needs encoding. Returns the same tuple as
    encode_events plus (halves reused, halves encoded).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_half: dict = {}
    for k, frames in events.items():
        by_half.setdefault((k[0], k[1]), {})[k] = frames
    encoder, reused, encoded = None, 0, 0
    parts = []
    for game, half in sorted(by_half):
        group = by_half[(game, half)]
        fp = _fingerprint(meta, group)
        path = cache_dir / f"game_{game}_half_{half}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as z:
                if str(z["fingerprint"]) == fp:
                    keys = [tuple(k) for k in z["keys"].tolist()]
                    parts.append((z["padded"], z["masks"], keys, z["n_frames"].tolist()))
                    reused += 1
                    continue
        encoder = encoder or get_encoder()
        padded, masks, keys, n_frames = encode_events(encoder, group, max_frames, batch_size, progress=False)
        np.savez(path, padded=padded, masks=masks, keys=np.array(keys), n_frames=np.array(n_frames),
                 fingerprint=np.array(fp))
        parts.append((padded, masks, keys, n_frames))
        encoded += 1
        print(f"  encoded game {game} half {half}: {len(keys)} events", flush=True)
    padded = np.concatenate([p[0] for p in parts])
    masks = np.concatenate([p[1] for p in parts])
    keys = [k for p in parts for k in p[2]]
    n_frames = [n for p in parts for n in p[3]]
    return padded, masks, keys, n_frames, reused, encoded


def pick_device(requested: str | None, min_free_gb: float = 4.0) -> str:
    """'cuda' only if a GPU exists and has min_free_gb free (shared GPUs fill up)."""
    if requested and requested != "auto":
        return requested
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    try:
        free = torch.cuda.mem_get_info()[0] / 1e9
    except RuntimeError as e:  # a GPU that is completely full cannot even create a CUDA context
        print(f"GPU unusable ({str(e).splitlines()[0]}); using the CPU", flush=True)
        return "cpu"
    if free < min_free_gb:
        print(f"GPU has only {free:.1f} GB free (another job?); using the CPU", flush=True)
        return "cpu"
    return "cuda"
