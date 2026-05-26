"""
Temporal CLIP Feature Extraction for Soccer Events
===================================================
Groups extracted frames by event (game × half × label × event_tag),
encodes each frame individually with CLIP, and saves per-event feature
tensors of shape (n_frames, embed_dim).

With ±30 s extraction at 1 fps → up to 60 frames per event.

Filename pattern expected:
    {label}_frame_{event_tag}_t{frame_tag}.png
    e.g.  corner_frame_09-00.000_t08-45.000.png

Outputs (in --out directory)
----------------------------
  clip_temporal.pkl           – list of per-event dicts (unpadded arrays)
  clip_temporal_padded.npz    – padded features (E, T, D) + masks (E, T)
  temporal_metadata.csv       – event-level metadata

Usage
-----
    python temporal_clip_features.py
    python temporal_clip_features.py --max-frames 60 --batch-size 64 --device cuda
"""

import argparse
import pathlib
import re
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from model_encoders import build_encoder
from tqdm import tqdm

# Matches the event_tag and frame_tag at the end of frame filenames.
# Works regardless of how many underscores the label name has.
FRAME_RE = re.compile(r'_frame_(\d{2}-\d{2}\.\d{3})_t(\d{2}-\d{2}\.\d{3})\.png$')


def tag_to_sec(tag: str) -> float:
    """Convert 'MM-SS.mmm' filename tag to float seconds."""
    mm, ss = tag.split("-", 1)
    return int(mm) * 60.0 + float(ss)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Extract per-event temporal CLIP features from soccer frames."
)
parser.add_argument(
    "--frames-root",
    default="results_by_label_epl_2021_2022",
    help="Root dir containing game_XX_half_N/frames/label/*.png",
)
parser.add_argument("--out", default="features/temporal")
parser.add_argument("--model", default="openai/clip-vit-base-patch32",
                    help="HuggingFace CLIP model name")
parser.add_argument(
    "--max-frames",
    type=int,
    default=60,
    help="Max frames to keep per event (uniform subsample if exceeded). "
         "With ±30 s at 1 fps → 60 frames.",
)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument(
    "--device",
    default=None,
    help="cuda / cpu (auto-detected if omitted)",
)
args = parser.parse_args()

frames_root = pathlib.Path(args.frames_root)
out_dir     = pathlib.Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
device      = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
max_frames  = args.max_frames

print(f"Device      : {device}")
print(f"Model       : {args.model}")
print(f"Max frames  : {max_frames}")
print(f"Input       : {frames_root}")
print(f"Output      : {out_dir}")

# ---------------------------------------------------------------------------
# Load encoder (CLIP or DINOv2, auto-detected from --model name)
# ---------------------------------------------------------------------------
print("\nLoading encoder ...")
encoder = build_encoder(args.model, device)
print("Encoder loaded.\n")

# ---------------------------------------------------------------------------
# Step 1 – Scan frames and group by event
#   event_key = (game_id, half_id, label, event_tag)
#   value     = list of (frame_time_sec, path_str)
# ---------------------------------------------------------------------------
events: dict = defaultdict(list)

for game_half_dir in sorted(frames_root.iterdir()):
    if not game_half_dir.is_dir():
        continue
    frames_dir = game_half_dir / "frames"
    if not frames_dir.exists():
        continue

    parts = game_half_dir.name.split("_")
    try:
        game_id = parts[1]   # "01"
        half_id = parts[3]   # "1" or "2"
    except IndexError:
        game_id = game_half_dir.name
        half_id = "?"

    for label_dir in sorted(frames_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name

        for img_path in sorted(label_dir.glob("*.png")):
            m = FRAME_RE.search(img_path.name)
            if not m:
                continue
            event_tag, frame_tag = m.group(1), m.group(2)
            events[(game_id, half_id, label, event_tag)].append(
                (tag_to_sec(frame_tag), str(img_path))
            )

event_keys = sorted(events.keys())
n_events   = len(event_keys)
print(f"Found {n_events} events across "
      f"{len(set(k[2] for k in event_keys))} labels.\n")

# ---------------------------------------------------------------------------
# Step 2 – Sort frames per event; subsample to max_frames if needed
# ---------------------------------------------------------------------------
event_frame_lists: list[list] = []
for key in event_keys:
    frames = sorted(events[key], key=lambda x: x[0])   # sort by time
    if len(frames) > max_frames:
        idxs   = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]
    event_frame_lists.append(frames)

n_per_event = [len(f) for f in event_frame_lists]
print(f"Frames per event  min={min(n_per_event)}  "
      f"max={max(n_per_event)}  mean={np.mean(n_per_event):.1f}")

# ---------------------------------------------------------------------------
# Step 3 – Flatten all selected frames into a single ordered list
# ---------------------------------------------------------------------------
flat_paths:      list[str] = []
flat_event_idx:  list[int] = []

for i, frames in enumerate(event_frame_lists):
    for _, p in frames:
        flat_paths.append(p)
        flat_event_idx.append(i)

n_total = len(flat_paths)
print(f"Total frames to encode : {n_total}\n")


def load_safe(path: str):
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception as e:
        print(f"  [WARN] Skipping {path}: {e}")
        return None


embed_dim = encoder.embed_dim
print(f"Embedding dim : {embed_dim}")

# ---------------------------------------------------------------------------
# Step 4 – Batch-encode all frames
# ---------------------------------------------------------------------------
all_features = np.zeros((n_total, embed_dim), dtype=np.float32)
valid_mask   = np.zeros(n_total, dtype=bool)

print("Encoding ...")
for start in tqdm(range(0, n_total, args.batch_size), unit="batch"):
    batch_paths = flat_paths[start: start + args.batch_size]

    loaded = [(j, load_safe(p)) for j, p in enumerate(batch_paths)]
    valid  = [(j, img) for j, img in loaded if img is not None]
    if not valid:
        continue

    offsets = [j   for j, _   in valid]
    images  = [img for _, img in valid]

    feats_np = encoder.encode_batch(images)
    for local_j, feat in zip(offsets, feats_np):
        global_j = start + local_j
        all_features[global_j] = feat
        valid_mask[global_j]   = True

flat_event_arr = np.array(flat_event_idx)

# ---------------------------------------------------------------------------
# Step 5 – Reconstruct per-event tensors
# ---------------------------------------------------------------------------
padded = np.zeros((n_events, max_frames, embed_dim), dtype=np.float32)
masks  = np.zeros((n_events, max_frames), dtype=np.bool_)

event_records = []
pkl_data      = []

for i, key in enumerate(event_keys):
    game_id, half_id, label, event_tag = key

    global_idxs = np.where(flat_event_arr == i)[0]
    valid_idxs  = global_idxs[valid_mask[global_idxs]]
    feats_i     = all_features[valid_idxs]   # (n_valid, embed_dim)
    n_valid     = len(valid_idxs)

    # Pad to max_frames (trailing rows stay zero)
    padded[i, :n_valid] = feats_i
    masks[i,  :n_valid] = True

    event_records.append({
        "game":      game_id,
        "half":      half_id,
        "label":     label,
        "event_tag": event_tag,
        "n_frames":  n_per_event[i],
        "n_valid":   n_valid,
    })
    pkl_data.append({
        "features":  feats_i,        # (n_valid, embed_dim) – unpadded
        "label":     label,
        "game":      game_id,
        "half":      half_id,
        "event_tag": event_tag,
    })

df = pd.DataFrame(event_records)

# ---------------------------------------------------------------------------
# Step 6 – Save
# ---------------------------------------------------------------------------

# (a) Padded numpy archive – fixed shape, ready for batched model training
#     features : (E, T, D)   E=events  T=max_frames  D=embed_dim
#     masks    : (E, T)       True for valid frames, False for padding
npz_path = out_dir / "clip_temporal_padded.npz"
np.savez_compressed(
    npz_path,
    features   = padded,
    masks      = masks,
    labels     = np.array(df["label"].tolist()),
    games      = np.array(df["game"].tolist()),
    halves     = np.array(df["half"].tolist()),
    event_tags = np.array(df["event_tag"].tolist()),
)
print(f"\nSaved padded features  -> {npz_path}")
print(f"  features : {padded.shape}   masks : {masks.shape}")

# (b) Pickle – list of per-event dicts with unpadded arrays
#     Easy to iterate; each dict['features'] has its own (n, D) shape
pkl_path = out_dir / "clip_temporal.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved pickle           -> {pkl_path}  ({len(pkl_data)} events)")

# (c) Metadata CSV – one row per event
meta_path = out_dir / "temporal_metadata.csv"
df.to_csv(meta_path, index=False)
print(f"Saved metadata         -> {meta_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nLabel distribution (events):")
print(df["label"].value_counts().to_string())

total_valid   = int(valid_mask.sum())
total_skipped = n_total - total_valid
print(f"\nFrames encoded  : {total_valid} / {n_total}  ({total_skipped} skipped)")
print(f"\nDone. {n_events} events  x  up to {max_frames} frames  x  {embed_dim}-dim features.")
