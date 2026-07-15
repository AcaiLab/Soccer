"""
Model-Agnostic Frame Feature Extraction
========================================
Extracts per-frame embeddings from any vision model (CLIP, DINOv2, etc.)
using the unified encoder interface from model_encoders.py.

Usage:
    python extract_frame_features.py
    python extract_frame_features.py --model facebook/dinov2-base --out features/dino_flat
    python extract_frame_features.py --model facebook/dinov2-large --batch-size 64 --device cuda
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from model_encoders import build_encoder
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Extract image features from soccer event frames using any model.")
parser.add_argument(
    "--frames-root",
    default="results_by_label_epl_2021_2022",
    help="Root directory containing game/half/frames/label/*.png structure",
)
parser.add_argument(
    "--out",
    default="features",
    help="Output directory for saved features",
)
parser.add_argument(
    "--model",
    default="facebook/dinov2-base",
    help="HuggingFace model name (CLIP, DINOv2, etc.)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Number of images to process per batch",
)
parser.add_argument(
    "--device",
    default=None,
    help="Device to run model on (cuda / cpu). Auto-detected if not specified.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
frames_root = pathlib.Path(args.frames_root)
out_dir = pathlib.Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Model  : {args.model}")
print(f"Input  : {frames_root}")
print(f"Output : {out_dir}")

# Extract model prefix for output filenames (e.g., "dinov2-base" from "facebook/dinov2-base")
model_prefix = args.model.split("/")[-1]

# ---------------------------------------------------------------------------
# Load encoder (CLIP, DINOv2, or any HuggingFace vision model)
# ---------------------------------------------------------------------------
print("\nLoading encoder ...")
encoder = build_encoder(args.model, device)
embed_dim = encoder.embed_dim
print(f"Embedding dimension: {embed_dim}\n")

# ---------------------------------------------------------------------------
# Collect all frame paths with metadata
# ---------------------------------------------------------------------------
records = []  # list of dicts: {game, half, label, filepath}

for game_half_dir in sorted(frames_root.iterdir()):
    if not game_half_dir.is_dir():
        continue

    frames_dir = game_half_dir / "frames"
    if not frames_dir.exists():
        continue

    # Parse game and half from folder name, e.g. "game_01_half_1"
    parts = game_half_dir.name.split("_")
    # Expected format: game_{XX}_half_{N}
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

        for img_path in sorted(list(label_dir.glob("*.png")) +
                               list(label_dir.glob("*.jpg"))):
            records.append({
                "game":     game_id,
                "half":     half_id,
                "label":    label,
                "filepath": str(img_path),
            })

total = len(records)
print(f"Found {total} frames across {len(set(r['label'] for r in records))} labels.\n")

if total == 0:
    print("No frames found. Check --frames-root path.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Extract features in batches
# ---------------------------------------------------------------------------
def load_image_safe(path: str) -> Image.Image | None:
    """Open an image, returning None if the file is corrupted or unreadable."""
    try:
        img = Image.open(path)
        img.load()          # force full decode so truncation errors surface here
        return img.convert("RGB")
    except Exception as e:
        print(f"\n  [WARN] Skipping corrupted image {path}: {e}")
        return None


def extract_features_batch(image_paths: list) -> tuple:
    """Load images, run through encoder, return (N, D) array and valid paths."""
    loaded = [(p, load_image_safe(p)) for p in image_paths]
    valid_paths  = [p   for p, img in loaded if img is not None]
    valid_images = [img for p, img in loaded if img is not None]

    if not valid_images:
        return np.empty((0, embed_dim), dtype=np.float32), []

    feats = encoder.encode_batch(valid_images)
    return feats, valid_paths


all_features = []
valid_records = []   # only records for successfully loaded images
batch_size = args.batch_size
filepaths = [r["filepath"] for r in records]

print("Extracting features …")
for start in tqdm(range(0, total, batch_size), unit="batch"):
    batch_paths = filepaths[start: start + batch_size]
    batch_records = records[start: start + batch_size]

    feats, valid_paths = extract_features_batch(batch_paths)
    if feats.shape[0] == 0:
        continue

    valid_set = set(valid_paths)
    all_features.append(feats)
    valid_records.extend(r for r in batch_records if r["filepath"] in valid_set)

all_features = np.vstack(all_features)  # shape: (N, embed_dim)
skipped = total - len(valid_records)
print(f"\nFeature matrix shape : {all_features.shape}")
print(f"Skipped (corrupted)  : {skipped} frames")

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
# 1. Save feature matrix as .npy
features_path = out_dir / f"{model_prefix}_features.npy"
np.save(features_path, all_features)
print(f"Saved features  -> {features_path}")

# 2. Save metadata as CSV (rows align 1-to-1 with feature matrix rows)
metadata_path = out_dir / "metadata.csv"
df = pd.DataFrame(valid_records)
df["feature_idx"] = range(len(df))
df.to_csv(metadata_path, index=False)
print(f"Saved metadata  -> {metadata_path}")

# 3. Save everything together as a compressed .npz archive
npz_path = out_dir / f"{model_prefix}_features.npz"
np.savez_compressed(
    npz_path,
    features=all_features,
    labels=np.array([r["label"] for r in valid_records]),
    games=np.array([r["game"] for r in valid_records]),
    halves=np.array([r["half"] for r in valid_records]),
    filepaths=np.array([r["filepath"] for r in valid_records]),
)
print(f"Saved .npz      -> {npz_path}")

# ---------------------------------------------------------------------------
# Quick sanity check: label distribution in extracted set
# ---------------------------------------------------------------------------
print("\nLabel distribution:")
label_counts = df["label"].value_counts()
for label, count in label_counts.items():
    print(f"  {label:<30s} {count:5d} frames")

print(f"\nDone. {total} frames encoded into {embed_dim}-dim {args.model} embeddings.")
