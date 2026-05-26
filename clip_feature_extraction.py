"""
CLIP Feature Extraction for Soccer Event Frames
================================================
Uses OpenAI's CLIP model (via HuggingFace transformers) to extract visual
features from soccer event frames extracted by soccerProjNewCode-1.py.

Based on: https://medium.com/@kerry.halupka/getting-started-with-openais-clip-a3b8f5277867

Model: openai/clip-vit-base-patch32
  - Image encoder: ViT-B/32 (Vision Transformer)
  - Text encoder: causal language model
  - Output image embedding: 512-dimensional vector

Usage:
    python clip_feature_extraction.py
    python clip_feature_extraction.py --frames-root results_by_label_epl_2021_2022 --out features
    python clip_feature_extraction.py --batch-size 64 --device cuda
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Extract CLIP image features from soccer event frames.")
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
    default="openai/clip-vit-base-patch32",
    help="HuggingFace CLIP model name",
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

# ---------------------------------------------------------------------------
# Load CLIP model and processor
# ---------------------------------------------------------------------------
# CLIPModel wraps two sub-models:
#   - a ViT-like transformer for visual features
#   - a causal language model for text features
# CLIPProcessor wraps:
#   - CLIPFeatureExtractor  (prepares images for the vision encoder)
#   - CLIPTokenizer         (encodes text for the language model)
print("\nLoading CLIP model …")
model = CLIPModel.from_pretrained(args.model).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
model.eval()
print("Model loaded.\n")

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

        for img_path in sorted(label_dir.glob("*.png")):
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
# Extract CLIP image features in batches
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


def extract_features_batch(image_paths: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load images, run through CLIP vision encoder, return (N, D) array and valid paths."""
    loaded = [(p, load_image_safe(p)) for p in image_paths]
    valid_paths  = [p   for p, img in loaded if img is not None]
    valid_images = [img for p, img in loaded if img is not None]

    if not valid_images:
        return np.empty((0, 512), dtype=np.float32), []

    # CLIPProcessor resizes and normalises images for the ViT encoder
    inputs = processor(images=valid_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        # Run the vision encoder (ViT-B/32) to get pooled CLS representation
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        # Project from vision hidden size (768) to shared embedding space (512)
        # This matches how CLIP aligns image and text embeddings
        image_features = model.visual_projection(vision_outputs.pooler_output)

    # L2-normalise so that cosine similarity == dot product
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy(), valid_paths


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

all_features = np.vstack(all_features)  # shape: (N, 512)
skipped = total - len(valid_records)
print(f"\nFeature matrix shape : {all_features.shape}")
print(f"Skipped (corrupted)  : {skipped} frames")

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
# 1. Save feature matrix as .npy
features_path = out_dir / "clip_features.npy"
np.save(features_path, all_features)
print(f"Saved features  -> {features_path}")

# 2. Save metadata as CSV (rows align 1-to-1 with feature matrix rows)
metadata_path = out_dir / "metadata.csv"
df = pd.DataFrame(valid_records)
df["feature_idx"] = range(len(df))
df.to_csv(metadata_path, index=False)
print(f"Saved metadata  -> {metadata_path}")

# 3. Save everything together as a compressed .npz archive
npz_path = out_dir / "clip_features.npz"
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

print(f"\nDone. {total} frames encoded into {all_features.shape[1]}-dim CLIP embeddings.")
