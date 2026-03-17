"""
clip_features.py
----------------
Extracts CLIP image embeddings from all frames in the extracts_output folder,
then produces:
  1. A summary of cosine-similarity clusters per event label
  2. A UMAP/t-SNE 2-D scatter plot coloured by event label
  3. A CSV of all embeddings + metadata (for downstream use)

Requirements:
    pip install torch torchvision ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
    pip install umap-learn scikit-learn matplotlib pandas

Usage:
    python clip_features.py --extracts-folder extracts_output --out clip_output
"""

import pathlib
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Extract CLIP embeddings from event frames.")
parser.add_argument("--extracts-folder", required=True,
                    help="Root of the extracts_output folder produced by extract_clips.py")
parser.add_argument("--out", default="clip_output",
                    help="Output folder for embeddings, CSV, and plots")
args = parser.parse_args()

extracts_root = pathlib.Path(args.extracts_folder)
out_dir       = pathlib.Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# ── load CLIP model ───────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP (ViT-B/32) on {device} ...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("CLIP loaded.\n")

# ── collect all PNG frames ────────────────────────────────────────────────────
# Expected path structure:
#   extracts_root/
#     <match>/
#       first_half|second_half/
#         frames/
#           <event_label>/
#             <event_label>_frame_*.png
records = []
png_files = sorted(extracts_root.rglob("frames/**/*.png"))
print(f"Found {len(png_files)} PNG frames to process.\n")

if not png_files:
    print("ERROR: No PNG frames found. Run extract_clips.py first.")
    exit(1)

# ── extract embeddings in batches ─────────────────────────────────────────────
BATCH_SIZE = 64

def get_event_label_from_path(png_path: pathlib.Path) -> str:
    """
    The event label is the name of the directory that directly contains the PNG
    (i.e. the event-named subdirectory under frames/).
    """
    return png_path.parent.name

def get_match_name_from_path(png_path: pathlib.Path) -> str:
    """Walk up the path to find the match folder name (2 levels above 'first_half'/'second_half')."""
    parts = png_path.parts
    for i, p in enumerate(parts):
        if p in ("first_half", "second_half"):
            return parts[i - 1] if i > 0 else "unknown"
    return "unknown"

all_embeddings = []

for batch_start in range(0, len(png_files), BATCH_SIZE):
    batch_paths = png_files[batch_start: batch_start + BATCH_SIZE]
    images = []
    valid_paths = []

    for p in batch_paths:
        try:
            img = preprocess(Image.open(p).convert("RGB"))
            images.append(img)
            valid_paths.append(p)
        except Exception as e:
            print(f"  Warning: could not load {p.name}: {e}")

    if not images:
        continue

    image_tensor = torch.stack(images).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalise

    emb_np = features.cpu().numpy()

    for i, path in enumerate(valid_paths):
        records.append({
            "path":        str(path),
            "match":       get_match_name_from_path(path),
            "event_label": get_event_label_from_path(path),
            "embedding":   emb_np[i],
        })

    processed = min(batch_start + BATCH_SIZE, len(png_files))
    print(f"  Embedded {processed}/{len(png_files)} frames...")

print(f"\nTotal embeddings collected: {len(records)}")

# ── save embeddings to CSV ────────────────────────────────────────────────────
emb_matrix = np.stack([r["embedding"] for r in records])   # (N, 512)
meta_df    = pd.DataFrame([{k: v for k, v in r.items() if k != "embedding"} for r in records])

np.save(str(out_dir / "clip_embeddings.npy"), emb_matrix)
meta_df.to_csv(str(out_dir / "clip_metadata.csv"), index=False)
print(f"Embeddings saved → {out_dir / 'clip_embeddings.npy'}")
print(f"Metadata  saved  → {out_dir / 'clip_metadata.csv'}")

# ── dimensionality reduction + scatter plot ───────────────────────────────────
labels      = meta_df["event_label"].tolist()
unique_lbls = sorted(set(labels))
color_map   = {lbl: cm.tab20(i / max(len(unique_lbls) - 1, 1))
               for i, lbl in enumerate(unique_lbls)}

def make_scatter(coords_2d: np.ndarray, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(12, 8))
    for lbl in unique_lbls:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax.scatter(coords_2d[idx, 0], coords_2d[idx, 1],
                   label=lbl, color=color_map[lbl], s=18, alpha=0.75)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, markerscale=1.5)
    plt.tight_layout()
    path = out_dir / filename
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")

# t-SNE (always available via scikit-learn)
from sklearn.manifold import TSNE
print("\nRunning t-SNE (this may take a minute) ...")
tsne   = TSNE(n_components=2, perplexity=min(30, len(records) - 1),
              random_state=42, n_iter=1000)
coords = tsne.fit_transform(emb_matrix)
make_scatter(coords, "CLIP embeddings — t-SNE (coloured by event label)", "clip_tsne.png")

# UMAP (optional, only if installed)
try:
    import umap
    print("Running UMAP ...")
    reducer    = umap.UMAP(n_components=2, random_state=42)
    coords_umap = reducer.fit_transform(emb_matrix)
    make_scatter(coords_umap, "CLIP embeddings — UMAP (coloured by event label)", "clip_umap.png")
except ImportError:
    print("umap-learn not installed — skipping UMAP plot (pip install umap-learn to enable).")

# ── per-label average cosine similarity ──────────────────────────────────────
print("\nPer-label intra-class cosine similarity (higher = more visually consistent):")
print(f"  {'Event label':<35} {'Avg cosine sim':>15}  {'Frames':>6}")
print("  " + "-"*60)

label_to_idxs = defaultdict(list)
for i, lbl in enumerate(labels):
    label_to_idxs[lbl].append(i)

similarity_rows = []
for lbl in sorted(label_to_idxs.keys()):
    idxs = label_to_idxs[lbl]
    vecs = emb_matrix[idxs]           # (k, 512) — already L2-normalised
    if len(vecs) < 2:
        avg_sim = 1.0
    else:
        sim_matrix = vecs @ vecs.T     # cosine similarities
        # mean of off-diagonal elements
        n = len(vecs)
        avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
    similarity_rows.append({"event_label": lbl, "avg_cosine_sim": round(float(avg_sim), 4),
                             "n_frames": len(idxs)})
    print(f"  {lbl:<35} {avg_sim:>15.4f}  {len(idxs):>6}")

sim_df = pd.DataFrame(similarity_rows).sort_values("avg_cosine_sim", ascending=False)
sim_df.to_csv(str(out_dir / "label_cosine_similarity.csv"), index=False)
print(f"\nSimilarity summary saved → {out_dir / 'label_cosine_similarity.csv'}")

# ── bar chart: frames per label ───────────────────────────────────────────────
frame_counts = meta_df["event_label"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(max(10, len(frame_counts) * 0.7), 5))
bars = ax.bar(frame_counts.index, frame_counts.values, color="#4C72B0",
              edgecolor="white", linewidth=0.6)
ax.set_title("Frames per event label (CLIP dataset)", fontsize=13, fontweight="bold")
ax.set_xlabel("Event label"); ax.set_ylabel("# Frames")
ax.set_xticks(range(len(frame_counts)))
ax.set_xticklabels(frame_counts.index, rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
for bar, val in zip(bars, frame_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=7)
plt.tight_layout()
fig.savefig(str(out_dir / "frames_per_label.png"), dpi=150)
plt.close(fig)
print(f"Frame-count bar chart saved → {out_dir / 'frames_per_label.png'}")

print(f"\n{'='*60}")
print(f"CLIP feature extraction complete!")
print(f"All outputs saved to: {out_dir}")
print(f"{'='*60}")
