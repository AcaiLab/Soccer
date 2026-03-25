"""
clip_features.py
----------------
Extracts CLIP image embeddings from all frames in the extracts_output folder,
then produces scatter plots coloured by:
  1. Event label      (colour families: attack/set-piece/defence/discipline/admin/other)
  2. Half             (first half vs second half)
  3. 10-minute segment (0-10, 10-20, ... min, parsed from filename)

Also produces:
  - CSV of all embeddings + metadata
  - Per-label intra-class cosine similarity summary
  - Frames per label bar chart

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
import matplotlib.patches as mpatches

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

# ── colour families ───────────────────────────────────────────────────────────
EVENT_FAMILIES = {
    # Attack — reds
    "goal":                  "attack",
    "shot_off_target":       "attack",
    "saved_by_goal-keeper":  "attack",
    "saved_by_goalkeeper":   "attack",
    "lead_to_corner":        "attack",
    # Set pieces — blues
    "corner":                "set_piece",
    "free_kick":             "set_piece",
    "penalty":               "set_piece",
    "off_side":              "set_piece",
    # Defence — greens
    "clearance":             "defence",
    "ball_possession":       "defence",
    # Discipline — ambers
    "yellow_card":           "discipline",
    "foul_with_no_card":     "discipline",
    "foul_lead_to_penalty":  "discipline",
    # Admin — purples
    "substitution":          "admin",
    "injury":                "admin",
    "start_of_half_game":    "admin",
    "end_of_half_game":      "admin",
    "show_added_time":       "admin",
}

FAMILY_PALETTES = {
    "attack":    ["#A32D2D", "#E24B4A", "#F09595", "#F7C1C1", "#FAECE7"],
    "set_piece": ["#185FA5", "#378ADD", "#85B7EB", "#B5D4F4", "#E6F1FB"],
    "defence":   ["#3B6D11", "#639922", "#97C459", "#C0DD97", "#EAF3DE"],
    "discipline":["#854F0B", "#BA7517", "#EF9F27", "#FAC775", "#FAEEDA"],
    "admin":     ["#3C3489", "#7F77DD", "#AFA9EC", "#CECBF6", "#EEEDFE"],
    "other":     ["#5F5E5A", "#888780", "#B4B2A9", "#D3D1C7", "#F1EFE8"],
}

_family_event_index: dict = defaultdict(int)
EVENT_COLOURS: dict = {}

def get_event_colour(event_label: str) -> str:
    if event_label in EVENT_COLOURS:
        return EVENT_COLOURS[event_label]
    family = EVENT_FAMILIES.get(event_label, "other")
    palette = FAMILY_PALETTES[family]
    idx = _family_event_index[family] % len(palette)
    _family_event_index[family] += 1
    colour = palette[idx]
    EVENT_COLOURS[event_label] = colour
    return colour

HALF_COLOURS = {"first_half": "#378ADD", "second_half": "#E24B4A", "unknown": "#888780"}

SEGMENT_COLOURS = {
    "0-10":  "#3C3489", "10-20": "#7F77DD", "20-30": "#378ADD",
    "30-40": "#639922", "40-50": "#BA7517", "50-60": "#E24B4A",
    "60-70": "#A32D2D", "70-80": "#854F0B", "80-90": "#5F5E5A",
    "90+":   "#888780",
}

def minute_to_segment(minute: int) -> str:
    for low in range(0, 90, 10):
        if minute < low + 10:
            return f"{low}-{low+10}"
    return "90+"

# ── metadata helpers ──────────────────────────────────────────────────────────
def get_event_label_from_path(p: pathlib.Path) -> str:
    return p.parent.name

def get_half_from_path(p: pathlib.Path) -> str:
    for part in p.parts:
        if part in ("first_half", "second_half"):
            return part
    return "unknown"

def get_match_from_path(p: pathlib.Path) -> str:
    parts = p.parts
    for i, part in enumerate(parts):
        if part in ("first_half", "second_half") and i > 0:
            return parts[i - 1]
    return "unknown"

def get_minute_from_filename(p: pathlib.Path) -> int:
    m = re.search(r'frame_(\d+)-[\d.]+_', p.name)
    return int(m.group(1)) if m else -1

# ── load CLIP ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP (ViT-B/32) on {device} ...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("CLIP loaded.\n")

# ── collect frames ────────────────────────────────────────────────────────────
png_files = sorted(extracts_root.rglob("frames/**/*.png"))
print(f"Found {len(png_files)} PNG frames to process.\n")
if not png_files:
    print("ERROR: No PNG frames found. Run extract_clips.py first.")
    exit(1)

# ── extract embeddings ────────────────────────────────────────────────────────
BATCH_SIZE = 64
records = []

for batch_start in range(0, len(png_files), BATCH_SIZE):
    batch_paths = png_files[batch_start: batch_start + BATCH_SIZE]
    images, valid_paths = [], []

    for p in batch_paths:
        try:
            images.append(preprocess(Image.open(p).convert("RGB")))
            valid_paths.append(p)
        except Exception as e:
            print(f"  Warning: could not load {p.name}: {e}")

    if not images:
        continue

    image_tensor = torch.stack(images).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    emb_np = features.cpu().numpy()
    for i, path in enumerate(valid_paths):
        minute = get_minute_from_filename(path)
        records.append({
            "path":        str(path),
            "match":       get_match_from_path(path),
            "half":        get_half_from_path(path),
            "event_label": get_event_label_from_path(path),
            "minute":      minute,
            "segment":     minute_to_segment(minute) if minute >= 0 else "unknown",
            "embedding":   emb_np[i],
        })

    print(f"  Embedded {min(batch_start + BATCH_SIZE, len(png_files))}/{len(png_files)} frames...")

print(f"\nTotal embeddings: {len(records)}")

# ── save ──────────────────────────────────────────────────────────────────────
emb_matrix = np.stack([r["embedding"] for r in records])
meta_df    = pd.DataFrame([{k: v for k, v in r.items() if k != "embedding"} for r in records])

np.save(str(out_dir / "clip_embeddings.npy"), emb_matrix)
meta_df.to_csv(str(out_dir / "clip_metadata.csv"), index=False)
print(f"Embeddings → {out_dir / 'clip_embeddings.npy'}")
print(f"Metadata   → {out_dir / 'clip_metadata.csv'}")

# ── dimensionality reduction ──────────────────────────────────────────────────
from sklearn.manifold import TSNE
print("\nRunning t-SNE ...")
coords_tsne = TSNE(n_components=2, perplexity=min(30, len(emb_matrix) - 1),
                   random_state=42, n_iter=1000).fit_transform(emb_matrix)

coords_umap = None
try:
    import umap
    print("Running UMAP ...")
    coords_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(emb_matrix)
except ImportError:
    print("umap-learn not installed — skipping UMAP.")

# ── scatter plot helper ───────────────────────────────────────────────────────
def make_scatter(coords, labels_list, colour_fn, legend_items, title, filename):
    fig, ax = plt.subplots(figsize=(13, 8))
    for lbl in sorted(set(labels_list)):
        idx = [i for i, l in enumerate(labels_list) if l == lbl]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   color=colour_fn(lbl), s=18, alpha=0.80, linewidths=0)
    handles = [mpatches.Patch(color=c, label=l) for l, c in legend_items]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, frameon=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    plt.tight_layout()
    fig.savefig(str(out_dir / filename), dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out_dir / filename}")

# ── plot 1: by event label ────────────────────────────────────────────────────
event_labels  = meta_df["event_label"].tolist()
unique_events = sorted(set(event_labels))
for e in unique_events:
    get_event_colour(e)
event_legend = [(e, EVENT_COLOURS[e]) for e in unique_events]

for method, coords in [("tsne", coords_tsne)] + ([("umap", coords_umap)] if coords_umap is not None else []):
    make_scatter(coords, event_labels, get_event_colour, event_legend,
                 f"CLIP embeddings — {method.upper()} coloured by event label",
                 f"clip_{method}_by_event.png")

# ── plot 2: by half ───────────────────────────────────────────────────────────
half_labels = meta_df["half"].tolist()
half_legend = [(h, HALF_COLOURS.get(h, "#888780")) for h in sorted(set(half_labels))]

for method, coords in [("tsne", coords_tsne)] + ([("umap", coords_umap)] if coords_umap is not None else []):
    make_scatter(coords, half_labels,
                 lambda h: HALF_COLOURS.get(h, "#888780"),
                 half_legend,
                 f"CLIP embeddings — {method.upper()} coloured by half",
                 f"clip_{method}_by_half.png")

# ── plot 3: by 10-min segment ─────────────────────────────────────────────────
segment_labels = meta_df["segment"].tolist()
seg_order      = ["0-10","10-20","20-30","30-40","40-50",
                  "50-60","60-70","70-80","80-90","90+","unknown"]
present_segs   = [s for s in seg_order if s in set(segment_labels)]
seg_legend     = [(s, SEGMENT_COLOURS.get(s, "#888780")) for s in present_segs]

for method, coords in [("tsne", coords_tsne)] + ([("umap", coords_umap)] if coords_umap is not None else []):
    make_scatter(coords, segment_labels,
                 lambda s: SEGMENT_COLOURS.get(s, "#888780"),
                 seg_legend,
                 f"CLIP embeddings — {method.upper()} coloured by 10-min segment",
                 f"clip_{method}_by_segment.png")

# ── cosine similarity ─────────────────────────────────────────────────────────
print("\nPer-label intra-class cosine similarity:")
print(f"  {'Event label':<35} {'Avg cosine sim':>15}  {'Frames':>6}")
print("  " + "-"*60)

label_to_idxs = defaultdict(list)
for i, lbl in enumerate(event_labels):
    label_to_idxs[lbl].append(i)

sim_rows = []
for lbl in sorted(label_to_idxs.keys()):
    idxs = label_to_idxs[lbl]
    vecs = emb_matrix[idxs]
    avg_sim = 1.0
    if len(vecs) >= 2:
        n = len(vecs)
        avg_sim = (vecs @ vecs.T).sum() - n
        avg_sim /= n * (n - 1)
    sim_rows.append({"event_label": lbl, "avg_cosine_sim": round(float(avg_sim), 4),
                     "n_frames": len(idxs)})
    print(f"  {lbl:<35} {avg_sim:>15.4f}  {len(idxs):>6}")

pd.DataFrame(sim_rows).sort_values("avg_cosine_sim", ascending=False)\
  .to_csv(str(out_dir / "label_cosine_similarity.csv"), index=False)

# ── frames per label bar chart ────────────────────────────────────────────────
frame_counts = meta_df["event_label"].value_counts().sort_index()
bar_colours  = [get_event_colour(lbl) for lbl in frame_counts.index]

fig, ax = plt.subplots(figsize=(max(10, len(frame_counts) * 0.7), 5))
bars = ax.bar(frame_counts.index, frame_counts.values,
              color=bar_colours, edgecolor="white", linewidth=0.6)
ax.set_title("Frames per event label", fontsize=13, fontweight="bold")
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

print(f"\n{'='*60}")
print("CLIP feature extraction complete!")
print(f"All outputs saved to: {out_dir}")
print(f"{'='*60}")
