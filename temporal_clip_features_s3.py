"""
Temporal Feature Extraction — S3 Streaming Version
====================================================
Downloads one game_half directory at a time from S3, extracts DINOv2 features,
saves them incrementally as .npz parts, deletes local frames, moves on.

Disk usage stays under ~2GB. Progress survives Colab runtime resets.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

FRAME_RE = re.compile(r'_frame_(\d{2}-\d{2}\.\d{3})_t(\d{2}-\d{2}\.\d{3})\.(?:png|jpg)$')


def tag_to_sec(tag: str) -> float:
    mm, ss = tag.split("-", 1)
    return int(mm) * 60.0 + float(ss)


parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True)
parser.add_argument("--prefix", default="frames/")
parser.add_argument("--out", default="features/temporal_dinov2-base_all")
parser.add_argument("--model", default="facebook/dinov2-base")
parser.add_argument("--max-frames", type=int, default=60)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--device", default=None)
parser.add_argument("--tmp-dir", default="/content/_tmp_frames")
args = parser.parse_args()

out_dir = pathlib.Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
parts_dir = out_dir / "parts"
parts_dir.mkdir(parents=True, exist_ok=True)
tmp_dir = pathlib.Path(args.tmp_dir)
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
max_frames = args.max_frames
embed_dim = 768

progress_file = out_dir / "s3_progress.json"
if progress_file.exists():
    progress = json.load(open(progress_file))
else:
    progress = {"done": []}

done_set = set(progress["done"])

# List game_half dirs in S3
print("Listing game directories in S3...")
s3_prefix = f"s3://{args.bucket}/{args.prefix}"
result = subprocess.run(["aws", "s3", "ls", s3_prefix], capture_output=True, text=True)
if result.returncode != 0:
    print(f"ERROR listing S3: {result.stderr}")
    sys.exit(1)

game_dirs = []
for line in result.stdout.strip().split("\n"):
    line = line.strip()
    if line.startswith("PRE "):
        dirname = line[4:].rstrip("/")
        if dirname.startswith("game_"):
            game_dirs.append(dirname)

game_dirs.sort()
remaining = [d for d in game_dirs if d not in done_set]
print(f"Total game_half dirs: {len(game_dirs)}, Done: {len(done_set)}, Remaining: {len(remaining)}")

if not remaining:
    print("All done! Assembling final output...")
else:
    print(f"\nLoading encoder: {args.model} on {device} ...")
    from model_encoders import build_encoder
    encoder = build_encoder(args.model, device)
    embed_dim = encoder.embed_dim
    print(f"Encoder loaded. embed_dim={embed_dim}\n")

    def load_safe(path):
        try:
            img = Image.open(path)
            img.load()
            return img.convert("RGB")
        except Exception:
            return None

    def scan_events(game_half_dir):
        events = defaultdict(list)
        frames_dir = game_half_dir / "frames"
        if not frames_dir.exists():
            return {}
        parts = game_half_dir.name.split("_")
        try:
            game_id, half_id = parts[1], parts[3]
        except IndexError:
            game_id, half_id = game_half_dir.name, "?"
        for label_dir in sorted(frames_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for img_path in sorted(list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpg"))):
                m = FRAME_RE.search(img_path.name)
                if not m:
                    continue
                event_tag, frame_tag = m.group(1), m.group(2)
                events[(game_id, half_id, label, event_tag)].append((tag_to_sec(frame_tag), str(img_path)))
        return events

    def extract_features(events_dict):
        event_keys = sorted(events_dict.keys())
        if not event_keys:
            return None

        event_frame_lists = []
        for key in event_keys:
            frames = sorted(events_dict[key], key=lambda x: x[0])
            if len(frames) > max_frames:
                idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                frames = [frames[i] for i in idxs]
            event_frame_lists.append(frames)

        flat_paths, flat_event_idx = [], []
        for i, frames in enumerate(event_frame_lists):
            for _, p in frames:
                flat_paths.append(p)
                flat_event_idx.append(i)

        n_total = len(flat_paths)
        all_feats = np.zeros((n_total, embed_dim), dtype=np.float32)
        valid_mask = np.zeros(n_total, dtype=bool)

        for start in range(0, n_total, args.batch_size):
            batch = flat_paths[start:start + args.batch_size]
            loaded = [(j, load_safe(p)) for j, p in enumerate(batch)]
            valid = [(j, img) for j, img in loaded if img is not None]
            if not valid:
                continue
            images = [img for _, img in valid]
            feats_np = encoder.encode_batch(images)
            for (local_j, _), feat in zip(valid, feats_np):
                all_feats[start + local_j] = feat
                valid_mask[start + local_j] = True

        flat_arr = np.array(flat_event_idx)
        n_ev = len(event_keys)
        padded = np.zeros((n_ev, max_frames, embed_dim), dtype=np.float32)
        masks = np.zeros((n_ev, max_frames), dtype=np.bool_)
        labels, games_l, halves_l, etags_l, meta = [], [], [], [], []

        for i, key in enumerate(event_keys):
            gid, hid, label, etag = key
            gidxs = np.where(flat_arr == i)[0]
            vidxs = gidxs[valid_mask[gidxs]]
            nv = len(vidxs)
            padded[i, :nv] = all_feats[vidxs]
            masks[i, :nv] = True
            labels.append(label)
            games_l.append(gid)
            halves_l.append(hid)
            etags_l.append(etag)
            meta.append({"game": gid, "half": hid, "label": label, "event_tag": etag,
                         "n_frames": len(event_frame_lists[i]), "n_valid": nv})

        return padded, masks, np.array(labels), np.array(games_l), np.array(halves_l), np.array(etags_l), meta

    # Main loop
    for idx, ghname in enumerate(remaining):
        print(f"[{idx+1}/{len(remaining)}] {ghname}", end="", flush=True)
        local_dir = tmp_dir / ghname
        local_dir.mkdir(parents=True, exist_ok=True)
        s3_path = f"s3://{args.bucket}/{args.prefix}{ghname}/"
        dl = subprocess.run(["aws", "s3", "sync", s3_path, str(local_dir), "--only-show-errors"],
                            capture_output=True, text=True, timeout=600)
        if dl.returncode != 0:
            print(f"  DOWNLOAD FAILED")
            shutil.rmtree(local_dir, ignore_errors=True)
            continue

        events_dict = scan_events(local_dir)
        n_ev = len(events_dict)
        if n_ev > 0:
            result = extract_features(events_dict)
            if result is not None:
                pad, msk, lab, gm, hv, et, mt = result
                np.savez_compressed(parts_dir / f"{ghname}.npz",
                                    features=pad, masks=msk, labels=lab,
                                    games=gm, halves=hv, event_tags=et)
                with open(parts_dir / f"{ghname}_meta.json", "w") as f:
                    json.dump(mt, f)
                print(f"  {n_ev} events", flush=True)
            else:
                print(f"  0 events", flush=True)
        else:
            print(f"  0 events", flush=True)

        shutil.rmtree(local_dir, ignore_errors=True)
        progress["done"].append(ghname)
        done_set.add(ghname)
        with open(progress_file, "w") as f:
            json.dump(progress, f)

# Assemble all parts
print(f"\n{'='*60}\nAssembling final output...")
part_files = sorted(parts_dir.glob("*.npz"))
if not part_files:
    print("No part files found!")
    sys.exit(1)

af, am, al, ag, ah, ae, ameta = [], [], [], [], [], [], []
for pf in part_files:
    d = np.load(pf, allow_pickle=True)
    af.append(d["features"]); am.append(d["masks"]); al.append(d["labels"])
    ag.append(d["games"]); ah.append(d["halves"]); ae.append(d["event_tags"])
    mf = pf.with_name(pf.stem + "_meta.json")
    if mf.exists():
        ameta.extend(json.load(open(mf)))

features = np.concatenate(af)
masks = np.concatenate(am)
npz_path = out_dir / "clip_temporal_padded.npz"
np.savez_compressed(npz_path, features=features, masks=masks,
                    labels=np.concatenate(al), games=np.concatenate(ag),
                    halves=np.concatenate(ah), event_tags=np.concatenate(ae))
print(f"Saved: {npz_path}  features: {features.shape}")

df = pd.DataFrame(ameta)
df.to_csv(out_dir / "temporal_metadata.csv", index=False)
print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")
print(f"\nDone. {features.shape[0]} events x {max_frames} frames x {features.shape[2]}-dim.")
