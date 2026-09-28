"""
Batch Pipeline: Download → Extract Frames → DINOv2 Features → Cleanup
======================================================================
Downloads games from Dropbox in batches, extracts frames using the event
annotations, computes DINOv2 features, then deletes videos to free disk.

Accumulates features across all batches into a single output file.

Usage:
    python batch_pipeline.py --batch-size 10 --max-batches 5
    python batch_pipeline.py --batch-size 10 --max-batches 0   # all games
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=10,
                    help="Games per download batch")
parser.add_argument("--max-batches", type=int, default=0,
                    help="Max batches to run (0 = all)")
parser.add_argument("--start-game-num", type=int, default=31,
                    help="Game numbering continues from this")
parser.add_argument("--token-file", default="tmp_dbx/token.txt")
parser.add_argument("--queue-file", default="tmp_dbx/download_queue.json")
parser.add_argument("--progress-file", default="tmp_dbx/batch_progress.json")
parser.add_argument("--features-out", default="features/temporal_dinov2-base_all")
parser.add_argument("--frames-root", default="results_by_label_epl_2021_2022")
args = parser.parse_args()

SHARED_URL = ("https://www.dropbox.com/scl/fo/x7klm2zbd5tnw1qn4ilbs/"
              "ABe03XYpBBPtolxrsIf0lK4?rlkey=ftc9m99fzegazg5vhnpik7nmk"
              "&st=vbbm0p2c&dl=0")
LINK_PASSWORD = "sjtu-ai4sports4ever"

VIDEO_BASE = pathlib.Path("data/videos")
JSON_BASE = pathlib.Path("data/json/Soccer_Data_Json")
SCRIPT = pathlib.Path("preprocess/extract_clips_frames_label.py")
OUTPUT_BASE = pathlib.Path(args.frames_root)
FEATURES_OUT = pathlib.Path(args.features_out)
FEATURES_OUT.mkdir(parents=True, exist_ok=True)

token = pathlib.Path(args.token_file).read_text().strip()
queue = json.load(open(args.queue_file))
session = requests.Session()

# Load progress
progress_file = pathlib.Path(args.progress_file)
if progress_file.exists():
    progress = json.load(open(progress_file))
else:
    progress = {"completed_games": [], "current_game_num": args.start_game_num}


def save_progress():
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


# Filter out already-completed games
completed = set(tuple(g) for g in progress["completed_games"])
remaining = [(l, g) for l, g in queue if (l, g) not in completed]
print(f"Queue: {len(queue)} total, {len(completed)} done, {len(remaining)} remaining")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def list_folder(path):
    r = session.post(
        "https://api.dropboxapi.com/2/files/list_folder",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        json={"path": path,
              "shared_link": {"url": SHARED_URL, "password": LINK_PASSWORD}},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["entries"]


def download_file(remote_path, local_path):
    tmp = local_path.with_suffix(local_path.suffix + ".part")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with session.post(
        "https://content.dropboxapi.com/2/sharing/get_shared_link_file",
        headers={
            "Authorization": f"Bearer {token}",
            "Dropbox-API-Arg": json.dumps(
                {"url": SHARED_URL, "path": remote_path,
                 "link_password": LINK_PASSWORD}),
        },
        stream=True, timeout=120,
    ) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        done = 0
        t0 = time.time()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
        speed = done / max(time.time() - t0, 1e-6) / 1e6
        tmp.rename(local_path)
        print(f"    {local_path.name}  {done/1e9:.2f} GB  ({speed:.1f} MB/s)")


def download_game(league, game_name):
    """Download all .mkv files for a game."""
    out_dir = VIDEO_BASE / game_name
    out_dir.mkdir(parents=True, exist_ok=True)

    remote_path = f"/{league}/{game_name}"
    try:
        entries = list_folder(remote_path)
    except Exception as e:
        print(f"    LIST FAILED: {e}")
        return False

    mkv_files = [e for e in entries if e[".tag"] == "file" and e["name"].endswith(".mkv")]
    if not mkv_files:
        print(f"    No .mkv files found")
        return False

    for e in mkv_files:
        local = out_dir / e["name"]
        if local.exists() and local.stat().st_size == e["size"]:
            print(f"    {e['name']}  already exists, skipping")
            continue
        remote = f"/{league}/{game_name}/{e['name']}"
        for attempt in range(3):
            try:
                download_file(remote, local)
                break
            except Exception as ex:
                print(f"    attempt {attempt+1} failed: {ex}")
                time.sleep(5)
        else:
            print(f"    GIVING UP on {e['name']}")
            return False
    return True


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def find_json_file(game_name, league):
    """Find JSON annotation file for a game."""
    for split in ["train", "test", "valid"]:
        json_folder = JSON_BASE / split / league / game_name
        if json_folder.exists():
            jsons = list(json_folder.glob("*.json"))
            if jsons:
                return jsons[0]
    return None


def extract_frames(game_name, league, game_num):
    """Extract event-aligned frames from video."""
    json_file = find_json_file(game_name, league)
    if not json_file:
        print(f"    No JSON annotation found")
        return False

    video_folder = VIDEO_BASE / game_name
    script_content = SCRIPT.read_text(encoding="utf-8")

    ok_all = True
    for half_num in [1, 2]:
        half_name = f"half_{half_num}"
        output_dir = OUTPUT_BASE / f"game_{game_num:02d}_{half_name}"

        if (output_dir / "frames").exists():
            frame_count = sum(1 for _ in output_dir.rglob("*.jpg")) + sum(1 for _ in output_dir.rglob("*.png"))
            if frame_count > 0:
                print(f"    [{half_name}] already extracted ({frame_count} frames), skipping")
                continue

        video_files = sorted(video_folder.glob(f"*_{half_num}.mkv"))
        if not video_files:
            print(f"    [{half_name}] no video file found")
            ok_all = False
            continue

        video_path = video_files[0]
        print(f"    [{half_name}] extracting from {video_path.name} ...")

        mod = script_content
        mod = mod.replace(
            'video_path = "/Users/yuntingyin/Documents/Research/Soccer/watford-fc-liverpool-fc_1.mkv"',
            f'video_path = r"{str(video_path.absolute())}"')
        mod = mod.replace(
            'json_path  = "/Users/yuntingyin/Documents/Research/Soccer/2017-08-12_Watford_3-3_Liverpool_AaZvBO5T.json"',
            f'json_path = r"{str(json_file.absolute())}"')
        mod = mod.replace(
            'base_out   = pathlib.Path("extracts_first_half")',
            f'base_out = pathlib.Path(r"{str(output_dir.absolute())}")')
        # Frames-only: skip video writing
        mod = mod.replace("        vout.write(frame)\n", "\n")
        # JPEG instead of PNG
        mod = mod.replace(
            'out_png = frames_dir / f"{event_dir}_frame_{tag_ts}_t{label}.png"\n'
            "            cv2.imwrite(str(out_png), frame)",
            'out_png = frames_dir / f"{event_dir}_frame_{tag_ts}_t{label}.jpg"\n'
            "            cv2.imwrite(str(out_png), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])")

        temp_script = pathlib.Path(f"temp_extract_{game_num:02d}_{half_num}.py")
        temp_script.write_text(mod, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, str(temp_script)],
                capture_output=True, text=True, timeout=2400,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            if result.returncode == 0:
                print(f"    [{half_name}] done")
            else:
                print(f"    [{half_name}] FAILED: {result.stderr[:300]}")
                ok_all = False
        except subprocess.TimeoutExpired:
            print(f"    [{half_name}] TIMED OUT")
            ok_all = False
        finally:
            temp_script.unlink(missing_ok=True)
            clips_dir = output_dir / "clips"
            if clips_dir.exists():
                shutil.rmtree(clips_dir, ignore_errors=True)

    return ok_all


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------
total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
if args.max_batches > 0:
    total_batches = min(total_batches, args.max_batches)

game_num = progress["current_game_num"]
batch_idx = 0

for batch_start in range(0, len(remaining), args.batch_size):
    if args.max_batches > 0 and batch_idx >= args.max_batches:
        break

    batch = remaining[batch_start:batch_start + args.batch_size]
    batch_idx += 1
    print(f"\n{'='*70}")
    print(f"BATCH {batch_idx}/{total_batches}  ({len(batch)} games)")
    print(f"{'='*70}")

    batch_games = []  # (league, game_name, game_num) for successful downloads

    # Step 1: Download batch
    for i, (league, game_name) in enumerate(batch):
        gn = game_num + i
        print(f"\n  [{i+1}/{len(batch)}] game_{gn:02d}: {league}/{game_name}")
        print(f"  Downloading ...")
        ok = download_game(league, game_name)
        if ok:
            batch_games.append((league, game_name, gn))
        else:
            print(f"  DOWNLOAD FAILED, skipping")

    # Step 2: Extract frames
    print(f"\n--- Extracting frames for {len(batch_games)} games ---")
    for league, game_name, gn in batch_games:
        print(f"\n  game_{gn:02d}: {game_name}")
        extract_frames(game_name, league, gn)

    # Step 3: Delete video files to free disk
    print(f"\n--- Cleaning up video files ---")
    freed = 0
    for league, game_name, gn in batch_games:
        video_dir = VIDEO_BASE / game_name
        if video_dir.exists():
            size = sum(f.stat().st_size for f in video_dir.rglob("*") if f.is_file())
            shutil.rmtree(video_dir, ignore_errors=True)
            freed += size
            print(f"  Deleted {game_name} ({size/1e9:.1f} GB)")
    print(f"  Total freed: {freed/1e9:.1f} GB")

    # Step 4: Mark games as completed
    for league, game_name, gn in batch_games:
        progress["completed_games"].append([league, game_name])
    game_num += len(batch)
    progress["current_game_num"] = game_num
    save_progress()
    print(f"\n  Progress saved. {len(progress['completed_games'])} games completed total.")

print(f"\n{'='*70}")
print(f"ALL BATCHES COMPLETE")
print(f"{'='*70}")
print(f"Total games processed: {len(progress['completed_games'])}")
print(f"Frames saved to: {OUTPUT_BASE}")
print(f"\nNext step: run temporal feature extraction on all frames:")
print(f"  python temporal_clip_features.py --model facebook/dinov2-base --out {FEATURES_OUT}")
