"""
Fast Batch Pipeline: Continuous download → extract → cleanup cycle.
Processes games serially but skips already-done work and saves progress.

Usage:
    python fast_batch_pipeline.py --max-games 50
    python fast_batch_pipeline.py --max-games 0   # all games
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--max-games", type=int, default=0, help="0 = all")
parser.add_argument("--start-game-num", type=int, default=31)
parser.add_argument("--token-file", default="tmp_dbx/token.txt")
parser.add_argument("--queue-file", default="tmp_dbx/download_queue.json")
parser.add_argument("--progress-file", default="tmp_dbx/fast_progress.json")
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

token = pathlib.Path(args.token_file).read_text().strip()
queue = json.load(open(args.queue_file))
session = requests.Session()

# Progress
progress_file = pathlib.Path(args.progress_file)
if progress_file.exists():
    progress = json.load(open(progress_file))
else:
    progress = {"done": [], "failed": [], "game_num": args.start_game_num}


def save_progress():
    with open(progress_file, "w") as f:
        json.dump(progress, f)


done_set = set(tuple(x) for x in progress["done"])
failed_set = set(tuple(x) for x in progress["failed"])
remaining = [(l, g) for l, g in queue if (l, g) not in done_set and (l, g) not in failed_set]

if args.max_games > 0:
    remaining = remaining[:args.max_games]

print(f"Total queue: {len(queue)}, Done: {len(done_set)}, Failed: {len(failed_set)}, "
      f"Remaining: {len(remaining)}")
print(flush=True)


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
        return done, speed


def find_json_file(game_name, league):
    for split in ["train", "test", "valid"]:
        json_folder = JSON_BASE / split / league / game_name
        if json_folder.exists():
            jsons = list(json_folder.glob("*.json"))
            if jsons:
                return jsons[0]
    return None


def extract_half(video_path, json_path, half_num, game_num):
    output_dir = OUTPUT_BASE / f"game_{game_num:02d}_half_{half_num}"
    if (output_dir / "frames").exists():
        n = sum(1 for _ in output_dir.rglob("*.jpg")) + sum(1 for _ in output_dir.rglob("*.png"))
        if n > 0:
            return True  # already done

    script_content = SCRIPT.read_text(encoding="utf-8")
    script_content = script_content.replace(
        'video_path = "/Users/yuntingyin/Documents/Research/Soccer/watford-fc-liverpool-fc_1.mkv"',
        f'video_path = r"{str(video_path.absolute())}"')
    script_content = script_content.replace(
        'json_path  = "/Users/yuntingyin/Documents/Research/Soccer/2017-08-12_Watford_3-3_Liverpool_AaZvBO5T.json"',
        f'json_path = r"{str(json_path.absolute())}"')
    script_content = script_content.replace(
        'base_out   = pathlib.Path("extracts_first_half")',
        f'base_out = pathlib.Path(r"{str(output_dir.absolute())}")')
    script_content = script_content.replace("        vout.write(frame)\n", "\n")
    script_content = script_content.replace(
        'out_png = frames_dir / f"{event_dir}_frame_{tag_ts}_t{label}.png"\n'
        "            cv2.imwrite(str(out_png), frame)",
        'out_png = frames_dir / f"{event_dir}_frame_{tag_ts}_t{label}.jpg"\n'
        "            cv2.imwrite(str(out_png), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])")

    temp_script = pathlib.Path(f"temp_extract_{game_num:04d}_{half_num}.py")
    temp_script.write_text(script_content, encoding="utf-8")
    ok = False
    try:
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True, text=True, timeout=1800,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        ok = result.returncode == 0
        if not ok:
            print(f"      EXTRACT FAIL: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"      EXTRACT TIMEOUT")
    finally:
        temp_script.unlink(missing_ok=True)
        clips_dir = output_dir / "clips"
        if clips_dir.exists():
            shutil.rmtree(clips_dir, ignore_errors=True)
    return ok


game_num = progress["game_num"]
t_start = time.time()

for i, (league, game_name) in enumerate(remaining):
    elapsed = time.time() - t_start
    rate = (i / elapsed * 3600) if elapsed > 0 and i > 0 else 0
    print(f"\n[{i+1}/{len(remaining)}] game_{game_num:04d}: {game_name}  "
          f"({rate:.0f} games/hr)", flush=True)

    # Step 1: Download
    out_dir = VIDEO_BASE / game_name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        entries = list_folder(f"/{league}/{game_name}")
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "Unauthorized" in err_str or "expired" in err_str.lower():
            print(f"\n*** TOKEN EXPIRED after {i} games ***", flush=True)
            save_progress()
            sys.exit(1)
        if "NameResolutionError" in err_str or "getaddrinfo failed" in err_str or "ConnectionError" in err_str:
            print(f"\n*** NETWORK ERROR after {i} games ***", flush=True)
            save_progress()
            sys.exit(1)
        print(f"  LIST FAILED: {e}", flush=True)
        progress["failed"].append([league, game_name])
        game_num += 1
        progress["game_num"] = game_num
        save_progress()
        continue

    mkv_files = [e for e in entries if e[".tag"] == "file" and e["name"].endswith(".mkv")]
    if not mkv_files:
        print(f"  No .mkv files", flush=True)
        progress["failed"].append([league, game_name])
        game_num += 1
        progress["game_num"] = game_num
        save_progress()
        continue

    dl_ok = True
    for e in mkv_files:
        local = out_dir / e["name"]
        if local.exists() and local.stat().st_size == e["size"]:
            continue
        remote = f"/{league}/{game_name}/{e['name']}"
        for attempt in range(3):
            try:
                sz, spd = download_file(remote, local)
                print(f"  DL {e['name']}  {sz/1e9:.1f}GB  {spd:.0f}MB/s", flush=True)
                break
            except Exception as ex:
                if "invalid_access_token" in str(ex):
                    print(f"\n*** TOKEN EXPIRED after {i} games ***", flush=True)
                    save_progress()
                    sys.exit(1)
                print(f"  attempt {attempt+1} failed: {str(ex)[:100]}", flush=True)
                time.sleep(3)
        else:
            dl_ok = False
            break

    if not dl_ok:
        print(f"  DOWNLOAD FAILED", flush=True)
        progress["failed"].append([league, game_name])
        shutil.rmtree(out_dir, ignore_errors=True)
        game_num += 1
        progress["game_num"] = game_num
        save_progress()
        continue

    # Step 2: Extract frames
    json_file = find_json_file(game_name, league)
    if not json_file:
        print(f"  NO JSON ANNOTATION", flush=True)
        progress["failed"].append([league, game_name])
        shutil.rmtree(out_dir, ignore_errors=True)
        game_num += 1
        progress["game_num"] = game_num
        save_progress()
        continue

    for half_num in [1, 2]:
        video_files = sorted(out_dir.glob(f"*_{half_num}.mkv"))
        if video_files:
            ok = extract_half(video_files[0], json_file, half_num, game_num)
            if ok:
                print(f"  H{half_num} extracted", flush=True)
            else:
                print(f"  H{half_num} FAILED", flush=True)

    # Step 3: Delete video immediately
    vid_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"  Cleanup: freed {vid_size/1e9:.1f} GB", flush=True)

    progress["done"].append([league, game_name])
    game_num += 1
    progress["game_num"] = game_num
    save_progress()

elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"DONE. Processed {len(progress['done'])} games in {elapsed/3600:.1f} hours")
print(f"Failed: {len(progress['failed'])}")
print(f"Next game_num: {game_num}")
print(f"{'='*60}")
