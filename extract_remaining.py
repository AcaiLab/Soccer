"""Extract frames from already-downloaded games 34-40, then clean up."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

queue = json.load(open("tmp_dbx/download_queue.json"))
JSON_BASE = pathlib.Path("data/json/Soccer_Data_Json")
VIDEO_BASE = pathlib.Path("data/videos")
SCRIPT = pathlib.Path("preprocess/extract_clips_frames_label.py")
OUTPUT_BASE = pathlib.Path("results_by_label_epl_2021_2022")


def find_json_file(game_name, league):
    for split in ["train", "test", "valid"]:
        jf = JSON_BASE / split / league / game_name
        if jf.exists():
            jsons = list(jf.glob("*.json"))
            if jsons:
                return jsons[0]
    return None


def extract_half(video_path, json_path, half_num, game_num):
    output_dir = OUTPUT_BASE / f"game_{game_num:02d}_half_{half_num}"
    if (output_dir / "frames").exists():
        n = sum(1 for _ in output_dir.rglob("*.jpg")) + sum(1 for _ in output_dir.rglob("*.png"))
        if n > 0:
            print(f"    H{half_num} already done ({n} frames)")
            return True

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
            print(f"    H{half_num} FAIL: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"    H{half_num} TIMEOUT")
    finally:
        temp_script.unlink(missing_ok=True)
        clips_dir = output_dir / "clips"
        if clips_dir.exists():
            shutil.rmtree(clips_dir, ignore_errors=True)
    return ok


# Extract games 34-40 (queue indices 3-9)
for i in range(3, 10):
    league, game_name = queue[i]
    game_num = i + 31
    video_dir = VIDEO_BASE / game_name

    if not video_dir.exists():
        print(f"game_{game_num}: NO VIDEO DIR, skipping")
        continue

    json_file = find_json_file(game_name, league)
    if not json_file:
        print(f"game_{game_num}: NO JSON, skipping")
        continue

    print(f"\ngame_{game_num}: {game_name}")
    for half_num in [1, 2]:
        video_files = sorted(video_dir.glob(f"*_{half_num}.mkv"))
        if video_files:
            ok = extract_half(video_files[0], json_file, half_num, game_num)
            if ok:
                print(f"    H{half_num} extracted")
            else:
                print(f"    H{half_num} FAILED")
        else:
            print(f"    H{half_num} no video file")

# Also check games 41+ from the 2021-22 EPL videos that were downloaded
# Count how many more video folders exist beyond game 40
extra_videos = []
for i in range(10, 40):  # queue indices 10-39
    league, game_name = queue[i]
    game_num = i + 31
    video_dir = VIDEO_BASE / game_name
    if video_dir.exists() and list(video_dir.glob("*.mkv")):
        json_file = find_json_file(game_name, league)
        if json_file:
            extra_videos.append((i, league, game_name, game_num, json_file))

if extra_videos:
    print(f"\n--- Found {len(extra_videos)} more downloaded games (41+) ---")
    for idx, league, game_name, game_num, json_file in extra_videos:
        print(f"\ngame_{game_num}: {game_name}")
        video_dir = VIDEO_BASE / game_name
        for half_num in [1, 2]:
            video_files = sorted(video_dir.glob(f"*_{half_num}.mkv"))
            if video_files:
                ok = extract_half(video_files[0], json_file, half_num, game_num)
                if ok:
                    print(f"    H{half_num} extracted")
                else:
                    print(f"    H{half_num} FAILED")

print("\n--- Done extracting. Cleaning up ALL video files ---")
if VIDEO_BASE.exists():
    size = sum(f.stat().st_size for f in VIDEO_BASE.rglob("*") if f.is_file())
    shutil.rmtree(VIDEO_BASE, ignore_errors=True)
    print(f"Freed {size/1e9:.1f} GB")
else:
    print("No videos to clean up")
