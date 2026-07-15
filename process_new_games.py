"""
Process additional EPL 2021-2022 games (11+) using extract_clips_frames_label.py.
Frames-only mode: clip writing is patched out to save disk space and time.

Output structure matches process_10_games.py:
    results_by_label_epl_2021_2022/game_NN_half_H/frames/<event>/*.png

Usage:
    python process_new_games.py --games-file tmp_dbx/selected_games.json --start-num 11
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--games-file", required=True,
                    help="JSON list of game folder names")
parser.add_argument("--start-num", type=int, default=11,
                    help="Game number to assign to the first game")
args = parser.parse_args()

GAMES = json.load(open(args.games_file))

VIDEO_BASE = Path("data/videos")
JSON_BASE = Path("data/json/Soccer_Data_Json/train/england_epl_2021-2022")
SCRIPT = Path("preprocess/extract_clips_frames_label.py")
OUTPUT_BASE = Path("results_by_label_epl_2021_2022")


def find_json_file(game_folder_name):
    json_folder = JSON_BASE / game_folder_name
    if json_folder.exists():
        json_files = list(json_folder.glob("*.json"))
        if json_files:
            return json_files[0]
    return None


def process_game_half(video_path, json_path, half_num, game_num):
    half_name = f"half_{half_num}"
    output_dir = OUTPUT_BASE / f"game_{game_num:02d}_{half_name}"
    if (output_dir / "frames").exists():
        print(f"  [{half_name}] already extracted, skipping")
        return True
    print(f"  [{half_name}] processing {video_path.name} ...", flush=True)

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
    # Frames-only: disable clip frame writing (headers-only mp4s remain, removed below)
    script_content = script_content.replace(
        "        vout.write(frame)\n", "\n")
    # Write JPEG instead of PNG (~10x smaller; DINOv2 resizes to 224px anyway)
    script_content = script_content.replace(
        'out_png = frames_dir / f"{event_dir}_frame_{tag_ts}_t{label}.png"\n'
        "            cv2.imwrite(str(out_png), frame)",
        'out_png = frames_dir / f"{event_dir}_frame_{tag_ts}_t{label}.jpg"\n'
        "            cv2.imwrite(str(out_png), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])")

    temp_script = Path(f"temp_extract_{game_num:02d}_{half_num}.py")
    temp_script.write_text(script_content, encoding="utf-8")
    ok = False
    try:
        result = subprocess.run([sys.executable, str(temp_script)],
                                capture_output=True, text=True, timeout=2400,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if result.returncode == 0:
            ok = True
            print(f"  [{half_name}] done")
        else:
            print(f"  [{half_name}] FAILED: {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        print(f"  [{half_name}] TIMED OUT (40 min)")
    finally:
        temp_script.unlink(missing_ok=True)
        # Remove empty clip stubs
        clips_dir = output_dir / "clips"
        if clips_dir.exists():
            shutil.rmtree(clips_dir, ignore_errors=True)
    return ok


def main():
    print(f"Processing {len(GAMES)} games starting at game_{args.start_num:02d}")
    failed = []
    for i, game_name in enumerate(GAMES):
        game_num = args.start_num + i
        print(f"\n[{i+1}/{len(GAMES)}] game_{game_num:02d}: {game_name}", flush=True)

        video_folder = VIDEO_BASE / game_name
        json_file = find_json_file(game_name)
        video_h1 = sorted(video_folder.glob("*_1.mkv")) if video_folder.exists() else []
        video_h2 = sorted(video_folder.glob("*_2.mkv")) if video_folder.exists() else []

        if not json_file or not video_h1 or not video_h2:
            print(f"  MISSING: json={bool(json_file)} h1={bool(video_h1)} h2={bool(video_h2)}")
            failed.append(game_name)
            continue

        ok1 = process_game_half(video_h1[0], json_file, 1, game_num)
        ok2 = process_game_half(video_h2[0], json_file, 2, game_num)
        if not (ok1 and ok2):
            failed.append(game_name)

    print(f"\nComplete. {len(GAMES)-len(failed)}/{len(GAMES)} games OK.")
    if failed:
        print("Failed games:")
        for g in failed:
            print(f"  - {g}")


if __name__ == "__main__":
    main()
