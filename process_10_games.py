"""
Wrapper script to process 10 EPL 2021-2022 games using extract_clips_frames_label.py
Organizes output by EVENT LABEL
"""
import subprocess
from pathlib import Path
import sys

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# 10 EPL 2021-2022 games (with date prefixes)
GAMES = [
    "2021-08-22_arsenal-fc-chelsea-fc-premier-league-2021-2022",
    "2021-08-28_manchester-city-arsenal-fc-premier-league-2021-2022",
    "2021-09-11_manchester-united-newcastle-united-premier-league-2021-2022",
    "2021-09-19_tottenham-hotspur-chelsea-fc-premier-league-2021-2022",
    "2021-09-25_manchester-united-aston-villa-premier-league-2021-2022",
    "2021-10-03_liverpool-fc-manchester-city-premier-league-2021-2022",
    "2021-10-16_leicester-city-manchester-united-premier-league-2021-2022",
    "2021-10-24_manchester-united-liverpool-fc-premier-league-2021-2022",
    "2021-10-24_west-ham-united-tottenham-hotspur-premier-league-2021-2022",
    "2021-11-06_manchester-united-manchester-city-premier-league-2021-2022",
]

VIDEO_BASE = Path("data/videos")
JSON_BASE = Path("data/json/Soccer_Data_Json/train/england_epl_2021-2022")
SCRIPT = Path("preprocess/extract_clips_frames_label.py")
OUTPUT_BASE = Path("results_by_label_epl_2021_2022")

def find_json_file(game_folder_name):
    """Find matching JSON file for a game."""
    # Remove date prefix for matching
    game_name_no_date = game_folder_name.split('_', 1)[1] if '_' in game_folder_name else game_folder_name

    # Try exact match
    json_folder = JSON_BASE / game_folder_name
    if json_folder.exists():
        json_files = list(json_folder.glob("*.json"))
        if json_files:
            return json_files[0]

    # Try without date
    json_folder = JSON_BASE / game_name_no_date
    if json_folder.exists():
        json_files = list(json_folder.glob("*.json"))
        if json_files:
            return json_files[0]

    # Search all folders
    for folder in JSON_BASE.iterdir():
        if folder.is_dir():
            # Check if folder name contains key parts of the game name
            game_parts = game_name_no_date.replace("-premier-league-2021-2022", "").split('-')
            folder_parts = folder.name.split('-')

            # If most parts match, it's likely the right game
            matches = sum(1 for part in game_parts if part in folder_parts)
            if matches >= len(game_parts) - 2:  # Allow some variation
                json_files = list(folder.glob("*.json"))
                if json_files:
                    return json_files[0]

    return None

def process_game_half(game_name, video_path, json_path, half_num, game_num):
    """Process one half of a game."""
    half_name = f"half_{half_num}"
    print(f"  [{half_name}] Processing...")
    print(f"    Video: {video_path.name}")
    print(f"    JSON: {json_path.name}")

    # Output folder for this half
    output_dir = OUTPUT_BASE / f"game_{game_num:02d}_{half_name}"

    # Read the original script
    with open(SCRIPT, 'r', encoding='utf-8') as f:
        script_content = f.read()

    # Replace paths and output directory
    script_content = script_content.replace(
        'video_path = "/Users/yuntingyin/Documents/Research/Soccer/watford-fc-liverpool-fc_1.mkv"',
        f'video_path = r"{str(video_path.absolute())}"'
    )
    script_content = script_content.replace(
        'json_path  = "/Users/yuntingyin/Documents/Research/Soccer/2017-08-12_Watford_3-3_Liverpool_AaZvBO5T.json"',
        f'json_path = r"{str(json_path.absolute())}"'
    )
    script_content = script_content.replace(
        'base_out   = pathlib.Path("extracts_first_half")',
        f'base_out = pathlib.Path(r"{str(output_dir.absolute())}")'
    )

    # Write temporary script
    temp_script = Path("temp_extract.py")
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(script_content)

    # Run it
    try:
        result = subprocess.run([sys.executable, str(temp_script)],
                              capture_output=True, text=True, timeout=1200)
        if result.returncode == 0:
            print(f"  [{half_name}] ✓ Complete")
        else:
            print(f"  [{half_name}] ✗ Failed")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  [{half_name}] ✗ Timed out (20 min limit)")
    except Exception as e:
        print(f"  [{half_name}] ✗ Error: {e}")
    finally:
        # Cleanup temp script
        if temp_script.exists():
            temp_script.unlink()

def main():
    print("="*70)
    print("PROCESSING 10 EPL 2021-2022 GAMES")
    print("Extracting clips and frames organized by EVENT LABEL")
    print("="*70)
    print(f"Using: {SCRIPT}")
    print(f"Output: {OUTPUT_BASE}")
    print("="*70)
    print()

    if not SCRIPT.exists():
        print(f"❌ ERROR: Script not found: {SCRIPT}")
        return

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    failed_games = []

    for i, game_name in enumerate(GAMES, 1):
        print(f"\n{'='*70}")
        print(f"Game {i}/10: {game_name}")
        print('='*70)

        # Find video folder
        video_folder = VIDEO_BASE / game_name
        if not video_folder.exists():
            print(f"  ❌ Video folder not found: {video_folder}")
            failed_games.append(game_name)
            continue

        # Find JSON file
        json_file = find_json_file(game_name)
        if not json_file:
            print(f"  ❌ JSON file not found for {game_name}")
            failed_games.append(game_name)
            continue

        # Find video files
        video_h1 = list(video_folder.glob("*_1.mkv"))
        video_h2 = list(video_folder.glob("*_2.mkv"))

        if not video_h1 or not video_h2:
            print(f"  ❌ Missing video files (need both halves)")
            failed_games.append(game_name)
            continue

        print(f"  ✓ All files found")
        print(f"  📋 JSON: {json_file.name}")

        # Process both halves
        try:
            process_game_half(game_name, video_h1[0], json_file, 1, i)
            process_game_half(game_name, video_h2[0], json_file, 2, i)
            total_processed += 1
            print(f"  ✅ Game {i} complete")
        except Exception as e:
            print(f"  ❌ Game {i} failed: {e}")
            failed_games.append(game_name)

    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"Successfully processed: {total_processed}/10 games")

    if failed_games:
        print(f"\n❌ Failed games ({len(failed_games)}):")
        for game in failed_games:
            print(f"  - {game}")

    print(f"\n📁 Output location: {OUTPUT_BASE.absolute()}")
    print("\n📂 Structure:")
    print(f"  {OUTPUT_BASE}/")
    print("    ├── game_01_half_1/")
    print("    │   ├── clips/")
    print("    │   │   ├── goal/")
    print("    │   │   ├── foul/")
    print("    │   │   └── ...")
    print("    │   └── frames/")
    print("    │       ├── goal/")
    print("    │       └── ...")
    print("    ├── game_01_half_2/")
    print("    └── ...")
    print("="*70)

if __name__ == "__main__":
    main()
