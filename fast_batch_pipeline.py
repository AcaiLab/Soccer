"""
Fast Batch Pipeline: Continuous download -> extract -> cleanup cycle.
Processes games from a queue file serially, skips finished work and saves progress.

Superseded by batch_pipeline.py, which reads the games manifest instead of a
gitignored queue file; kept until that switch is approved (docs/cleanup_log.md).

Credentials come from the environment or AWS SSM, never from code; see
soccer/dropbox.py and the README.

Usage:
    python fast_batch_pipeline.py --max-games 50
    python fast_batch_pipeline.py --max-games 0   # all games
"""

import argparse
import json
import pathlib
import shutil
import sys
import time

from soccer.annotations import find_annotation_file
from soccer.config import data_path, load_config
from soccer.dropbox import DropboxSharedLink, TokenExpired
from soccer.extract import ExtractionSettings
from soccer.manifest import Game
from soccer.pipeline import download_videos, extract_game

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-games", type=int, default=0, help="0 = all")
    parser.add_argument("--start-game-num", type=int, default=31)
    parser.add_argument("--token-file", default="tmp_dbx/token.txt")
    parser.add_argument("--queue-file", default="tmp_dbx/download_queue.json")
    parser.add_argument("--progress-file", default="tmp_dbx/fast_progress.json")
    parser.add_argument("--frames-root", default=str(data_path(cfg, "frames")))
    args = parser.parse_args(argv)

    videos_root = data_path(cfg, "videos")
    annotations_root = data_path(cfg, "annotations")
    frames_root = pathlib.Path(args.frames_root)
    settings = ExtractionSettings.from_config(cfg)
    client = DropboxSharedLink.from_env(token_file=args.token_file)

    with open(args.queue_file, encoding="utf-8") as f:
        queue = json.load(f)
    progress_file = pathlib.Path(args.progress_file)
    if progress_file.exists():
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
    else:
        progress = {"done": [], "failed": [], "game_num": args.start_game_num}

    def save_progress():
        progress_file.write_text(json.dumps(progress), encoding="utf-8")

    def mark_failed(league, match_id, reason):
        print(f"  FAILED: {reason}", flush=True)
        progress["failed"].append([league, match_id])
        progress["game_num"] += 1
        save_progress()

    finished = {tuple(x) for x in progress["done"] + progress["failed"]}
    remaining = [(lg, m) for lg, m in queue if (lg, m) not in finished]
    if args.max_games > 0:
        remaining = remaining[:args.max_games]
    print(f"Total queue: {len(queue)}, Done: {len(progress['done'])}, "
          f"Failed: {len(progress['failed'])}, Remaining: {len(remaining)}\n", flush=True)

    t_start = time.time()
    for i, (league, match_id) in enumerate(remaining):
        elapsed = time.time() - t_start
        rate = (i / elapsed * 3600) if elapsed > 0 and i > 0 else 0
        num = f"{progress['game_num']:02d}"
        print(f"\n[{i+1}/{len(remaining)}] game_{num}: {match_id}  ({rate:.0f} games/hr)", flush=True)

        found = find_annotation_file(annotations_root, league, match_id)
        if found is None:
            mark_failed(league, match_id, "no JSON annotation")
            continue
        annotation, split = found
        game = Game(num, match_id, league, split, "unassigned")
        try:
            video_dir = download_videos(client, game, videos_root)
            extract_game(game, video_dir, annotation, frames_root, settings)
        except TokenExpired as e:
            print(f"\n*** {e} after {i} games ***", flush=True)
            save_progress()
            sys.exit(1)
        except Exception as e:
            shutil.rmtree(videos_root / match_id, ignore_errors=True)
            mark_failed(league, match_id, e)
            continue

        size = sum(f.stat().st_size for f in video_dir.rglob("*") if f.is_file())
        shutil.rmtree(video_dir, ignore_errors=True)
        print(f"  Cleanup: freed {size / 1e9:.1f} GB", flush=True)
        progress["done"].append([league, match_id])
        progress["game_num"] += 1
        save_progress()

    print(f"\n{'=' * 60}")
    print(f"DONE. Processed {len(progress['done'])} games in {(time.time() - t_start) / 3600:.1f} hours")
    print(f"Failed: {len(progress['failed'])}")
    print(f"Next game_num: {progress['game_num']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
