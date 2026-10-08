"""
Batch Pipeline: Download -> Extract Frames -> Delete Video
==========================================================
For every game in the manifest (configs/games.csv): download both halves
from the SoccerReplay-1988 Dropbox link, extract event-centred frames from
each half, then delete the video to free disk.

Resumable: a half is skipped once its extraction marker exists, and an
interrupted half is redone. Feature extraction is a separate GPU step
(temporal_clip_features.py), so this script runs on any CPU machine.

Credentials come from the environment or AWS SSM, never from code; see
soccer/dropbox.py and the README.

Usage:
    python scripts/batch_pipeline.py                       # every manifest game
    python scripts/batch_pipeline.py --games 01 02 25      # selected games
    python scripts/batch_pipeline.py --keep-videos         # keep the .mkv files
    python scripts/batch_pipeline.py --local-only          # use videos already in data/videos
"""

import argparse
import shutil
import sys

from soccer.config import data_path, load_config, repo_path
from soccer.dropbox import DropboxSharedLink, TokenExpired
from soccer.extract import ExtractionSettings
from soccer.manifest import load_manifest, normalize_game_id
from soccer.pipeline import annotation_for, download_videos, extract_game, is_extracted


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Download, extract frames, delete video.")
    parser.add_argument("--games", nargs="+", default=None, help="game numbers (default: all in manifest)")
    parser.add_argument("--max-games", type=int, default=0, help="stop after N games (0 = no limit)")
    parser.add_argument("--keep-videos", action="store_true", help="do not delete videos after extraction")
    parser.add_argument("--local-only", action="store_true",
                        help="no download: extract from videos already in the videos folder (implies --keep-videos)")
    parser.add_argument("--token-file", default="tmp_dbx/token.txt",
                        help="Dropbox token file, used if DROPBOX_TOKEN is not set")
    args = parser.parse_args(argv)

    games = load_manifest(repo_path(cfg, "manifest"), repo_path(cfg, "excluded_matches"))
    if args.games:
        wanted = {normalize_game_id(g) for g in args.games}
        games = [g for g in games if g.game_num in wanted]
    frames_root = data_path(cfg, "frames")
    videos_root = data_path(cfg, "videos")
    annotations_root = data_path(cfg, "annotations")
    settings = ExtractionSettings.from_config(cfg)

    todo = [g for g in games if not is_extracted(frames_root, g)]
    if args.max_games:
        todo = todo[:args.max_games]
    print(f"{len(games)} games selected, {len(games) - len(todo)} already extracted, {len(todo)} to do")
    print(f"Window: -{settings.before_sec}s / +{settings.after_sec}s, every {settings.sample_every_sec}s\n")
    if not todo:
        return

    client = None if args.local_only else DropboxSharedLink.from_env(token_file=args.token_file)
    failed = []
    for i, game in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] game_{game.game_num}: {game.league}/{game.match_id}", flush=True)
        try:
            annotation = annotation_for(annotations_root, game)
            video_dir = (videos_root / game.match_id if args.local_only
                         else download_videos(client, game, videos_root))
            extract_game(game, video_dir, annotation, frames_root, settings)
        except TokenExpired as e:
            print(f"\n*** {e}. Set a new token and rerun; finished games are kept. ***")
            sys.exit(2)
        except Exception as e:  # keep going; one bad game should not stop the batch
            print(f"    FAILED: {e}", flush=True)
            failed.append(game.game_num)
            continue
        if not (args.keep_videos or args.local_only):
            shutil.rmtree(video_dir, ignore_errors=True)

    print(f"\nDone. {len(todo) - len(failed)}/{len(todo)} games extracted to {frames_root}")
    if failed:
        print(f"Failed: {' '.join(failed)}")
        sys.exit(1)
    print("Next: python scripts/temporal_clip_features.py --model facebook/dinov2-base")


if __name__ == "__main__":
    main()
