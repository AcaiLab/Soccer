"""
Download game videos from the SoccerReplay-1988 Dropbox shared folder.

Downloads both halves of each selected manifest game into
data/videos/<match_id>/ without extracting anything. Use batch_pipeline.py
to download, extract and clean up in one go.

Credentials come from the environment or AWS SSM, never from code; see
soccer/dropbox.py and the README.

Usage:
    python scripts/download_dropbox_games.py --games 01 02
    python scripts/download_dropbox_games.py              # every manifest game
"""

import argparse
import sys

from soccer.config import data_path, load_config, repo_path
from soccer.dropbox import DropboxSharedLink, TokenExpired
from soccer.manifest import load_manifest, normalize_game_id
from soccer.pipeline import download_videos


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Download SoccerReplay-1988 videos for manifest games.")
    parser.add_argument("--games", nargs="+", default=None, help="game numbers (default: all in manifest)")
    parser.add_argument("--token-file", default="tmp_dbx/token.txt",
                        help="Dropbox token file, used if DROPBOX_TOKEN is not set")
    args = parser.parse_args(argv)

    games = load_manifest(repo_path(cfg, "manifest"), repo_path(cfg, "excluded_matches"))
    if args.games:
        wanted = {normalize_game_id(g) for g in args.games}
        games = [g for g in games if g.game_num in wanted]
    client = DropboxSharedLink.from_env(token_file=args.token_file)
    videos_root = data_path(cfg, "videos")

    for i, game in enumerate(games, 1):
        print(f"[{i}/{len(games)}] game_{game.game_num}: {game.match_id}", flush=True)
        try:
            download_videos(client, game, videos_root)
        except TokenExpired as e:
            print(f"*** {e}. Set a new token and rerun. ***")
            sys.exit(2)


if __name__ == "__main__":
    main()
