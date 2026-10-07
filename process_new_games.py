"""
Extract frames for additional games from videos already in data/videos/<match_id>/.

Superseded by `python batch_pipeline.py --local-only --games ...`, which this
forwards to; kept until its removal is approved (docs/cleanup_log.md).

Every game in --games-file must have a row in configs/games.csv, which fixes
its game number and split; numbering is no longer derived from --start-num.

Usage:
    python process_new_games.py --games-file tmp_dbx/selected_games.json
"""

import argparse
import json

import batch_pipeline
from soccer.config import load_config, repo_path
from soccer.manifest import load_manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-file", required=True, help="JSON list of match folder names")
    args = parser.parse_args(argv)

    with open(args.games_file, encoding="utf-8") as f:
        match_ids = json.load(f)
    cfg = load_config()
    by_match = {g.match_id: g.game_num for g in load_manifest(repo_path(cfg, "manifest"))}
    missing = [m for m in match_ids if m not in by_match]
    if missing:
        raise SystemExit("Add these matches to configs/games.csv first:\n  " + "\n  ".join(missing))
    batch_pipeline.main(["--local-only", "--games", *(by_match[m] for m in match_ids)])


if __name__ == "__main__":
    main()
