"""
Copy manifest games' videos from the SoccerReplay-1988 Dropbox link into S3.

Every file streams from Dropbox straight into S3 (no local disk), so run this
wherever bandwidth is cheap: the GPU server, an EC2 instance, or as the Lambda
in soccer/transfer.py. Files already in S3 with the right size are skipped, so
it can be rerun after an interruption or an expired token.

Credentials: Dropbox settings from the environment or SSM (soccer/dropbox.py);
AWS from the standard chain (instance role, AWS_PROFILE).

Usage:
    python scripts/transfer_dropbox_to_s3.py                       # all manifest games
    python scripts/transfer_dropbox_to_s3.py --games 01 02 --workers 2
    python scripts/transfer_dropbox_to_s3.py --dest s3://my-bucket/videos
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from soccer.config import load_config, repo_path, s3_uri
from soccer.dropbox import DropboxSharedLink, TokenExpired
from soccer.manifest import load_manifest, normalize_game_id
from soccer.transfer import copy_game


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Stream manifest games' videos from Dropbox into S3.")
    parser.add_argument("--dest", default=None, help="s3://bucket/prefix (default: <s3_root>/videos)")
    parser.add_argument("--games", nargs="+", default=None, help="game numbers (default: all in manifest)")
    parser.add_argument("--workers", type=int, default=2, help="games copied in parallel")
    parser.add_argument("--token-file", default=None,
                        help="Dropbox token file if DROPBOX_TOKEN is not set (default: $DROPBOX_TOKEN_FILE, else tmp_dbx/token.txt)")
    args = parser.parse_args(argv)

    dest = args.dest or s3_uri(cfg, "videos")
    games = load_manifest(repo_path(cfg, "manifest"), repo_path(cfg, "excluded_matches"))
    if args.games:
        wanted = {normalize_game_id(g) for g in args.games}
        games = [g for g in games if g.game_num in wanted]
    client = DropboxSharedLink.from_env(token_file=args.token_file)
    print(f"Copying {len(games)} games to {dest} with {args.workers} workers", flush=True)

    t0, total, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(copy_game, client, g, dest): g for g in games}
        for fut in as_completed(futures):
            g = futures[fut]
            try:
                for name, size, outcome in fut.result():
                    total += size if outcome == "copied" else 0
                    print(f"game_{g.game_num} {name}  {size / 1e9:.2f} GB  {outcome}", flush=True)
            except TokenExpired as e:
                print(f"\n*** {e}. Set a new token and rerun; finished files are kept. ***", flush=True)
                pool.shutdown(cancel_futures=True)
                sys.exit(2)
            except Exception as e:  # one bad game should not stop the rest
                print(f"game_{g.game_num} FAILED: {e}", flush=True)
                failed.append(g.game_num)

    hours = (time.time() - t0) / 3600
    print(f"\nCopied {total / 1e9:.1f} GB in {hours:.2f} h; {len(games) - len(failed)}/{len(games)} games complete")
    if failed:
        print("Failed: " + " ".join(sorted(failed)))
        sys.exit(1)


if __name__ == "__main__":
    main()
