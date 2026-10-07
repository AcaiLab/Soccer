"""
Extract frames for the 10 pilot EPL 2021-2022 games (game_01 ... game_10)
from videos already downloaded to data/videos/<match_id>/.

Superseded by `python batch_pipeline.py --local-only --games 01 ... 10`, which
this forwards to; kept until its removal is approved (docs/cleanup_log.md).
The game numbering lives in configs/games.csv.
"""

import batch_pipeline

PILOT_GAMES = [f"{n:02d}" for n in range(1, 11)]


def main():
    batch_pipeline.main(["--local-only", "--games", *PILOT_GAMES])


if __name__ == "__main__":
    main()
