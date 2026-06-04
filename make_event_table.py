#!/usr/bin/env python3
"""Make a simple event table from SoccerReplay annotation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("Soccer_Data/HF_SoccerReplay_1988/extracted"),
        help="Folder containing SoccerReplay annotation JSON files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path for saving the event table.",
    )
    return parser.parse_args()


def make_event_table(annotation_root: Path) -> pd.DataFrame:
    rows = []

    for path in annotation_root.rglob("*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        game_id = path.stem

        for comment in data["comments"]:
            if not comment["time_stamp"]:
                continue

            rows.append(
                {
                    "game_id": game_id,
                    "half": comment["half"],
                    "time": comment["time_stamp"],
                    "event": comment["comments_type"],
                    "text": comment["comments_text"],
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    df = make_event_table(args.annotation_root)

    print(df.head())
    print(df["event"].value_counts())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} rows to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
