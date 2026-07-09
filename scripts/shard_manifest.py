import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline_utils import ensure_dir, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/local_windows.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("shards"))
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--by-game", action="store_true", help="Assign complete games to shards.")
    return parser.parse_args()


def shard_by_index(rows: list[dict], num_shards: int) -> list[list[dict]]:
    shards = [[] for _ in range(num_shards)]
    for i, row in enumerate(rows):
        shards[i % num_shards].append(row)
    return shards


def shard_by_game(rows: list[dict], num_shards: int) -> list[list[dict]]:
    games = sorted({row["game_label"] for row in rows})
    game_to_shard = {game: i % num_shards for i, game in enumerate(games)}
    shards = [[] for _ in range(num_shards)]
    for row in rows:
        shards[game_to_shard[row["game_label"]]].append(row)
    return shards


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    rows = read_jsonl(args.manifest)
    shards = shard_by_game(rows, args.num_shards) if args.by_game else shard_by_index(rows, args.num_shards)
    ensure_dir(args.out_dir)
    summary = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "num_shards": args.num_shards,
        "by_game": args.by_game,
        "total_rows": len(rows),
        "shards": [],
    }
    for shard_id, shard_rows in enumerate(shards):
        shard_dir = ensure_dir(args.out_dir / f"shard_{shard_id:04d}")
        out = shard_dir / "windows.jsonl"
        write_jsonl(out, shard_rows)
        item = {
            "shard_id": shard_id,
            "path": str(out),
            "rows": len(shard_rows),
            "games": len({row["game_label"] for row in shard_rows}),
            "labels": dict(Counter(row["event_label"] for row in shard_rows)),
        }
        write_json(shard_dir / "shard_summary.json", item)
        summary["shards"].append(item)
    write_json(args.out_dir / "shard_index.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
