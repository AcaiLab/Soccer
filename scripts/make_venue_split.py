"""
Build a leave-venue-out version of the games manifest.

The default manifest splits by match, but 5 of its 6 test games are played at
stadiums that also appear in training, so a model could still score by
recognising the ground. This writes a manifest in which every stadium belongs
to exactly one split: test and validation stadiums never appear in training.

Stadiums come from each match's annotation JSON (match_info.venue). Whole
stadiums are assigned greedily, smallest first, in a seeded random order,
until test and validation reach their target number of games.

Usage:
    python scripts/make_venue_split.py                     # writes configs/games_venue_split.csv
    SOCCER_MANIFEST=configs/games_venue_split.csv python run_dino_pipeline.py --stages contrastive classify
"""

import argparse
import csv
import json
import random
from collections import defaultdict

from soccer.config import REPO_ROOT, data_path, load_config, repo_path
from soccer.manifest import assert_disjoint, load_manifest
from soccer.pipeline import annotation_for


def assign(venues: dict[str, list], n_test: int, n_val: int, seed: int) -> dict[str, str]:
    """venue -> split, filling test then val with whole venues."""
    order = sorted(venues, key=lambda v: (len(venues[v]), v))
    rng = random.Random(seed)
    # Shuffle within equal sizes so the choice is not just alphabetical.
    order = sorted(order, key=lambda v: (len(venues[v]), rng.random()))
    split, filled = {}, {"test": 0, "val": 0}
    for v in order:
        for name, target in (("test", n_test), ("val", n_val)):
            if filled[name] + len(venues[v]) <= target:
                split[v], filled[name] = name, filled[name] + len(venues[v])
                break
        else:
            split[v] = "train"
    return split


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Write a leave-venue-out games manifest.")
    parser.add_argument("--test-games", type=int, default=6)
    parser.add_argument("--val-games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--out", default="configs/games_venue_split.csv")
    args = parser.parse_args(argv)

    games = load_manifest(repo_path(cfg, "manifest"), repo_path(cfg, "excluded_matches"))
    annotations = data_path(cfg, "annotations")
    venue_of = {}
    for g in games:
        with open(annotation_for(annotations, g), encoding="utf-8") as f:
            venue_of[g.game_num] = json.load(f)["match_info"].get("venue") or f"unknown-{g.game_num}"
    venues = defaultdict(list)
    for g in games:
        venues[venue_of[g.game_num]].append(g)

    split_of_venue = assign(venues, args.test_games, args.val_games, args.seed)
    rows = [{"game_num": g.game_num, "match_id": g.match_id, "league": g.league,
             "official_split": g.official_split, "split": split_of_venue[venue_of[g.game_num]]} for g in games]
    by_split = {s: {venue_of[r["game_num"]] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    assert_disjoint(**by_split)  # no stadium in two splits

    out = REPO_ROOT / args.out
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for s in ("train", "val", "test"):
        n = sum(r["split"] == s for r in rows)
        print(f"{s:5s} {n:2d} games, {len(by_split[s])} stadiums: {', '.join(sorted(by_split[s]))}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
