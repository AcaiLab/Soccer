"""
Write grouped cross-validation manifests: every match is a test game exactly once.

The official split scores models on 8 test matches (65-game set), so a different
handful of test matches could move the result by several points. K-fold
cross-validation by match tests every match once: fold k tests about 1/K of the
matches, draws `--val-games` validation matches from the others (seeded), and trains
on the rest. Splits stay by whole match, as everywhere else.

Usage:
    python scripts/make_cv_folds.py      # configs/games_epl2122_cv1.csv ... _cv5.csv
    python scripts/robustness.py --splits cv1 cv2 cv3 cv4 cv5 --seeds 42 43 --device cpu
"""

import argparse
import csv
import random

from soccer.config import REPO_ROOT, load_config
from soccer.manifest import assert_disjoint, load_manifest


def make_folds(game_nums: list[str], k: int, n_val: int, seed: int) -> list[dict[str, str]]:
    """For each fold, game_num -> train / val / test. Test sets partition the games."""
    order = sorted(game_nums)
    random.Random(seed).shuffle(order)
    folds = []
    for f in range(k):
        test = set(order[f::k])
        rest = sorted(g for g in game_nums if g not in test)
        val = set(random.Random(seed * 1000 + f).sample(rest, n_val))
        folds.append({g: "test" if g in test else "val" if g in val else "train" for g in game_nums})
    return folds


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Write K grouped cross-validation manifests.")
    parser.add_argument("--manifest", default="configs/games_epl2122.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-games", type=int, default=7)
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    args = parser.parse_args(argv)

    base = REPO_ROOT / args.manifest
    games = load_manifest(base)
    for f, split in enumerate(make_folds([g.game_num for g in games], args.folds, args.val_games, args.seed), 1):
        groups = {s: [g for g, v in split.items() if v == s] for s in ("train", "val", "test")}
        assert_disjoint(**groups)
        out = base.with_name(f"{base.stem}_cv{f}.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["game_num", "match_id", "league", "official_split", "split"])
            w.writerows([g.game_num, g.match_id, g.league, g.official_split, split[g.game_num]] for g in games)
        print(f"cv{f}: train {len(groups['train'])}, val {len(groups['val'])}, test {len(groups['test'])} "
              f"({' '.join(sorted(groups['test']))}) -> {out}")


if __name__ == "__main__":
    main()
