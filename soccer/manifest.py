"""The games manifest: which matches an experiment uses and how they are split.

configs/games.csv has one row per match:

    game_num, match_id, league, official_split, split

`game_num` is the short id used in frame folders (game_07_half_1) and feature
files; `split` is this project's train / val / test assignment. Splits are by
whole match, so no match can contribute events to more than one split.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Game:
    game_num: str
    match_id: str
    league: str
    official_split: str
    split: str


def normalize_game_id(game_id) -> str:
    """Folder and feature game ids appear as "7", "07" or "0007"; use "07"."""
    return str(int(str(game_id))).zfill(2)


def load_excluded(path: str | pathlib.Path) -> dict[tuple[str, str], str]:
    """(league, match_id) -> reason, for matches that must never be used."""
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["league"], r["match_id"]): r["reason"] for r in csv.DictReader(f)}


def load_manifest(path: str | pathlib.Path,
                  excluded_path: str | pathlib.Path | None = None) -> list[Game]:
    """Read and validate the manifest.

    Raises ValueError on duplicate game numbers or matches, unknown splits, or
    a match listed in the exclusions file.
    """
    with open(path, newline="", encoding="utf-8") as f:
        games = [Game(normalize_game_id(r["game_num"]), r["match_id"], r["league"],
                      r["official_split"], r["split"]) for r in csv.DictReader(f)]
    excluded = load_excluded(excluded_path) if excluded_path else {}

    nums, matches = set(), set()
    for g in games:
        if g.split not in SPLITS:
            raise ValueError(f"game {g.game_num}: unknown split {g.split!r}")
        if g.game_num in nums:
            raise ValueError(f"game number {g.game_num} appears twice")
        if g.match_id in matches:
            raise ValueError(f"match {g.match_id} appears twice")
        if (g.league, g.match_id) in excluded:
            raise ValueError(f"match {g.match_id} is excluded: {excluded[(g.league, g.match_id)]}")
        nums.add(g.game_num)
        matches.add(g.match_id)
    return games


def game_nums(games: list[Game], *splits: str) -> list[str]:
    """Game numbers in the given splits, in manifest order."""
    return [g.game_num for g in games if g.split in splits]


def split_of(games: list[Game]) -> dict[str, str]:
    """game_num -> split."""
    return {g.game_num: g.split for g in games}


def assert_disjoint(**groups) -> None:
    """Raise if any id appears in more than one named group.

    assert_disjoint(train=train_games, test=test_games)
    """
    seen: dict[str, str] = {}
    for name, ids in groups.items():
        for i in ids:
            if i in seen and seen[i] != name:
                raise ValueError(f"{i!r} is in both {seen[i]} and {name}")
            seen[i] = name


def manifest_games(cfg: dict) -> list[str]:
    """Game numbers of the manifest in use (configs/games.csv or $SOCCER_MANIFEST)."""
    from soccer.config import repo_path

    return [g.game_num for g in load_manifest(repo_path(cfg, "manifest"))]
