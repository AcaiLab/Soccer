"""
Build a games manifest for whole league-seasons of the Dropbox share.

Lists every match folder of the given league-seasons on the SoccerReplay-1988
Dropbox link, looks up each match's official split in the installed annotation
JSONs, and numbers the matches from --start in league then folder (date) order,
so new games never reuse the numbers of games already extracted.

The project split is the official one (valid -> val). A match with no
annotation JSON yet is written as official train, since the valid and test
archives are installed in full and only the train archive can be missing; its
extraction then waits until the train annotations are installed
(batch_pipeline.py refuses a game whose JSON is missing or in another split).

Usage:
    python scripts/make_league_manifest.py --start 66 --out configs/games_epl_1720.csv \
        --leagues england_epl_2017-2018 england_epl_2018-2019 england_epl_2019-2020
"""

import argparse
import csv
import pathlib

from soccer.annotations import find_annotation_file
from soccer.config import REPO_ROOT, data_path, load_config, repo_path
from soccer.dropbox import DropboxSharedLink
from soccer.manifest import load_excluded, load_manifest

PROJECT_SPLIT = {"train": "train", "valid": "val", "test": "test"}


def build_rows(folders: dict[str, list[str]], official_split, start: int,
               excluded: dict[tuple[str, str], str]) -> tuple[list[dict], list[str]]:
    """Manifest rows for {league: [match folder, ...]}, and the matches with no annotation.

    official_split(league, match_id) returns "train" / "valid" / "test", or None
    when the match has no annotation JSON.
    """
    rows, missing = [], []
    num = start
    for league in sorted(folders):
        for match_id in sorted(folders[league]):
            if (league, match_id) in excluded:
                continue
            split = official_split(league, match_id)
            if split is None:
                missing.append(f"{league}/{match_id}")
                split = "train"
            rows.append({"game_num": f"{num:02d}", "match_id": match_id, "league": league,
                         "official_split": split, "split": PROJECT_SPLIT[split]})
            num += 1
    return rows, missing


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Write a manifest covering whole league-seasons.")
    parser.add_argument("--leagues", nargs="+", required=True, help="league-season folders, e.g. england_epl_2018-2019")
    parser.add_argument("--start", type=int, required=True, help="first game number")
    parser.add_argument("--out", required=True, help="manifest to write, relative to the repository")
    args = parser.parse_args(argv)

    client = DropboxSharedLink.from_env()
    folders = {lg: [e["name"] for e in client.list_folder(f"/{lg}") if e[".tag"] == "folder"]
               for lg in args.leagues}
    annotations_root = data_path(cfg, "annotations")

    def official_split(league, match_id):
        found = find_annotation_file(annotations_root, league, match_id)
        return found[1] if found else None

    excluded = load_excluded(repo_path(cfg, "excluded_matches"))
    rows, missing = build_rows(folders, official_split, args.start, excluded)

    out = pathlib.Path(REPO_ROOT, args.out)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    load_manifest(out, repo_path(cfg, "excluded_matches"))  # refuses duplicates and excluded matches

    for lg in args.leagues:
        n = {s: sum(r["league"] == lg and r["split"] == s for r in rows) for s in PROJECT_SPLIT.values()}
        print(f"{lg}: {sum(n.values())} matches (train {n['train']}, val {n['val']}, test {n['test']})")
    print(f"Wrote {len(rows)} games ({rows[0]['game_num']}-{rows[-1]['game_num']}) to {out}")
    if missing:
        print(f"{len(missing)} matches have no annotation JSON yet (listed as train): "
              f"install the train archive before extracting them")


if __name__ == "__main__":
    main()
