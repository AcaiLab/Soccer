"""
Game Flow Feature Extractor
============================
Builds temporal context features for each soccer event using only JSON
annotation files - no video required.

For each event, records:
  - its label (the target) and category (ATTACK / STOPPAGE / TRANSITION / OTHER)
  - the previous 3 event types and categories
  - half and minute in the match
  - rolling 5-minute counts of corners, shots, fouls, cards, attacks, stoppages

Only earlier events of the same match are used. Nothing about later events is
computed, and categorical columns are left as text so each classifier can fit
its encoder on training games only (soccer/game_flow.py).

By default the features cover the manifest's matches (configs/games.csv or
$SOCCER_MANIFEST), wherever they sit in the official splits. --all-leagues
covers every match of the given official splits instead.

Usage:
    python scripts/game_flow_features.py
    python scripts/game_flow_features.py --all-leagues
"""

import argparse
import pathlib

import pandas as pd

from soccer.annotations import SPLITS, match_minute, parse_timestamp, sanitize_label
from soccer.config import data_path, load_config, repo_path
from soccer.manifest import load_manifest
from soccer.pipeline import annotation_for

STOPPAGE_EVENTS = {
    "substitution", "injury", "show_added_time",
    "start_of_half_game", "end_of_half_game",
    "yellow_card", "red_card", "second_yellow_card", "var",
}
ATTACK_EVENTS = {
    "goal", "shot_off_target", "saved_by_goal-keeper", "corner",
    "lead_to_corner", "penalty", "foul_lead_to_penalty", "free_kick",
}
TRANSITION_EVENTS = {
    "clearance", "ball_possession", "off_side",
    "ball_out_of_play", "throw_in", "foul_with_no_card",
}
WINDOW_MINUTES = 5


def get_category(label: str) -> str:
    if label in ATTACK_EVENTS:
        return "attack"
    if label in STOPPAGE_EVENTS:
        return "stoppage"
    if label in TRANSITION_EVENTS:
        return "transition"
    return "other"


def match_events(comments: list[dict]) -> list[dict]:
    events = []
    for c in comments:
        half = c.get("half")
        if half not in (1, 2):
            continue  # pre-match
        try:
            seconds = parse_timestamp(c.get("time_stamp") or "")
        except ValueError:
            continue
        label = sanitize_label(c.get("comments_type"))
        events.append({"half": half, "minute": match_minute(seconds, half), "label": label,
                       "category": get_category(label),
                       "text": c.get("comments_text_anonymized", c.get("comments_text", ""))})
    events.sort(key=lambda e: e["minute"])
    return events


def count(window: list[dict], labels: set[str]) -> int:
    return sum(e["label"] in labels for e in window)


def features_for_match(events: list[dict]) -> list[dict]:
    rows = []
    for i, ev in enumerate(events):
        prev = list(reversed(events[max(0, i - 3):i])) + [None] * 3
        window = [e for e in events[:i] if e["minute"] >= ev["minute"] - WINDOW_MINUTES]
        rows.append({
            "half": ev["half"], "minute": ev["minute"], "label": ev["label"], "category": ev["category"],
            **{f"prev_{j + 1}": (prev[j] or {}).get("label", "none") for j in range(3)},
            **{f"prev_{j + 1}_cat": (prev[j] or {}).get("category", "none") for j in range(3)},
            "recent_corners": count(window, {"corner", "lead_to_corner"}),
            "recent_shots": count(window, {"shot_off_target", "saved_by_goal-keeper", "goal"}),
            "recent_fouls": count(window, {"foul_with_no_card", "foul_lead_to_penalty", "free_kick"}),
            "recent_cards": count(window, {"yellow_card", "red_card", "second_yellow_card"}),
            "recent_attacks": sum(e["category"] == "attack" for e in window),
            "recent_stoppages": sum(e["category"] == "stoppage" for e in window),
            "text": ev["text"],
        })
    return rows


def main(argv=None):
    cfg = load_config()
    root = data_path(cfg, "annotations")
    parser = argparse.ArgumentParser(description="Annotation-only temporal context features.")
    parser.add_argument("--data-dir", default=str(root), help="SoccerReplay JSON root (holds train/valid/test)")
    parser.add_argument("--out-dir", default=str(data_path(cfg, "features") / "game_flow"))
    parser.add_argument("--all-leagues", action="store_true",
                        help="every match in --splits instead of the manifest's matches "
                             "(for the cross-league annotation-only classifier)")
    parser.add_argument("--splits", nargs="+", default=["train"], choices=list(SPLITS),
                        help="with --all-leagues: official splits to read (default: train only, so "
                             "the official valid/test matches stay unseen)")
    args = parser.parse_args(argv)

    import json

    data_dir, out_dir = pathlib.Path(args.data_dir), pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.all_leagues:
        files = sorted(f for split in args.splits for f in data_dir.glob(f"{split}/*/*/*.json")
                       if not f.name.startswith("."))
    else:
        # The manifest's matches, from whichever official split each one is in,
        # so validation and test games get annotation features too.
        files = [annotation_for(data_dir, g) for g in load_manifest(repo_path(cfg, "manifest"))]
    print(f"Found {len(files)} JSON files under {data_dir}")

    rows, skipped = [], 0
    for path in files:
        split, league, match_id = path.parts[-4], path.parts[-3], path.parts[-2]
        try:
            comments = json.loads(path.read_text(encoding="utf-8")).get("comments", [])
        except json.JSONDecodeError:
            skipped += 1
            continue
        events = match_events(comments)
        if len(events) < 5:
            skipped += 1
            continue
        for r in features_for_match(events):
            rows.append({"match_id": match_id, "league": league, "official_split": split, **r})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "game_flow_features.csv", index=False)
    print(f"Processed {len(files) - skipped} matches ({skipped} skipped) -> {len(df)} events")
    print(f"Saved -> {out_dir / 'game_flow_features.csv'}")
    print(f"\nLeagues: {df['league'].nunique()}  Labels: {df['label'].nunique()}")
    print(df["label"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
