"""
Game Flow Feature Extractor
============================
Builds temporal context features for each soccer event using only
JSON annotation files — no video required.

For each event, extracts:
  - current event type
  - previous 3 event types
  - half (1 or 2)
  - minute in match
  - rolling 5-minute counts: corners, shots, fouls, cards
  - event category (ATTACK / STOPPAGE / TRANSITION / OTHER)

Works across all available JSON files (1488 games across 42 leagues).

Usage:
    python game_flow_features.py
    python game_flow_features.py --data-dir data/json/Soccer_Data_Json/train/england_epl_2021-2022
    python game_flow_features.py --data-dir data/json/Soccer_Data_Json/train --all-leagues
"""

import argparse
import glob
import json
import pathlib
import re
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Event category definitions (from Yunting's spec)
# ---------------------------------------------------------------------------
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

# Rolling window for counting recent events
WINDOW_MINUTES = 5

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    default="data/json/Soccer_Data_Json/train",
    help="Root directory to search for JSON files",
)
parser.add_argument(
    "--out-dir", default="features/game_flow",
    help="Output directory for feature files",
)
parser.add_argument(
    "--all-leagues", action="store_true",
    help="Use all leagues (default: EPL 2021-2022 only)",
)
args = parser.parse_args()

out_dir = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_label(label: str) -> str:
    """Lowercase + spaces to underscores + strip special chars."""
    label = label.lower().strip()
    label = re.sub(r"[^a-z0-9_\-]", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label


def parse_timestamp(ts: str, half: int) -> float:
    """
    Convert 'MM:SS' or 'MM+extra:SS' to absolute minutes in the match.
    Half 2 events are offset by 45 minutes.
    """
    if not ts or not ts.strip():
        return None
    ts = ts.strip()
    # Handle stoppage time: "45+2:30" -> treat as 45+2 minutes
    match = re.match(r"(\d+)(?:\+(\d+))?:(\d+)", ts)
    if not match:
        return None
    base_min = int(match.group(1))
    extra_min = int(match.group(2)) if match.group(2) else 0
    secs = int(match.group(3))
    minute = base_min + extra_min + secs / 60.0
    # Offset second half by 45 minutes for absolute match time
    if half == 2:
        minute += 45.0
    return minute


def get_category(label: str) -> str:
    if label in ATTACK_EVENTS:
        return "attack"
    if label in STOPPAGE_EVENTS:
        return "stoppage"
    if label in TRANSITION_EVENTS:
        return "transition"
    return "other"


# ---------------------------------------------------------------------------
# Collect JSON files
# ---------------------------------------------------------------------------
data_dir = pathlib.Path(args.data_dir)

if args.all_leagues:
    json_files = sorted(glob.glob(str(data_dir / "**" / "*.json"), recursive=True))
else:
    # Default: EPL 2021-2022
    epl_dir = data_dir / "england_epl_2021-2022" if "england_epl_2021-2022" not in str(data_dir) else data_dir
    json_files = sorted(glob.glob(str(epl_dir / "**" / "*.json"), recursive=True))
    if not json_files:
        json_files = sorted(glob.glob(str(data_dir / "**" / "*.json"), recursive=True))

print(f"Found {len(json_files)} JSON files in {data_dir}")

# ---------------------------------------------------------------------------
# Process each game
# ---------------------------------------------------------------------------
all_records = []
skipped_games = 0

for game_idx, json_path in enumerate(json_files):
    with open(json_path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            skipped_games += 1
            continue

    comments = data.get("comments", [])
    game_id = pathlib.Path(json_path).parent.name
    league = pathlib.Path(json_path).parent.parent.name

    # Parse and sort all valid events by (half, minute)
    events = []
    for c in comments:
        half = c.get("half", 0)
        if half not in (1, 2):
            continue  # skip pre-match (half=0)
        ts = c.get("time_stamp", "")
        minute = parse_timestamp(ts, half)
        if minute is None:
            continue
        label = normalize_label(c.get("comments_type", "unknown"))
        text = c.get("comments_text_anonymized", c.get("comments_text", ""))
        events.append({
            "half": half,
            "minute": minute,
            "label": label,
            "category": get_category(label),
            "text": text,
        })

    if len(events) < 5:
        skipped_games += 1
        continue

    # Sort by absolute match time
    events.sort(key=lambda e: e["minute"])

    # -------------------------------------------------------------------
    # Build features for each event
    # -------------------------------------------------------------------
    # Use a sliding window deque for rolling counts
    # For each event i, count events in (minute_i - WINDOW_MINUTES, minute_i)

    for i, ev in enumerate(events):
        t = ev["minute"]
        window_start = t - WINDOW_MINUTES

        # Previous 3 events (pad with "none" if not enough history)
        prev = ["none", "none", "none"]
        prev_cats = ["none", "none", "none"]
        prev_events = events[max(0, i-3):i]
        for j, pe in enumerate(reversed(prev_events)):
            prev[j] = pe["label"]
            prev_cats[j] = pe["category"]

        # Rolling counts in last WINDOW_MINUTES minutes
        window_events = [e for e in events[:i] if e["minute"] >= window_start]
        recent_corners = sum(1 for e in window_events if e["label"] in ("corner", "lead_to_corner"))
        recent_shots   = sum(1 for e in window_events if e["label"] in ("shot_off_target", "saved_by_goal-keeper", "goal"))
        recent_fouls   = sum(1 for e in window_events if e["label"] in ("foul_with_no_card", "foul_lead_to_penalty", "free_kick"))
        recent_cards   = sum(1 for e in window_events if e["label"] in ("yellow_card", "red_card", "second_yellow_card"))
        recent_attacks = sum(1 for e in window_events if e["category"] == "attack")
        recent_stops   = sum(1 for e in window_events if e["category"] == "stoppage")

        # Next 5 minutes of events (for context / future analysis)
        next_window = [e for e in events[i+1:] if e["minute"] <= t + WINDOW_MINUTES]
        next_event = next_window[0]["label"] if next_window else "none"
        n_next_attacks = sum(1 for e in next_window if e["category"] == "attack")

        all_records.append({
            # Identity
            "game_id":           game_id,
            "league":            league,
            "half":              ev["half"],
            "minute":            round(ev["minute"], 2),
            # Current event
            "label":             ev["label"],
            "category":          ev["category"],
            # Temporal context — previous events
            "prev_1":            prev[0],
            "prev_2":            prev[1],
            "prev_3":            prev[2],
            "prev_1_cat":        prev_cats[0],
            "prev_2_cat":        prev_cats[1],
            "prev_3_cat":        prev_cats[2],
            # Rolling counts (last 5 minutes)
            "recent_corners":    recent_corners,
            "recent_shots":      recent_shots,
            "recent_fouls":      recent_fouls,
            "recent_cards":      recent_cards,
            "recent_attacks":    recent_attacks,
            "recent_stoppages":  recent_stops,
            # Future context
            "next_event":        next_event,
            "n_next_attacks":    n_next_attacks,
            # Raw text
            "text":              ev["text"],
        })

print(f"Processed {len(json_files) - skipped_games} games  "
      f"({skipped_games} skipped)  ->  {len(all_records)} events")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df = pd.DataFrame(all_records)

# Save full CSV with text
csv_path = out_dir / "game_flow_features.csv"
df.to_csv(csv_path, index=False)
print(f"Saved -> {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")

# ---------------------------------------------------------------------------
# Build numeric feature matrix for ML
# ---------------------------------------------------------------------------
# Encode categorical columns
cat_cols = ["label", "category", "prev_1", "prev_2", "prev_3",
            "prev_1_cat", "prev_2_cat", "prev_3_cat", "next_event"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Feature columns for ML (no text, no raw labels, no game_id)
feature_cols = [
    "half", "minute",
    "prev_1_enc", "prev_2_enc", "prev_3_enc",
    "prev_1_cat_enc", "prev_2_cat_enc", "prev_3_cat_enc",
    "recent_corners", "recent_shots", "recent_fouls",
    "recent_cards", "recent_attacks", "recent_stoppages",
]
target_col = "label_enc"

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values

# Save as npz
npz_path = out_dir / "game_flow_features.npz"
np.savez_compressed(
    npz_path,
    X=X,
    y=y,
    game_ids=df["game_id"].values,
    leagues=df["league"].values,
    labels=df["label"].values,
    feature_names=np.array(feature_cols),
    label_classes=encoders["label"].classes_,
)
print(f"Saved -> {npz_path}  X={X.shape}  y={y.shape}")

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
print(f"\nDataset summary:")
print(f"  Games:          {df['game_id'].nunique()}")
print(f"  Leagues:        {df['league'].nunique()}")
print(f"  Total events:   {len(df)}")
print(f"  Unique labels:  {df['label'].nunique()}")
print(f"  Features:       {len(feature_cols)}")

print(f"\nTop 15 event types:")
print(df["label"].value_counts().head(15).to_string())

print(f"\nCategory distribution:")
print(df["category"].value_counts().to_string())

print(f"\nRolling count stats (last 5 min):")
for col in ["recent_corners", "recent_shots", "recent_fouls", "recent_cards"]:
    print(f"  {col:<20s}  mean={df[col].mean():.2f}  max={df[col].max()}")
