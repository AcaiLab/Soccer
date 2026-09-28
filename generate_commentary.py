"""
Improved Audience-Adaptive Commentary Generator
================================================
Generates commentary for beginner, expert, and low_vision audiences
with proper entity detail, surface variation, and correct facts for
all event types.

Fixes over original pipeline:
  1. Complete fact dictionaries for all 26 event types (no generic fallback)
  2. Entity-aware: injects player names, team names, score from raw JSON
  3. Surface variation: paraphrase pools with random selection
  4. No RAG jargon leaking into expert text
  5. Grammar fixes (a/an, proper label formatting)

Usage:
    python generate_commentary.py --shard commentary_review/shard_0000
    python generate_commentary.py --shard commentary_review/shard_0000 --json-dir data/json/Soccer_Data_Json
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--shard", required=True, help="Path to shard directory")
parser.add_argument("--json-dir", default="data/json/Soccer_Data_Json",
                    help="Root of raw JSON annotations")
parser.add_argument("--out", default=None,
                    help="Output JSONL path (default: <shard>/improved_commentary.jsonl)")
args = parser.parse_args()

shard_dir = Path(args.shard)
json_dir = Path(args.json_dir)
out_path = Path(args.out) if args.out else shard_dir / "improved_commentary.jsonl"

# ---------------------------------------------------------------------------
# FACT DICTIONARIES — one per event type, covering ALL 26 labels
# ---------------------------------------------------------------------------
EVENT_FACTS = {
    "goal": {
        "plain_event": "a goal has been scored",
        "rule_context": "the ball has crossed the goal line and the score changes",
        "next_state": "play will restart from the center circle",
        "tactical_context": "the goal alters the match dynamic and forces the conceding team to push forward",
        "visual_state": "players celebrate or react as the ball enters the net",
    },
    "corner": {
        "plain_event": "the attacking team has won a corner",
        "rule_context": "the ball went out over the goal line after touching a defender",
        "next_state": "play restarts from the corner flag",
        "tactical_context": "this becomes a set-piece chance with runners attacking the box",
        "visual_state": "players gather in the penalty area while the taker moves toward the corner flag",
    },
    "foul_with_no_card": {
        "plain_event": "the referee has called a foul",
        "rule_context": "play stops because of illegal contact or a challenge",
        "next_state": "the other team gets a free kick",
        "tactical_context": "the foul breaks the rhythm of the attack or stops a transition",
        "visual_state": "players slow down as the referee signals the direction of the restart",
    },
    "yellow_card": {
        "plain_event": "the referee has shown a yellow card",
        "rule_context": "a yellow card is an official warning to a player",
        "next_state": "the booked player must be careful for the rest of the match",
        "tactical_context": "the caution can change how aggressively that player challenges",
        "visual_state": "the referee holds the yellow card above their head near the offending player",
    },
    "red_card": {
        "plain_event": "a player has been shown a red card",
        "rule_context": "a red card means the player must leave the pitch immediately and the team plays with one fewer player",
        "next_state": "the team is reduced to fewer players for the remainder of the match",
        "tactical_context": "the dismissal forces a complete tactical reorganization, usually with a more defensive shape",
        "visual_state": "the referee holds up the red card while the player leaves the field",
    },
    "second_yellow_card": {
        "plain_event": "a player has received a second yellow card",
        "rule_context": "two yellow cards in the same match result in a red card and an automatic sending off",
        "next_state": "the player must leave the pitch and the team plays short-handed",
        "tactical_context": "the team must reorganize with one fewer player, often sacrificing an attacker for defensive cover",
        "visual_state": "the referee shows the second yellow followed by a red card",
    },
    "off_side": {
        "plain_event": "the attacking player has been called offside",
        "rule_context": "the attacker was ahead of the allowed position when the pass was played",
        "next_state": "the defending team gets the restart",
        "tactical_context": "the offside call nullifies the attacking move and relieves defensive pressure",
        "visual_state": "the assistant referee raises the flag to signal the offside position",
    },
    "saved_by_goal-keeper": {
        "plain_event": "the goalkeeper has made a save",
        "rule_context": "the shot was on target but did not become a goal",
        "next_state": "the ball may stay in play, go out for a corner, or be held by the goalkeeper",
        "tactical_context": "the attack creates a real chance but the keeper prevents the finish",
        "visual_state": "the goalkeeper moves toward the ball and stops it near the goal area",
    },
    "shot_off_target": {
        "plain_event": "a shot has missed the goal",
        "rule_context": "the attempt does not force the goalkeeper to make a save",
        "next_state": "play usually restarts with a goal kick if no defender touched it last",
        "tactical_context": "the chance ends without testing the goalkeeper, often from pressure or a difficult angle",
        "visual_state": "the ball travels wide or over the goal and players begin to reset",
    },
    "substitution": {
        "plain_event": "a substitution is being made",
        "rule_context": "one player leaves and a teammate enters",
        "next_state": "the team may be changing energy, roles, or tactics",
        "tactical_context": "the change may alter the shape, pressing, or attacking outlet",
        "visual_state": "activity shifts to the sideline as one player comes off and another prepares to enter",
    },
    "free_kick": {
        "plain_event": "a free kick has been awarded",
        "rule_context": "the fouled team gets an uncontested kick from the spot of the foul",
        "next_state": "play restarts once the ball is placed and the wall is set",
        "tactical_context": "depending on the location, this could be a direct shooting chance or a build-up opportunity",
        "visual_state": "the attacking team lines up while defenders form a wall",
    },
    "penalty": {
        "plain_event": "a penalty kick has been awarded",
        "rule_context": "a foul was committed inside the penalty area, giving the attacking team a direct shot from the spot",
        "next_state": "all other players must stay outside the box until the kick is taken",
        "tactical_context": "penalties are converted roughly 75-80% of the time, making this a high-probability scoring chance",
        "visual_state": "the penalty taker places the ball on the spot while the goalkeeper stands on the line",
    },
    "clearance": {
        "plain_event": "a defender has cleared the ball",
        "rule_context": "the defending player kicks the ball away from their goal area to relieve pressure",
        "next_state": "the ball goes upfield and both teams contest for possession",
        "tactical_context": "the clearance breaks the attacking momentum and gives the defense time to reorganize",
        "visual_state": "a defender strikes the ball away from the danger zone, often under pressure",
    },
    "lead_to_corner": {
        "plain_event": "the ball has been deflected out for a corner",
        "rule_context": "a defending player touched the ball last before it crossed the goal line",
        "next_state": "the attacking team will take a corner kick",
        "tactical_context": "the deflection prevents an immediate chance but concedes a set-piece opportunity",
        "visual_state": "the ball deflects off a defender and crosses the goal line near the corner flag",
    },
    "ball_possession": {
        "plain_event": "a team is controlling possession",
        "rule_context": "the team with the ball is passing and retaining it without losing control",
        "next_state": "the attacking team looks for an opening or builds up play",
        "tactical_context": "sustained possession pins the opposition back and creates space for movement",
        "visual_state": "players pass the ball among themselves while the opposition tries to press and win it back",
    },
    "ball_out_of_play": {
        "plain_event": "the ball has gone out of play",
        "rule_context": "the ball crossed the touchline or goal line and play is temporarily stopped",
        "next_state": "play restarts with a throw-in, goal kick, or corner depending on which team touched it last",
        "tactical_context": "the stoppage gives both teams a moment to reposition",
        "visual_state": "the ball crosses the boundary line and a player retrieves it for the restart",
    },
    "throw_in": {
        "plain_event": "a throw-in has been awarded",
        "rule_context": "the ball crossed the touchline and the team that did not touch it last gets to throw it in",
        "next_state": "the player must throw the ball with both hands from behind their head",
        "tactical_context": "a throw-in near the opponent's box can become a dangerous attacking set piece",
        "visual_state": "a player holds the ball at the sideline and throws it to a teammate",
    },
    "injury": {
        "plain_event": "play has stopped for an injury",
        "rule_context": "the referee stops play when a player appears to be seriously hurt",
        "next_state": "medical staff come onto the pitch to assess the player",
        "tactical_context": "the stoppage breaks the flow of the game and may lead to a substitution",
        "visual_state": "a player is on the ground receiving treatment while teammates and opponents wait",
    },
    "foul_lead_to_penalty": {
        "plain_event": "a foul inside the penalty area has been called",
        "rule_context": "the defending player committed an illegal challenge inside their own box",
        "next_state": "a penalty kick will be awarded to the attacking team",
        "tactical_context": "this is a critical error by the defense, giving the opponent a high-probability scoring chance",
        "visual_state": "the referee points to the penalty spot after the foul in the box",
    },
    "var": {
        "plain_event": "the video assistant referee is reviewing a decision",
        "rule_context": "VAR checks are used for goals, penalties, red cards, and mistaken identity",
        "next_state": "the on-field referee may go to the monitor or accept the VAR recommendation",
        "tactical_context": "the review pause creates uncertainty and can reverse a key moment in the match",
        "visual_state": "play is paused while the referee signals the VAR check with a rectangle gesture",
    },
    "start_of_half_game": {
        "plain_event": "the half is about to begin",
        "rule_context": "the referee blows the whistle to start play from the center circle",
        "next_state": "one team kicks off and the match is underway",
        "tactical_context": "teams begin in their planned formation and tactical setup",
        "visual_state": "players take their positions across the pitch as the referee prepares to start",
    },
    "end_of_half_game": {
        "plain_event": "the referee has blown the whistle to end this half",
        "rule_context": "no more play will happen in this half",
        "next_state": "the players leave the pitch for the break or the match is over",
        "tactical_context": "the manager will use the interval to make tactical adjustments",
        "visual_state": "players walk off the pitch toward the tunnel or shake hands",
    },
    "show_added_time": {
        "plain_event": "the fourth official is displaying the added time",
        "rule_context": "extra minutes are added to compensate for stoppages during the half",
        "next_state": "play continues for the indicated number of additional minutes",
        "tactical_context": "the trailing team pushes forward urgently while the leading team tries to manage the clock",
        "visual_state": "the electronic board on the sideline shows the number of added minutes",
    },
    "statistics_and_summary": {
        "plain_event": "match statistics are being shown",
        "rule_context": "a broadcast graphic displays possession, shots, and other match data",
        "next_state": "play will resume shortly or the graphic appears during a break",
        "tactical_context": "the statistics reveal which team has been dominant and where the match balance lies",
        "visual_state": "a graphic overlay appears on screen showing match statistics",
    },
    "unknown": {
        "plain_event": "an event has occurred",
        "rule_context": "the specific event type could not be determined from the annotations",
        "next_state": "play continues",
        "tactical_context": "the flow of the game continues",
        "visual_state": "the broadcast continues showing the match action",
    },
    "wcl-icon-settings-info-rounded": {
        "plain_event": "a broadcast graphic or overlay is being displayed",
        "rule_context": "this is a technical broadcast element, not a match event",
        "next_state": "the overlay will clear and the match view returns",
        "tactical_context": "no tactical significance — this is a broadcast production element",
        "visual_state": "a graphical overlay or icon appears on the broadcast feed",
    },
}

# ---------------------------------------------------------------------------
# PARAPHRASE POOLS — surface variation for each audience
# ---------------------------------------------------------------------------
BEGINNER_OPENERS = [
    "{event}.",
    "Here, {event}.",
    "In this moment, {event}.",
    "What just happened is that {event}.",
    "Right now, {event}.",
    "At this point, {event}.",
    "We can see that {event}.",
]

BEGINNER_BRIDGES = [
    "In simple terms, {context}.",
    "What this means is that {context}.",
    "Basically, {context}.",
    "To explain, {context}.",
    "This means {context}.",
    "Put simply, {context}.",
]

BEGINNER_CLOSERS = [
    "The next thing to watch is that {next}.",
    "What happens next: {next}.",
    "Now, {next}.",
    "From here, {next}.",
    "Going forward, {next}.",
    "Keep an eye out — {next}.",
]

EXPERT_OPENERS = [
    "{event}.",
    "Key moment — {event}.",
    "The decisive action: {event}.",
    "Tactically significant — {event}.",
    "{event}, a pivotal moment.",
]

EXPERT_BRIDGES = [
    "Tactically, {context}.",
    "The implication: {context}.",
    "From a tactical standpoint, {context}.",
    "This matters because {context}.",
    "Strategically, {context}.",
]

EXPERT_CLOSERS = [
    "The immediate consequence: {next}.",
    "What follows — {next}.",
    "This sets up a situation where {next}.",
    "Next, {next}.",
    "The knock-on effect: {next}.",
]

LOW_VISION_OPENERS = [
    "{event}.",
    "What's happening: {event}.",
    "Right now, {event}.",
    "On your screen, {event}.",
]

LOW_VISION_VISUAL = [
    "The broadcast shows {visual}.",
    "Visually, {visual}.",
    "On screen, {visual}.",
    "The camera shows {visual}.",
]

LOW_VISION_CLOSERS = [
    "The important change is that {next}.",
    "What to listen for next: {next}.",
    "Coming up, {next}.",
    "Next, {next}.",
]


# ---------------------------------------------------------------------------
# Entity resolution — load match data from raw JSON
# ---------------------------------------------------------------------------
def find_raw_json(game_label: str) -> dict | None:
    """Find and load the raw JSON for a game from the local dataset."""
    # game_label format: "2017-10-21_manchester-city-burnley-fc-premier-league"
    # Try each split and league
    for split in ["train", "test", "valid"]:
        split_dir = json_dir / split
        if not split_dir.exists():
            continue
        for league_dir in split_dir.iterdir():
            if not league_dir.is_dir():
                continue
            game_dir = league_dir / game_label
            if game_dir.exists():
                jsons = list(game_dir.glob("*.json"))
                if jsons:
                    with open(jsons[0], encoding="utf-8") as f:
                        return json.load(f)
    return None


def build_entity_context(raw_json: dict, event_label: str,
                         timestamp: str, half: int) -> dict:
    """Extract entity details relevant to this specific event."""
    ctx = {}
    mi = raw_json.get("match_info", {})
    ctx["home_team"] = mi.get("home_team", "")
    ctx["away_team"] = mi.get("away_team", "")
    ctx["score"] = mi.get("score", "")
    ctx["venue"] = mi.get("venue", "")

    ref = raw_json.get("referee_data", {})
    ctx["referee"] = ref.get("detected_name", ref.get("first_name", ""))

    # Find the matching comment to get player names
    ctx["player"] = ""
    ctx["team"] = ""
    ctx["original_text"] = ""
    for c in raw_json.get("comments", []):
        c_ts = c.get("time_stamp", "")
        c_half = c.get("half", 0)
        c_type = c.get("comments_type", "").lower().replace(" ", "_")
        event_norm = event_label.lower().replace(" ", "_")

        if c_ts == timestamp and c_half == half and c_type == event_norm:
            ctx["original_text"] = c.get("comments_text", "")
            # Extract first player name from original text
            text = ctx["original_text"]
            # Pattern: "PlayerName (TeamName)"
            m = re.search(r'^([A-Z][a-zà-ü\'-]+(?: [A-Z][a-zà-ü\'-]+)*)\s*\(([^)]+)\)', text)
            if m:
                ctx["player"] = m.group(1)
                ctx["team"] = m.group(2)
            break

    return ctx


def format_event_with_entities(facts: dict, entities: dict,
                               event_label: str) -> dict:
    """Enhance fact strings with entity detail when available."""
    enhanced = dict(facts)
    player = entities.get("player", "")
    team = entities.get("team", "")
    home = entities.get("home_team", "")
    away = entities.get("away_team", "")
    score = entities.get("score", "")
    referee = entities.get("referee", "")

    label = event_label.lower()

    if player and team:
        if label == "goal":
            enhanced["plain_event"] = f"{player} ({team}) has scored a goal"
        elif label == "yellow_card":
            enhanced["plain_event"] = f"{player} ({team}) has been shown a yellow card"
        elif label == "red_card":
            enhanced["plain_event"] = f"{player} ({team}) has been shown a red card"
        elif label == "second_yellow_card":
            enhanced["plain_event"] = f"{player} ({team}) has received a second yellow card"
        elif label in ("foul_with_no_card", "foul_lead_to_penalty"):
            enhanced["plain_event"] = f"{player} ({team}) has committed a foul"
        elif label == "off_side":
            enhanced["plain_event"] = f"{player} ({team}) has been caught offside"
        elif label == "saved_by_goal-keeper":
            enhanced["plain_event"] = f"{player} ({team}) has made a save"
        elif label == "shot_off_target":
            enhanced["plain_event"] = f"{player} ({team}) has fired a shot off target"
        elif label == "substitution":
            enhanced["plain_event"] = f"a substitution is being made for {team}"
        elif label == "clearance":
            enhanced["plain_event"] = f"{player} ({team}) clears the ball"
        elif label == "corner":
            enhanced["plain_event"] = f"{team} have won a corner"
        elif label == "free_kick":
            enhanced["plain_event"] = f"{team} have been awarded a free kick"
        elif label == "injury":
            enhanced["plain_event"] = f"{player} ({team}) is down with an injury"
        elif label == "penalty":
            enhanced["plain_event"] = f"a penalty has been awarded — {player} ({team}) to take it"

    if score and home and away:
        enhanced["score_context"] = f"The score is {home} {score} {away}"
    if referee and label in ("foul_with_no_card", "yellow_card", "red_card",
                             "penalty", "foul_lead_to_penalty", "var",
                             "second_yellow_card"):
        enhanced["plain_event"] = enhanced["plain_event"].rstrip(".") + f", as signaled by referee {referee}"

    return enhanced


# ---------------------------------------------------------------------------
# Grammar helpers
# ---------------------------------------------------------------------------
def a_or_an(word: str) -> str:
    """Return 'an' if word starts with a vowel sound, else 'a'."""
    if not word:
        return "a"
    return "an" if word[0].lower() in "aeiou" else "a"


def format_label(label: str) -> str:
    """Convert event_label to readable text."""
    return label.replace("_", " ").replace("-", " ")


# ---------------------------------------------------------------------------
# Commentary generation
# ---------------------------------------------------------------------------
def generate_beginner(facts: dict) -> str:
    event = facts["plain_event"]
    # Fix a/an
    if event.startswith("a ") and event[2:3] in "aeiou":
        event = "an " + event[2:]

    opener = random.choice(BEGINNER_OPENERS).format(event=event.capitalize() if not event[0].isupper() else event)
    bridge = random.choice(BEGINNER_BRIDGES).format(context=facts["rule_context"])
    closer = random.choice(BEGINNER_CLOSERS).format(next=facts["next_state"])
    return f"{opener} {bridge} {closer}"


def generate_expert(facts: dict) -> str:
    event = facts["plain_event"]
    if event.startswith("a ") and event[2:3] in "aeiou":
        event = "an " + event[2:]

    opener = random.choice(EXPERT_OPENERS).format(event=event.capitalize() if not event[0].isupper() else event)
    bridge = random.choice(EXPERT_BRIDGES).format(context=facts["tactical_context"])
    closer = random.choice(EXPERT_CLOSERS).format(next=facts["next_state"])

    parts = [opener, bridge, closer]
    if "score_context" in facts:
        parts.append(facts["score_context"] + ".")
    return " ".join(parts)


def generate_low_vision(facts: dict, visual_cues: dict | None) -> str:
    event = facts["plain_event"]
    if event.startswith("a ") and event[2:3] in "aeiou":
        event = "an " + event[2:]

    opener = random.choice(LOW_VISION_OPENERS).format(event=event.capitalize() if not event[0].isupper() else event)

    # Use actual visual cues if available, otherwise use facts
    if visual_cues:
        cam = visual_cues.get("camera", {})
        scene = visual_cues.get("scene", {})
        shot = cam.get("shot_type", "").replace("_", " ")
        motion = scene.get("motion_intensity", "")
        if shot and motion:
            visual_desc = f"a {shot} with {motion} motion"
        elif shot:
            visual_desc = f"a {shot}"
        else:
            visual_desc = facts["visual_state"]
    else:
        visual_desc = facts["visual_state"]

    visual = random.choice(LOW_VISION_VISUAL).format(visual=visual_desc)
    closer = random.choice(LOW_VISION_CLOSERS).format(next=facts["next_state"])
    return f"{opener} {visual} {closer}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
print(f"Shard:    {shard_dir}")
print(f"JSON dir: {json_dir}")
print(f"Output:   {out_path}")

# Load windows
windows_path = shard_dir / "windows.jsonl"
if not windows_path.exists():
    print(f"ERROR: {windows_path} not found")
    raise SystemExit(1)

windows = []
with open(windows_path) as f:
    for line in f:
        windows.append(json.loads(line))
print(f"Windows:  {len(windows)}")

# Load visual cues
visual_cues = {}
cues_path = shard_dir / "basic_visual_cues.jsonl"
if cues_path.exists():
    with open(cues_path) as f:
        for line in f:
            rec = json.loads(line)
            visual_cues[rec["window_id"]] = rec.get("visual_facts", {})
    print(f"Visual cues: {len(visual_cues)}")

# Cache raw JSON per game
json_cache = {}


def get_raw_json(game_label):
    if game_label not in json_cache:
        json_cache[game_label] = find_raw_json(game_label)
    return json_cache[game_label]


# Load original commentary for comparison
original = {}
orig_path = shard_dir / "natural_commentary_with_visual_facts.jsonl"
if orig_path.exists():
    with open(orig_path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["window_id"], rec["audience"])
            original[key] = rec["commentary"]
    print(f"Original commentary: {len(original)} records")

# Generate
print(f"\nGenerating improved commentary...")
records = []
entity_hits = 0
entity_misses = 0

for w in windows:
    wid = w["window_id"]
    game_label = w["game_label"]
    event_label = w["event_label"]
    half = w.get("half", 1)
    timestamp = w.get("timestamp", "")

    # Get base facts
    facts = EVENT_FACTS.get(event_label, EVENT_FACTS["unknown"]).copy()

    # Try to enhance with entity detail
    raw = get_raw_json(game_label)
    if raw:
        entities = build_entity_context(raw, event_label, timestamp, half)
        facts = format_event_with_entities(facts, entities, event_label)
        if entities.get("player"):
            entity_hits += 1
        else:
            entity_misses += 1
    else:
        entities = {}
        entity_misses += 1

    # Get visual cues for this window
    vcues = visual_cues.get(wid)

    for audience in ["beginner", "expert", "low_vision"]:
        if audience == "beginner":
            text = generate_beginner(facts)
        elif audience == "expert":
            text = generate_expert(facts)
        else:
            text = generate_low_vision(facts, vcues)

        orig_key = (wid, audience)
        orig_text = original.get(orig_key, "")

        records.append({
            "window_id": wid,
            "game_label": game_label,
            "event_label": event_label,
            "audience": audience,
            "commentary": text,
            "original_commentary": orig_text,
            "has_entity_detail": bool(entities.get("player")),
            "home_team": entities.get("home_team", ""),
            "away_team": entities.get("away_team", ""),
        })

# Write output
with open(out_path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Stats
n_windows = len(windows)
n_records = len(records)
unique_beginner = len(set(r["commentary"] for r in records if r["audience"] == "beginner"))
unique_expert = len(set(r["commentary"] for r in records if r["audience"] == "expert"))
unique_lv = len(set(r["commentary"] for r in records if r["audience"] == "low_vision"))

print(f"\n{'='*60}")
print(f"GENERATION COMPLETE")
print(f"{'='*60}")
print(f"Windows processed: {n_windows}")
print(f"Records written:   {n_records}")
print(f"Entity detail found: {entity_hits}/{n_windows} ({entity_hits/max(n_windows,1)*100:.1f}%)")
print(f"Entity detail missing: {entity_misses}/{n_windows}")
print(f"")
print(f"Unique strings (IMPROVED):")
print(f"  Beginner:   {unique_beginner}")
print(f"  Expert:     {unique_expert}")
print(f"  Low vision: {unique_lv}")

# Compare with original if available
if original:
    orig_beg = len(set(v for (_, a), v in original.items() if a == "beginner"))
    orig_exp = len(set(v for (_, a), v in original.items() if a == "expert"))
    orig_lv = len(set(v for (_, a), v in original.items() if a == "low_vision"))
    print(f"")
    print(f"Unique strings (ORIGINAL):")
    print(f"  Beginner:   {orig_beg}")
    print(f"  Expert:     {orig_exp}")
    print(f"  Low vision: {orig_lv}")
    print(f"")
    print(f"Improvement factor:")
    print(f"  Beginner:   {unique_beginner/max(orig_beg,1):.1f}x more diverse")
    print(f"  Expert:     {unique_expert/max(orig_exp,1):.1f}x more diverse")
    print(f"  Low vision: {unique_lv/max(orig_lv,1):.1f}x more diverse")

print(f"\nOutput -> {out_path}")
print("Done.")
