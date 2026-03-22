import json
import pathlib
import argparse
from collections import Counter, defaultdict

print("Script started...")

# ----------------------------
# Command line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Analyze event label distribution across matches.")
parser.add_argument("--project-folder", required=True, help="Path to folder containing match subfolders")
args = parser.parse_args()

project_folder = pathlib.Path(args.project_folder)

if not project_folder.exists():
    print(f"ERROR: Folder does not exist: {project_folder}")
    exit(1)

# ----------------------------
# Categories to EXCLUDE
# ----------------------------
EXCLUDED_TYPES = {
    "start of half game",
    "end of half game",
    "statistics and summary",
    "show added time"
}

global_counter = Counter()
per_match_stats = defaultdict(Counter)

match_folders = [f for f in project_folder.iterdir() if f.is_dir()]

if not match_folders:
    print("No match folders found.")
    exit(1)

print(f"\nFound {len(match_folders)} matches.\n")

# ----------------------------
# Process each match
# ----------------------------
for match_folder in sorted(match_folders):

    json_files = list(match_folder.glob("*.json"))
    if not json_files:
        print(f"Skipping {match_folder.name} (no JSON found)")
        continue

    json_path = json_files[0]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    match_counter = Counter()

    # Track goals that might be overturned
    pending_goal_index = None

    comments = data.get("comments", [])

    for i, comment in enumerate(comments):

        half = comment.get("half", 0)
        timestamp = comment.get("time_stamp", "")
        event_type = comment.get("comments_type", "").strip().lower()
        text = comment.get("comments_text", "").lower()

        # ----------------------------
        # Filtering rules
        # ----------------------------
        if half not in [1, 2]:
            continue
        if not timestamp:
            continue
        if event_type in EXCLUDED_TYPES:
            continue

        # ----------------------------
        # Handle goal logic carefully
        # ----------------------------
        if event_type == "goal":
            # Temporarily store this goal (might get overturned)
            pending_goal_index = i
            continue

        # ----------------------------
        # Detect VAR overturn
        # ----------------------------
        if event_type == "var":
            if "no goal" in text and pending_goal_index is not None:
                # Cancel the previous goal
                pending_goal_index = None
            continue

        # ----------------------------
        # If we reach here and there was a pending goal,
        # it means the goal was valid (no VAR cancellation)
        # ----------------------------
        if pending_goal_index is not None:
            match_counter["goal"] += 1
            global_counter["goal"] += 1
            pending_goal_index = None

        # Count normal events
        match_counter[event_type] += 1
        global_counter[event_type] += 1

    # If last event in match was a goal with no VAR after it
    if pending_goal_index is not None:
        match_counter["goal"] += 1
        global_counter["goal"] += 1

    per_match_stats[match_folder.name] = match_counter

# ----------------------------
# Print Global Summary
# ----------------------------
print("=" * 60)
print("GLOBAL EVENT DISTRIBUTION")
print("=" * 60)

total_events = sum(global_counter.values())
print(f"Total valid events across all matches: {total_events}\n")

for event, count in global_counter.most_common():
    percentage = (count / total_events) * 100 if total_events > 0 else 0
    print(f"{event:25s} : {count:4d} ({percentage:.2f}%)")

# ----------------------------
# Per-Match Summary
# ----------------------------
print("\n" + "=" * 60)
print("PER MATCH SUMMARY")
print("=" * 60)

for match_name, counter in per_match_stats.items():
    match_total = sum(counter.values())
    print(f"\n{match_name}")
    print(f"  Total Events: {match_total}")
    for event, count in counter.most_common():
        print(f"    {event:25s} : {count}")

# ----------------------------
# Averages Per Game
# ----------------------------
print("\n" + "=" * 60)
print("AVERAGE EVENTS PER GAME")
print("=" * 60)

num_matches = len(per_match_stats)

for event, count in global_counter.most_common():
    avg = count / num_matches if num_matches > 0 else 0
    print(f"{event:25s} : {avg:.2f}")

# ----------------------------
# Bar Plot (Global Distribution)
# ----------------------------
import matplotlib.pyplot as plt

if global_counter:
    events = [event for event, _ in global_counter.most_common()]
    counts = [global_counter[event] for event in events]

    plt.figure(figsize=(12, 6))
    plt.bar(events, counts)

    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Number of Occurrences")
    plt.title("Event Distribution Across All Matches")

    plt.tight_layout()
    plt.show()

