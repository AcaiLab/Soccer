"""
analyze_events.py
-----------------
Reads all JSON files from a project folder (one per match) and produces:
  1. A per-match event frequency table (printed to console)
  2. A combined "Clips per label" bar chart across all matches (saved as PNG)

Usage:
    python analyze_events.py --project-folder /path/to/project --out analysis_output
"""

import json
import re
import pathlib
import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Analyse event frequencies across soccer matches.")
parser.add_argument("--project-folder", required=True, help="Path to project folder with match subfolders")
parser.add_argument("--out", default="analysis_output", help="Output folder for charts and reports")
args = parser.parse_args()

project_folder = pathlib.Path(args.project_folder)
out_dir        = pathlib.Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def sanitize_event_name(event: str) -> str:
    name = event.strip().lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[\s]+', '_', name)
    return name or "unknown"

# ── collect events ───────────────────────────────────────────────────────────
match_folders = sorted([f for f in project_folder.iterdir() if f.is_dir()])
if not match_folders:
    print(f"ERROR: No match folders found in {project_folder}")
    exit(1)

print(f"Found {len(match_folders)} match folder(s)\n")

combined_counter = Counter()           # across all matches
per_match_counters = {}                # {match_name: Counter}

for match_folder in match_folders:
    json_files = list(match_folder.glob("*.json"))
    if not json_files:
        print(f"  WARNING: No JSON in {match_folder.name}, skipping.")
        continue

    json_path = json_files[0]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counter = Counter()
    for comment in data.get("comments", []):
        ts    = comment.get("time_stamp", "")
        event = comment.get("comments_type", "unknown") or "unknown"
        if ts:  # only count entries that have a timestamp
            counter[sanitize_event_name(event)] += 1

    per_match_counters[match_folder.name] = counter
    combined_counter.update(counter)
    print(f"  {match_folder.name}: {sum(counter.values())} events, {len(counter)} unique labels")

# ── print summary table ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("COMBINED EVENT FREQUENCY (all matches)")
print("="*60)
print(f"{'Event':<35} {'Count':>6}")
print("-"*45)
for event, count in sorted(combined_counter.items(), key=lambda x: -x[1]):
    print(f"  {event:<33} {count:>6}")
print("-"*45)
print(f"  {'TOTAL':<33} {sum(combined_counter.values()):>6}")

# ── per-match table ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PER-MATCH BREAKDOWN")
print("="*60)
all_labels = sorted(combined_counter.keys())
header = f"{'Event':<35}" + "".join(f"{n[:10]:>12}" for n in sorted(per_match_counters))
print(header)
print("-" * len(header))
for label in all_labels:
    row = f"  {label:<33}"
    for match_name in sorted(per_match_counters):
        row += f"{per_match_counters[match_name].get(label, 0):>12}"
    print(row)

# ── bar chart: combined ───────────────────────────────────────────────────────
labels  = sorted(combined_counter.keys())
counts  = [combined_counter[l] for l in labels]

fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 6))
bars = ax.bar(labels, counts, color="#4C72B0", edgecolor="white", linewidth=0.6)

ax.set_title("Clips per label (all matches combined)", fontsize=14, fontweight="bold")
ax.set_xlabel("Event label", fontsize=11)
ax.set_ylabel("# Clips", fontsize=11)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# value labels on bars
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            str(val), ha="center", va="bottom", fontsize=8)

plt.tight_layout()
chart_path = out_dir / "clips_per_label_combined.png"
fig.savefig(str(chart_path), dpi=150)
plt.close(fig)
print(f"\nCombined bar chart saved → {chart_path}")

# ── bar chart: per match ──────────────────────────────────────────────────────
for match_name, counter in sorted(per_match_counters.items()):
    m_labels = sorted(counter.keys())
    m_counts = [counter[l] for l in m_labels]

    fig, ax = plt.subplots(figsize=(max(8, len(m_labels) * 0.7), 5))
    bars = ax.bar(m_labels, m_counts, color="#55A868", edgecolor="white", linewidth=0.6)

    ax.set_title(f"Clips per label — {match_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Event label", fontsize=10)
    ax.set_ylabel("# Clips", fontsize=10)
    ax.set_xticks(range(len(m_labels)))
    ax.set_xticklabels(m_labels, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for bar, val in zip(bars, m_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    safe_name = re.sub(r'[^\w\-]', '_', match_name)
    path = out_dir / f"clips_per_label_{safe_name}.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"Per-match chart saved  → {path}")

print(f"\nAll analysis outputs saved to: {out_dir}")
