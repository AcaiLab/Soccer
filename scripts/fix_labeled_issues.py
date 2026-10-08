"""
Apply the improved commentary generator to a labeled-issues CSV and print a
side-by-side comparison showing the fixes.

Usage:
    python scripts/fix_labeled_issues.py --issues matched_commentary_strict_train_issues_labeled.csv \\
        --out improved_commentary_comparison.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from generate_commentary import (
    EVENT_FACTS,
    build_entity_context,
    find_raw_json,
    format_event_with_entities,
    generate_beginner,
    generate_expert,
    generate_low_vision,
)


def print_examples(title, rows, columns, limit):
    print(f"\n{title}")
    seen = set()
    for _, r in rows.iterrows():
        if r["event_label"] in seen:
            continue
        seen.add(r["event_label"])
        if len(seen) > limit:
            break
        print(f'\n  Event: {r["event_label"]}  |  Game: {r["game_label"][:55]}')
        for name, col in columns:
            print(f"  {name}: {str(r[col])[:150]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare original and improved commentary on labeled issues.")
    parser.add_argument("--issues", required=True, help="labeled issues CSV (one row per window)")
    parser.add_argument("--json-dir", default="data/json/Soccer_Data_Json", help="Root of raw JSON annotations")
    parser.add_argument("--out", default="improved_commentary_comparison.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    df = pd.read_csv(args.issues)
    json_dir = Path(args.json_dir)
    json_cache = {}

    results = []
    entity_hits = 0
    for _, row in df.iterrows():
        game_label = row["game_label"]
        event_label = row["generated_event_label"]
        facts = EVENT_FACTS.get(event_label, EVENT_FACTS.get("unknown", {})).copy()

        if game_label not in json_cache:
            json_cache[game_label] = find_raw_json(json_dir, game_label)
        raw = json_cache[game_label]
        entities = {}
        if raw:
            entities = build_entity_context(raw, event_label, str(row["generated_time"]), int(row["generated_half"]))
            facts = format_event_with_entities(facts, entities, event_label)
            entity_hits += bool(entities.get("player"))

        results.append({
            "window_id": row["window_id"],
            "game_label": game_label,
            "event_label": event_label,
            "improved_beginner": generate_beginner(facts),
            "improved_expert": generate_expert(facts),
            "improved_low_vision": generate_low_vision(facts, None),
            "original_beginner": row["beginner_commentary"],
            "original_expert": row["expert_commentary"],
            "original_low_vision": row["low_vision_commentary"],
            "has_entity": bool(entities.get("player")),
            "orig_has_jargon": row["has_jargon_leak"],
            "orig_is_fallback": row["is_fallback_junk_label"],
        })

    rdf = pd.DataFrame(results)
    n = len(rdf)
    print("=" * 70)
    print(f"BEFORE vs AFTER - {Path(args.issues).name}")
    print("=" * 70)
    print(f"Total events: {n}")
    print(f"Entity detail matched: {entity_hits}/{n} ({entity_hits / max(n, 1) * 100:.1f}%)\n")

    for aud in ["beginner", "expert", "low_vision"]:
        orig_u, impr_u = rdf[f"original_{aud}"].nunique(), rdf[f"improved_{aud}"].nunique()
        print(f"{aud:12s}  original: {orig_u:6d} unique  ->  improved: {impr_u:6d} unique  "
              f"({impr_u / max(orig_u, 1):.1f}x)")

    checks = [
        ("Expert jargon leak", "expert", r"retrieval memory|close example", True),
        ("Beginner fallback", "beginner", r"occurs\. In simple terms, this changes the current phase", False),
        ("Grammar errors (a/an)", "beginner", r"\bA injury\b|\bA off_side\b|\ba var occurs\b|\ba unknown occurs\b",
         False),
    ]
    print()
    for name, aud, pattern, ignore_case in checks:
        orig = rdf[f"original_{aud}"].str.contains(pattern, case=not ignore_case, na=False).sum()
        impr = rdf[f"improved_{aud}"].str.contains(pattern, case=not ignore_case, na=False).sum()
        print(f"{name:<22s} original={orig} ({orig / max(n, 1) * 100:.1f}%)  "
              f"improved={impr} ({impr / max(n, 1) * 100:.0f}%)")

    rdf.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")

    print_examples("SIDE-BY-SIDE EXAMPLES (with entity detail)", rdf[rdf["has_entity"]],
                   [("BEG ORIG", "original_beginner"), ("BEG IMPR", "improved_beginner"),
                    ("EXP ORIG", "original_expert"), ("EXP IMPR", "improved_expert")], limit=6)
    print_examples("--- FALLBACK FIX EXAMPLES ---", rdf[rdf["orig_is_fallback"].astype(bool)],
                   [("ORIG", "original_beginner"), ("IMPR", "improved_beginner")], limit=4)


if __name__ == "__main__":
    main()
