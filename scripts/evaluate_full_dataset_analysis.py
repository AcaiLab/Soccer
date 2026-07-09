import argparse
import csv
import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from commentary_eval_utils import (
    AUDIENCE_KEYWORDS,
    MOTION_WORDS,
    SPATIAL_WORDS,
    audience_style_score,
    count_matches,
    event_consistency,
    low_vision_spatial_score,
    normalize,
    slot_coverage,
    vague_visual_flags,
    word_count,
)
from pipeline_utils import ensure_dir, repo_root, write_json


TACTICAL_WORDS = [
    "tactical",
    "shape",
    "spacing",
    "phase",
    "press",
    "overload",
    "transition",
    "set-piece",
    "set piece",
    "defensive",
    "attacking",
    "possession",
    "rhythm",
]

EXPLANATORY_WORDS = [
    "means",
    "simple",
    "because",
    "rule",
    "watch",
    "next",
    "understand",
    "in simple terms",
]

VISUAL_WORDS = [
    "visually",
    "visible",
    "camera",
    "view",
    "wide",
    "field",
    "players",
    "ball",
    "referee",
    "motion",
    "moving",
    "scene",
    "position",
    "positions",
    "gather",
    "reset",
]

EVENTS_FOR_EXAMPLES = [
    "goal",
    "corner",
    "yellow_card",
    "substitution",
    "penalty",
    "saved_by_goal-keeper",
    "shot_off_target",
    "off_side",
    "injury",
]


def iter_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def safe_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else None


def pct(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 3)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9_'-]+\b", normalize(text)))


def jaccard(a: str, b: str) -> float:
    aa, bb = token_set(a), token_set(b)
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / max(1, len(aa | bb))


def ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    toks = re.findall(r"\b[a-z0-9_'-]+\b", normalize(text))
    return [tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]


def start_phrase(text: str, words: int = 4) -> str:
    toks = re.findall(r"\b[a-z0-9_'-]+\b", normalize(text))
    return " ".join(toks[:words])


def heuristic_audience(text: str) -> str:
    scores = {aud: count_matches(text, terms) for aud, terms in AUDIENCE_KEYWORDS.items()}
    return max(scores.items(), key=lambda item: item[1])[0]


def visual_grounding_row(row: dict[str, Any]) -> dict[str, float | int]:
    text = row.get("commentary") or ""
    t = normalize(text)
    visual_terms = count_matches(text, VISUAL_WORDS + SPATIAL_WORDS + MOTION_WORDS)
    mentions_wide_field = int(
        bool(row.get("field_visible")) and any(term in t for term in ["field", "wide", "view", "camera"])
    )
    motion = row.get("motion_intensity")
    mentions_motion = int(
        motion in {"medium", "high"} and count_matches(text, MOTION_WORDS + ["motion", "pace", "movement"]) > 0
    )
    mentions_graphic = int(
        bool(row.get("likely_graphic_overlay")) and any(term in t for term in ["graphic", "overlay", "scoreboard"])
    )
    return {
        "visual_terms": visual_terms,
        "mentions_wide_field_when_visible": mentions_wide_field,
        "mentions_motion_when_motionful": mentions_motion,
        "mentions_graphic_when_overlay": mentions_graphic,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames:
        fields = fieldnames
    else:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stratified_examples(rows_by_window: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    examples = []
    seen_labels = set()
    for wanted in EVENTS_FOR_EXAMPLES:
        for _, audience_rows in rows_by_window.items():
            if set(audience_rows) >= {"beginner", "expert", "low_vision"}:
                label = audience_rows["beginner"].get("event_label")
                if label == wanted and label not in seen_labels:
                    base = audience_rows["beginner"]
                    examples.append(
                        {
                            "event_label": label,
                            "game_label": base.get("game_label"),
                            "timestamp": base.get("timestamp"),
                            "beginner": audience_rows["beginner"].get("commentary"),
                            "expert": audience_rows["expert"].get("commentary"),
                            "low_vision": audience_rows["low_vision"].get("commentary"),
                        }
                    )
                    seen_labels.add(label)
                    break
    return examples


def write_examples_md(path: Path, examples: list[dict[str, Any]]) -> None:
    lines = ["# Qualitative Examples", ""]
    for i, ex in enumerate(examples, 1):
        lines.extend(
            [
                f"## {i}. {ex['event_label']} ({ex['game_label']}, {ex.get('timestamp')})",
                "",
                f"**Beginner:** {ex['beginner']}",
                "",
                f"**Expert:** {ex['expert']}",
                "",
                f"**Low vision:** {ex['low_vision']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis/full_dataset_20260625/merged_commentary.jsonl.gz"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/full_dataset_20260625"),
    )
    parser.add_argument("--max-pairwise-windows", type=int, default=50000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = repo_root()
    input_path = args.input if args.input.is_absolute() else base / args.input
    out_dir = ensure_dir(args.out_dir if args.out_dir.is_absolute() else base / args.out_dir)

    totals = Counter()
    by_audience: dict[str, Counter] = defaultdict(Counter)
    by_event: dict[str, Counter] = defaultdict(Counter)
    event_scores: dict[str, list[float]] = defaultdict(list)
    audience_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    visual_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    start_phrases = Counter()
    bigrams = Counter()
    trigrams = Counter()
    rows_by_window: dict[str, dict[str, dict[str, Any]]] = {}
    all_windows = set()
    all_games = set()
    all_window_events: dict[str, str] = {}
    heuristic_hits = 0

    rng = random.Random(7)
    pairwise_windows: dict[str, dict[str, dict[str, Any]]] = {}

    for row in iter_jsonl_gz(input_path):
        text = row.get("commentary") or ""
        audience = row.get("audience") or "unknown"
        label = row.get("event_label") or "unknown"
        window_id = row.get("window_id") or ""
        game = row.get("game_label") or ""

        totals["rows"] += 1
        by_audience[audience]["rows"] += 1
        by_event[label]["rows"] += 1
        all_windows.add(window_id)
        all_games.add(game)
        all_window_events[window_id] = label

        wc = word_count(text)
        ev = event_consistency(label, text)
        style = audience_style_score(audience, text)
        slots = slot_coverage(audience, text)
        heuristic_hits += int(heuristic_audience(text) == audience)
        event_scores[label].append(ev)
        audience_metrics[audience]["word_count"].append(float(wc))
        audience_metrics[audience]["event_consistency"].append(ev)
        audience_metrics[audience]["style_score"].append(style)
        audience_metrics[audience]["slot_coverage"].append(slots)
        audience_metrics[audience]["explanatory_terms"].append(float(count_matches(text, EXPLANATORY_WORDS)))
        audience_metrics[audience]["tactical_terms"].append(float(count_matches(text, TACTICAL_WORDS)))
        audience_metrics[audience]["visual_terms"].append(float(count_matches(text, VISUAL_WORDS + SPATIAL_WORDS + MOTION_WORDS)))
        audience_metrics[audience]["vague_visual_count"].append(float(len(vague_visual_flags(text))))
        if audience == "low_vision":
            audience_metrics[audience]["low_vision_spatial_score"].append(low_vision_spatial_score(text))

        visual = visual_grounding_row(row)
        for key, value in visual.items():
            visual_metrics[audience][key].append(float(value))

        start_phrases[start_phrase(text)] += 1
        bigrams.update(ngrams(text, 2))
        trigrams.update(ngrams(text, 3))

        if len(rows_by_window) < 200000:
            rows_by_window.setdefault(window_id, {})[audience] = row
        if window_id in pairwise_windows or len(pairwise_windows) < args.max_pairwise_windows:
            pairwise_windows.setdefault(window_id, {})[audience] = row
        elif rng.random() < args.max_pairwise_windows / max(1, len(all_windows)):
            pairwise_windows.setdefault(window_id, {})[audience] = row

    dataset_stats = {
        "input": str(input_path),
        "rows": totals["rows"],
        "unique_windows": len(all_windows),
        "unique_games": len(all_games),
        "unique_event_labels": len(by_event),
        "audiences": dict(sorted((k, v["rows"]) for k, v in by_audience.items())),
        "commentary_rows_per_window": totals["rows"] / max(1, len(all_windows)),
    }
    write_json(out_dir / "dataset_statistics.json", dataset_stats)

    event_rows = []
    for label, counter in sorted(by_event.items(), key=lambda item: (-item[1]["rows"], item[0])):
        scores = event_scores[label]
        event_rows.append(
            {
                "event_label": label,
                "commentary_rows": counter["rows"],
                "windows": counter["rows"] // 3,
                "event_consistency_rate": safe_mean(scores),
            }
        )
    write_csv(out_dir / "event_faithfulness_by_label.csv", event_rows)

    audience_rows = []
    for audience in sorted(audience_metrics):
        metrics = audience_metrics[audience]
        row = {"audience": audience, "rows": by_audience[audience]["rows"]}
        for key, vals in sorted(metrics.items()):
            row[f"{key}_mean"] = safe_mean(vals)
        audience_rows.append(row)
    write_csv(out_dir / "audience_adaptation_metrics.csv", audience_rows)

    pair_scores = defaultdict(list)
    complete_triplets = 0
    for aud_rows in pairwise_windows.values():
        if set(aud_rows) >= {"beginner", "expert", "low_vision"}:
            complete_triplets += 1
            pair_scores["beginner_vs_expert"].append(
                jaccard(aud_rows["beginner"]["commentary"], aud_rows["expert"]["commentary"])
            )
            pair_scores["beginner_vs_low_vision"].append(
                jaccard(aud_rows["beginner"]["commentary"], aud_rows["low_vision"]["commentary"])
            )
            pair_scores["expert_vs_low_vision"].append(
                jaccard(aud_rows["expert"]["commentary"], aud_rows["low_vision"]["commentary"])
            )
    audience_summary = {
        "heuristic_audience_accuracy": heuristic_hits / max(1, totals["rows"]),
        "pairwise_windows_evaluated": complete_triplets,
        "pairwise_jaccard": {key: safe_mean(vals) for key, vals in pair_scores.items()},
        "metrics_by_audience": audience_rows,
    }
    write_json(out_dir / "audience_adaptation_summary.json", audience_summary)

    visual_rows = []
    for audience in sorted(visual_metrics):
        row = {"audience": audience, "rows": by_audience[audience]["rows"]}
        for key, vals in sorted(visual_metrics[audience].items()):
            row[f"{key}_mean"] = safe_mean(vals)
        visual_rows.append(row)
    write_csv(out_dir / "visual_grounding_metrics.csv", visual_rows)
    write_json(out_dir / "visual_grounding_summary.json", {"metrics_by_audience": visual_rows})

    total_bigrams = sum(bigrams.values())
    total_trigrams = sum(trigrams.values())
    repetition_summary = {
        "top_start_phrases": start_phrases.most_common(25),
        "top_bigrams": [(" ".join(k), v) for k, v in bigrams.most_common(25)],
        "top_trigrams": [(" ".join(k), v) for k, v in trigrams.most_common(25)],
        "distinct_2": len(bigrams) / max(1, total_bigrams),
        "distinct_3": len(trigrams) / max(1, total_trigrams),
    }
    write_json(out_dir / "repetition_summary.json", repetition_summary)

    examples = stratified_examples(rows_by_window)
    write_csv(out_dir / "qualitative_examples.csv", examples)
    write_examples_md(out_dir / "qualitative_examples.md", examples)

    final_summary = {
        "dataset": dataset_stats,
        "event_faithfulness_mean": safe_mean([score for scores in event_scores.values() for score in scores]),
        "audience_adaptation": audience_summary,
        "visual_grounding": {"metrics_by_audience": visual_rows},
        "repetition": repetition_summary,
        "qualitative_examples": len(examples),
    }
    write_json(out_dir / "full_evaluation_summary.json", final_summary)

    md_lines = [
        "# Full Dataset Evaluation Summary",
        "",
        f"- Rows: {dataset_stats['rows']:,}",
        f"- Unique windows: {dataset_stats['unique_windows']:,}",
        f"- Unique games: {dataset_stats['unique_games']:,}",
        f"- Event faithfulness mean: {pct(final_summary['event_faithfulness_mean'])}%",
        f"- Heuristic audience-identification accuracy: {pct(audience_summary['heuristic_audience_accuracy'])}%",
        f"- Distinct-2: {repetition_summary['distinct_2']:.4f}",
        f"- Distinct-3: {repetition_summary['distinct_3']:.4f}",
        "",
        "## Audience Metrics",
        "",
    ]
    for row in audience_rows:
        md_lines.append(
            f"- {row['audience']}: words={row.get('word_count_mean', 0):.2f}, "
            f"event={pct(row.get('event_consistency_mean'))}%, "
            f"style={pct(row.get('style_score_mean'))}%, "
            f"slot={pct(row.get('slot_coverage_mean'))}%"
        )
    md_lines.extend(["", "## Pairwise Audience Similarity", ""])
    for key, value in audience_summary["pairwise_jaccard"].items():
        md_lines.append(f"- {key}: {value:.3f}")
    md_lines.extend(["", "## Outputs", ""])
    for filename in [
        "dataset_statistics.json",
        "event_faithfulness_by_label.csv",
        "audience_adaptation_metrics.csv",
        "audience_adaptation_summary.json",
        "visual_grounding_metrics.csv",
        "repetition_summary.json",
        "qualitative_examples.md",
    ]:
        md_lines.append(f"- {filename}")
    (out_dir / "full_evaluation_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
