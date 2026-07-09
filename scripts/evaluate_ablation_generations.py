import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from commentary_eval_utils import (
    MOTION_WORDS,
    SPATIAL_WORDS,
    audience_style_score,
    event_consistency,
    low_vision_spatial_score,
    slot_coverage,
    vague_visual_flags,
    word_count,
)
from evaluate_full_dataset_analysis import VISUAL_WORDS, count_matches, ngrams, start_phrase
from pipeline_utils import ensure_dir, read_jsonl, repo_root, write_json


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def distinct_n(texts: list[str], n: int) -> float:
    grams = Counter()
    total = 0
    for text in texts:
        ns = ngrams(text, n)
        grams.update(ns)
        total += len(ns)
    return len(grams) / max(1, total)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis/full_dataset_20260625/ablation_generations.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/full_dataset_20260625"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = repo_root()
    input_path = args.input if args.input.is_absolute() else base / args.input
    out_dir = ensure_dir(args.out_dir if args.out_dir.is_absolute() else base / args.out_dir)
    rows = read_jsonl(input_path)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_audience: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["generator"]].append(row)
        grouped_audience[(row["generator"], row["audience"])].append(row)

    table = []
    for gen, gen_rows in sorted(grouped.items()):
        texts = [r["commentary"] for r in gen_rows]
        event_scores = [event_consistency(r["event_label"], r["commentary"]) for r in gen_rows]
        style_scores = [audience_style_score(r["audience"], r["commentary"]) for r in gen_rows]
        slot_scores = [slot_coverage(r["audience"], r["commentary"]) for r in gen_rows]
        visual_terms = [float(count_matches(r["commentary"], VISUAL_WORDS + SPATIAL_WORDS + MOTION_WORDS)) for r in gen_rows]
        vague_counts = [float(len(vague_visual_flags(r["commentary"]))) for r in gen_rows]
        starts = Counter(start_phrase(t) for t in texts)
        table.append(
            {
                "generator": gen,
                "rows": len(gen_rows),
                "event_faithfulness": safe_mean(event_scores),
                "audience_style": safe_mean(style_scores),
                "slot_coverage": safe_mean(slot_scores),
                "visual_terms_mean": safe_mean(visual_terms),
                "vague_visual_count_mean": safe_mean(vague_counts),
                "word_count_mean": safe_mean([float(word_count(t)) for t in texts]),
                "distinct_2": distinct_n(texts, 2),
                "distinct_3": distinct_n(texts, 3),
                "top_start_phrase": starts.most_common(1)[0][0] if starts else "",
                "top_start_phrase_rate": starts.most_common(1)[0][1] / len(texts) if texts else 0,
            }
        )

    audience_table = []
    for (gen, audience), subset in sorted(grouped_audience.items()):
        audience_table.append(
            {
                "generator": gen,
                "audience": audience,
                "rows": len(subset),
                "event_faithfulness": safe_mean([event_consistency(r["event_label"], r["commentary"]) for r in subset]),
                "audience_style": safe_mean([audience_style_score(audience, r["commentary"]) for r in subset]),
                "slot_coverage": safe_mean([slot_coverage(audience, r["commentary"]) for r in subset]),
                "visual_terms_mean": safe_mean(
                    [float(count_matches(r["commentary"], VISUAL_WORDS + SPATIAL_WORDS + MOTION_WORDS)) for r in subset]
                ),
                "low_vision_spatial_score": safe_mean([low_vision_spatial_score(r["commentary"]) for r in subset])
                if audience == "low_vision"
                else None,
                "word_count_mean": safe_mean([float(word_count(r["commentary"])) for r in subset]),
            }
        )

    write_csv(out_dir / "ablation_comparison_by_generator.csv", table)
    write_csv(out_dir / "ablation_comparison_by_generator_audience.csv", audience_table)
    summary = {
        "input": str(input_path),
        "rows": len(rows),
        "generators": [row["generator"] for row in table],
        "by_generator": table,
    }
    write_json(out_dir / "ablation_comparison_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
