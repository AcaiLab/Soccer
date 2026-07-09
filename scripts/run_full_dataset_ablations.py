import argparse
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pipeline_utils import ensure_dir, repo_root, write_json, write_jsonl


AUDIENCES = ["beginner", "expert", "low_vision"]


def iter_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def cap(text: str | None) -> str:
    text = text or ""
    return text[:1].upper() + text[1:] if text else text


def event_phrase(row: dict[str, Any]) -> str:
    return row.get("plain_event") or f"a {str(row.get('event_label', 'event')).replace('_', ' ').replace('-', ' ')} occurs"


def next_state(row: dict[str, Any]) -> str:
    return row.get("next_state") or "players reset for the next action"


def retrieval_sentence(row: dict[str, Any], audience: str) -> str:
    same = int(row.get("retrieval_same_label_neighbors") or 0)
    if same <= 0:
        return ""
    noun = "example" if same == 1 else "examples"
    if audience == "expert":
        return f"Similar event patterns appear in {same} retrieved {noun}, which helps compare this phase across matches."
    return f"The system found {same} similar {noun} of this kind of moment."


def visual_sentence(row: dict[str, Any], audience: str) -> str:
    shot = str(row.get("camera_shot_type") or "broadcast view").replace("_", " ")
    motion = row.get("motion_intensity")
    field_visible = row.get("field_visible")
    if audience == "low_vision":
        base = f"The broadcast view is {shot}"
        if motion:
            base += f" with {motion} motion"
        if field_visible is True:
            base += ", and the field is visible"
        return base + "."
    if audience == "expert":
        if motion:
            return f"The visual tempo is {motion}, which frames the speed of the phase."
        return "The visual evidence provides broadcast context for the phase."
    return "The video gives context for what is happening on the field."


def event_only(row: dict[str, Any]) -> str:
    return f"{cap(event_phrase(row))}. {cap(next_state(row))}."


def audience_only(row: dict[str, Any]) -> str:
    ev = event_phrase(row)
    nxt = next_state(row)
    audience = row["audience"]
    if audience == "beginner":
        return f"{cap(ev)}. In simple terms, {row.get('rule_context') or 'this changes the current phase of play'}. Watch for this next: {nxt}."
    if audience == "expert":
        return f"{cap(ev)}. The tactical implication is that {row.get('tactical_context') or 'the phase and spacing may change'}. Immediate next state: {nxt}."
    return f"{cap(ev)}. For a low-vision viewer, the key state change is that {nxt}."


def retrieval_only(row: dict[str, Any]) -> str:
    base = audience_only(row)
    retrieval = retrieval_sentence(row, row["audience"])
    return f"{base} {retrieval}".strip()


def visual_only(row: dict[str, Any]) -> str:
    ev = event_phrase(row)
    audience = row["audience"]
    if audience == "beginner":
        return f"{cap(ev)}. In simple terms, {row.get('rule_context') or 'this changes the current phase of play'}. {visual_sentence(row, audience)}"
    if audience == "expert":
        return f"{cap(ev)}. {visual_sentence(row, audience)} The phase implication is that {next_state(row)}."
    return f"{cap(ev)}. {visual_sentence(row, audience)} The important state change is that {next_state(row)}."


NATURAL_OPENERS = {
    "beginner": [
        "{event}.",
        "This is {event_lower}.",
        "The match has reached {event_lower}.",
        "What just happened is {event_lower}.",
    ],
    "expert": [
        "{event}.",
        "This phase turns on {event_lower}.",
        "The decisive signal is {event_lower}.",
        "From a tactical view, this is {event_lower}.",
    ],
    "low_vision": [
        "{event}.",
        "For orientation, {event_lower}.",
        "The broadcast moment is {event_lower}.",
        "The play state changes because {event_lower}.",
    ],
}


def stable_choice(options: list[str], key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def grounded_naturalized(row: dict[str, Any]) -> str:
    audience = row["audience"]
    ev = event_phrase(row)
    opener = stable_choice(NATURAL_OPENERS[audience], row["window_id"] + audience).format(
        event=cap(ev), event_lower=ev
    )
    if audience == "beginner":
        return " ".join(
            [
                opener,
                f"That means {row.get('rule_context') or 'the current phase of play changes'}, so the useful thing to watch next is whether {next_state(row)}.",
            ]
        )
    if audience == "expert":
        retrieval = retrieval_sentence(row, audience)
        parts = [
            opener,
            f"The phase context is {row.get('tactical_context') or 'the spacing and rhythm can shift'}, with {next_state(row)} as the immediate consequence.",
        ]
        if retrieval:
            parts.append(retrieval)
        return " ".join(parts)
    return " ".join(
        [
            opener,
            visual_sentence(row, audience),
            f"The useful takeaway is that {next_state(row)}.",
        ]
    )


GENERATORS = {
    "event_only": event_only,
    "audience_only": audience_only,
    "retrieval_only": retrieval_only,
    "visual_only": visual_only,
    "full_template_existing": lambda row: row.get("commentary") or "",
    "grounded_naturalized": grounded_naturalized,
}


def select_sample(input_path: Path, max_windows_per_event: int, seed: int) -> dict[str, dict[str, dict[str, Any]]]:
    rng = random.Random(seed)
    by_event: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    seen_counts = Counter()
    for row in iter_jsonl_gz(input_path):
        window_id = row.get("window_id")
        label = row.get("event_label", "unknown")
        if not window_id:
            continue
        if window_id not in by_event[label] and seen_counts[label] >= max_windows_per_event:
            continue
        by_event[label].setdefault(window_id, {})[row.get("audience")] = row
        if set(by_event[label][window_id]) >= set(AUDIENCES):
            seen_counts[label] += 1
    sampled: dict[str, dict[str, dict[str, Any]]] = {}
    for label, windows in by_event.items():
        complete = [(wid, rows) for wid, rows in windows.items() if set(rows) >= set(AUDIENCES)]
        rng.shuffle(complete)
        for wid, rows in complete[:max_windows_per_event]:
            sampled[wid] = rows
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis/full_dataset_20260625/merged_commentary.jsonl.gz"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/full_dataset_20260625/ablation_generations.jsonl"),
    )
    parser.add_argument("--max-windows-per-event", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = repo_root()
    input_path = args.input if args.input.is_absolute() else base / args.input
    out = args.out if args.out.is_absolute() else base / args.out
    ensure_dir(out.parent)

    sampled = select_sample(input_path, args.max_windows_per_event, args.seed)
    rows = []
    for window_id, audience_rows in sampled.items():
        for audience in AUDIENCES:
            source = audience_rows[audience]
            for name, fn in GENERATORS.items():
                rows.append(
                    {
                        "window_id": window_id,
                        "game_label": source.get("game_label"),
                        "event_label": source.get("event_label"),
                        "audience": audience,
                        "generator": name,
                        "commentary": fn(source),
                    }
                )
    write_jsonl(out, rows)
    summary = {
        "input": str(input_path),
        "output": str(out),
        "sample_windows": len(sampled),
        "rows": len(rows),
        "generators": list(GENERATORS),
        "max_windows_per_event": args.max_windows_per_event,
    }
    write_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
