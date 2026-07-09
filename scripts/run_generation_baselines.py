import argparse
import json
from pathlib import Path

from build_content_plans import generic_event_facts
from generate_natural_commentary import generate_commentary
from pipeline_utils import ensure_dir, read_jsonl, repo_root, write_jsonl


def template_baseline(plan: dict) -> str:
    facts = plan["facts"]
    return f"{facts['plain_event'].capitalize()}. {facts['next_state'].capitalize()}."


def generic_no_audience_baseline(plan: dict) -> str:
    facts = plan["facts"]
    return " ".join(
        [
            f"{facts['plain_event'].capitalize()}.",
            f"{facts['rule_context'].capitalize()}.",
            f"{facts['next_state'].capitalize()}.",
        ]
    )


def label_only_baseline(plan: dict) -> str:
    label = plan["event_label"].replace("_", " ").replace("-", " ")
    facts = generic_event_facts(plan["event_label"])
    return f"A {label} moment occurs. {facts['next_state'].capitalize()}."


GENERATORS = {
    "label_only": label_only_baseline,
    "template": template_baseline,
    "generic_no_audience": generic_no_audience_baseline,
    "content_plan_natural": generate_commentary,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=Path("outputs/content_plans.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/generation_baselines.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plans = read_jsonl(args.plans)
    if args.limit:
        plans = plans[: args.limit]
    rows = []
    for plan in plans:
        for name, fn in GENERATORS.items():
            rows.append(
                {
                    "window_id": plan["window_id"],
                    "game_label": plan["game_label"],
                    "event_label": plan["event_label"],
                    "audience": plan["audience"],
                    "generator": name,
                    "commentary": fn(plan),
                }
            )
    ensure_dir(args.out.parent)
    write_jsonl(args.out, rows)
    print(json.dumps({"output": str(args.out), "rows": len(rows), "baselines": list(GENERATORS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
