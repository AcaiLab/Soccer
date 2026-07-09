import argparse
import json
from pathlib import Path

from generate_llm_commentary import SYSTEM_PROMPT, build_user_prompt
from pipeline_utils import ensure_dir, read_jsonl, write_jsonl


VARIANT_SYSTEMS = {
    "minimal": "Generate one short soccer commentary sentence for the requested audience using only the provided facts.",
    "strict_grounded": SYSTEM_PROMPT,
    "accessibility_strict": SYSTEM_PROMPT + "\nFor low-vision outputs, prioritize concrete visible state over tactics.",
    "broadcast_style": SYSTEM_PROMPT + "\nMake the language sound like natural live broadcast commentary, not a report.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=Path("outputs/content_plans.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/prompt_variants.jsonl"))
    parser.add_argument("--limit", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plans = read_jsonl(args.plans)
    if args.limit:
        plans = plans[: args.limit]
    rows = []
    for plan in plans:
        for name, system in VARIANT_SYSTEMS.items():
            rows.append(
                {
                    "window_id": plan["window_id"],
                    "event_label": plan["event_label"],
                    "audience": plan["audience"],
                    "variant": name,
                    "system_prompt": system,
                    "user_prompt": build_user_prompt(plan),
                }
            )
    ensure_dir(args.out.parent)
    write_jsonl(args.out, rows)
    print(json.dumps({"output": str(args.out), "rows": len(rows), "variants": list(VARIANT_SYSTEMS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
