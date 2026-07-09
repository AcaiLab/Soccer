import argparse
import json
from pathlib import Path

from pipeline_utils import ensure_dir, read_jsonl, write_jsonl


def load_by_window(path: Path, key: str) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    return {row["window_id"]: row.get(key, {}) for row in read_jsonl(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=Path("outputs/content_plans.jsonl"))
    parser.add_argument("--basic-cues", type=Path, default=Path("features/basic_visual_cues.jsonl"))
    parser.add_argument("--zeroshot-cues", type=Path, default=Path("features/zeroshot_visual_cues.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/content_plans_with_visual_facts.jsonl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plans = read_jsonl(args.plans)
    basic = load_by_window(args.basic_cues, "visual_facts")
    zeroshot = load_by_window(args.zeroshot_cues, "zeroshot_visual_cues")
    merged = []
    for plan in plans:
        visual_facts = basic.get(plan["window_id"], {})
        if zeroshot.get(plan["window_id"]):
            visual_facts = {**visual_facts, "zeroshot": zeroshot[plan["window_id"]]}
        plan = dict(plan)
        plan["visual_facts"] = visual_facts
        plan["must_include"] = list(plan.get("must_include", []))
        if plan.get("audience") == "low_vision" and "grounded visual facts when available" not in plan["must_include"]:
            plan["must_include"].append("grounded visual facts when available")
        merged.append(plan)
    ensure_dir(args.out.parent)
    write_jsonl(args.out, merged)
    print(json.dumps({"output": str(args.out), "plans": len(merged), "basic_windows": len(basic), "zeroshot_windows": len(zeroshot)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
