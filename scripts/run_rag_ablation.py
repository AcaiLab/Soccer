import argparse
import copy
import json
from pathlib import Path

from generate_natural_commentary import generate_commentary
from pipeline_utils import ensure_dir, read_jsonl, write_jsonl


def ablate_plan(plan: dict, mode: str, top_k: int) -> dict:
    plan = copy.deepcopy(plan)
    retrieval = plan["retrieval"]
    neighbors = retrieval.get("neighbors", [])
    if mode == "no_retrieval":
        neighbors = []
    elif mode == "top1":
        neighbors = neighbors[:1]
    elif mode == "topk":
        neighbors = neighbors[:top_k]
    elif mode == "same_label_only":
        neighbors = [n for n in neighbors if n["event_label"] == plan["event_label"]][:top_k]
    elif mode == "cross_label_only":
        neighbors = [n for n in neighbors if n["event_label"] != plan["event_label"]][:top_k]
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")
    retrieval["neighbors"] = neighbors
    retrieval["neighbor_count"] = len(neighbors)
    retrieval["same_label_neighbors"] = sum(n["event_label"] == plan["event_label"] for n in neighbors)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=Path("outputs/content_plans.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/rag_ablation_natural.jsonl"))
    parser.add_argument("--modes", nargs="+", default=["no_retrieval", "top1", "topk", "same_label_only", "cross_label_only"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plans = read_jsonl(args.plans)
    if args.limit:
        plans = plans[: args.limit]
    rows = []
    for plan in plans:
        for mode in args.modes:
            ablated = ablate_plan(plan, mode, args.top_k)
            rows.append(
                {
                    "window_id": plan["window_id"],
                    "game_label": plan["game_label"],
                    "event_label": plan["event_label"],
                    "audience": plan["audience"],
                    "generator": f"rag_{mode}",
                    "ablation_mode": mode,
                    "commentary": generate_commentary(ablated),
                    "retrieval": ablated["retrieval"],
                }
            )
    ensure_dir(args.out.parent)
    write_jsonl(args.out, rows)
    print(json.dumps({"output": str(args.out), "rows": len(rows), "modes": args.modes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
