import argparse
import json
import os
from pathlib import Path

from openai import OpenAI, OpenAIError

from pipeline_utils import ensure_dir, load_config, read_jsonl, repo_root, resolve_path, write_jsonl


SYSTEM_PROMPT = """You generate soccer commentary adapted to a specific audience.

Rules:
- Use only the provided facts and retrieval summary.
- Do not invent player names, team names, score, exact field coordinates, or tactical details that are not provided.
- Do not infer exact field zones such as "left side", "outside the box", "near the defensive area", or "near the goal" unless the provided facts explicitly say so.
- Do not expand generic facts into specific actions such as a defensive wall, direct shot, referee location, or player formation unless explicitly provided.
- Do not start with "The event is".
- Keep the output one or two natural spoken commentary sentences.
- For beginner viewers: explain the rule/context and what happens next in simple language.
- For expert viewers: focus on tactical implication and phase-of-play meaning, without basic rule teaching.
- For blind or low-vision users: emphasize concrete visual/spatial state, visible signals, movement, and restart state.
- When visual_facts are provided, prefer them over generic soccer priors.
- If visual_facts report an unknown field zone or unknown object visibility, do not invent those details.
- Avoid "as you can see", "here", or "there" unless the reference is explicit.
"""


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines without printing or overwriting existing env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_user_prompt(plan: dict) -> str:
    retrieval = plan["retrieval"]
    neighbors = retrieval.get("neighbors", [])
    compact_neighbors = [
        {
            "rank": item["rank"],
            "event_label": item["event_label"],
            "similarity": item["similarity"],
        }
        for item in neighbors[:5]
    ]
    payload = {
        "audience": plan["audience"],
        "audience_goal": plan["audience_goal"],
        "style": plan["style"],
        "must_include": plan["must_include"],
        "avoid": plan["avoid"],
        "event_label": plan["event_label"],
        "facts": plan["facts"],
        "visual_facts": plan.get("visual_facts", {}),
        "retrieval_summary": {
            "same_label_neighbors": retrieval.get("same_label_neighbors", 0),
            "neighbor_count": retrieval.get("neighbor_count", len(neighbors)),
            "nearest_examples": compact_neighbors,
        },
        "grounding_limits": [
            "No player names are provided.",
            "No team names are provided.",
            "No score or match clock is provided.",
            "No exact field coordinates are provided.",
            "Only mention a specific field area if it appears in the facts.",
            "If visual_facts.field.zone is unknown, do not name a specific field zone.",
            "If object visibility is null or unknown, do not claim the object is visible.",
            "Do not turn generic defender organization into a wall, line, or direct-shot setup unless stated.",
        ],
    }
    return (
        "Write one audience-adapted soccer commentary output for this plan.\n"
        "Return only the commentary text, no bullet points, no JSON.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def call_openai(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_output_tokens=120,
    )
    return response.output_text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--dry-run-prompts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    base = repo_root()
    output_root = resolve_path(cfg["project"]["output_root"], base) or base
    load_env_file(base / ".env")
    load_env_file(output_root / ".env")
    output_dir = ensure_dir(output_root / "outputs")
    plans_path = args.plans or output_dir / "content_plans.jsonl"
    out = args.out or output_dir / "llm_commentary_samples.jsonl"
    model = args.model or cfg.get("generation", {}).get("llm_model", "gpt-4.1-mini")

    plans = read_jsonl(plans_path)
    if args.limit and args.limit > 0:
        plans = plans[: args.limit]

    rows = []
    if args.dry_run_prompts:
        for plan in plans:
            rows.append(
                {
                    "window_id": plan["window_id"],
                    "game_label": plan["game_label"],
                    "event_label": plan["event_label"],
                    "audience": plan["audience"],
                    "model": model,
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": build_user_prompt(plan),
                    "commentary": None,
                    "dry_run": True,
                }
            )
        write_jsonl(out, rows)
        print(json.dumps({"output": str(out), "rows": len(rows), "dry_run_prompts": True}, indent=2))
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Set it in your shell, or rerun with --dry-run-prompts."
        )

    client = OpenAI()
    for i, plan in enumerate(plans, start=1):
        user_prompt = build_user_prompt(plan)
        try:
            commentary = call_openai(client, model, SYSTEM_PROMPT, user_prompt, args.temperature)
        except OpenAIError as exc:
            rows.append(
                {
                    "window_id": plan["window_id"],
                    "game_label": plan["game_label"],
                    "event_label": plan["event_label"],
                    "audience": plan["audience"],
                    "model": model,
                    "commentary": None,
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            write_jsonl(out, rows)
            raise SystemExit(
                f"OpenAI generation stopped after {i - 1}/{len(plans)} completed rows: "
                f"{exc.__class__.__name__}. Partial output written to {out}"
            ) from exc
        rows.append(
            {
                "window_id": plan["window_id"],
                "game_label": plan["game_label"],
                "event_label": plan["event_label"],
                "audience": plan["audience"],
                "model": model,
                "commentary": commentary,
                "content_plan": {
                    "must_include": plan["must_include"],
                    "avoid": plan["avoid"],
                    "facts": plan["facts"],
                    "retrieval": plan["retrieval"],
                },
            }
        )
        print(f"generated {i}/{len(plans)}: {plan['event_label']} / {plan['audience']}")

    write_jsonl(out, rows)
    print(json.dumps({"output": str(out), "rows": len(rows), "model": model}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
