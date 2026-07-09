import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI, OpenAIError

from pipeline_utils import ensure_dir, read_jsonl, repo_root, write_json, write_jsonl


SYSTEM_PROMPT = """You are evaluating soccer commentary for a research paper.

Return strict JSON with integer scores from 1 to 5:
{
  "event_faithfulness": 1-5,
  "audience_fit": 1-5,
  "visual_grounding": 1-5,
  "naturalness": 1-5,
  "notes": "short reason"
}

Scoring:
- event_faithfulness: preserves the provided event label and does not contradict it.
- audience_fit: suits beginner, expert, or low-vision users as requested.
- visual_grounding: uses visual information appropriately; do not reward invented details.
- naturalness: sounds like concise spoken commentary, not a rigid template.
"""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:]
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def prompt(row: dict[str, Any]) -> str:
    payload = {
        "event_label": row.get("event_label"),
        "audience": row.get("audience"),
        "generator": row.get("generator"),
        "commentary": row.get("commentary"),
    }
    return "Score this generated commentary:\n\n" + json.dumps(payload, indent=2)


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def call_openai(client: OpenAI, model: str, row: dict[str, Any]) -> dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt(row)},
        ],
        temperature=0,
        max_output_tokens=160,
    )
    return parse_json(response.output_text)


def balanced_sample(rows: list[dict[str, Any]], per_generator: int) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("generator")].append(row)
    sample = []
    for _, items in sorted(grouped.items()):
        sample.extend(items[:per_generator])
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis/full_dataset_20260625/ablation_generations.jsonl"),
    )
    parser.add_argument(
        "--extra-input",
        type=Path,
        default=Path("analysis/full_dataset_20260625/llm_naturalization_sample.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/full_dataset_20260625/llm_judge_scores.jsonl"),
    )
    parser.add_argument("--per-generator", type=int, default=24)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--dry-run-prompts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = repo_root()
    load_env_file(base / ".env")
    load_env_file(base / ".env")
    input_path = args.input if args.input.is_absolute() else base / args.input
    extra_path = args.extra_input if args.extra_input.is_absolute() else base / args.extra_input
    out = args.out if args.out.is_absolute() else base / args.out
    ensure_dir(out.parent)

    rows = read_jsonl(input_path)
    if extra_path.exists():
        rows += read_jsonl(extra_path)
    rows = balanced_sample(rows, args.per_generator)
    if args.dry_run_prompts:
        write_jsonl(out, [{**row, "model": args.model, "system_prompt": SYSTEM_PROMPT, "user_prompt": prompt(row)} for row in rows])
        print(json.dumps({"output": str(out), "rows": len(rows), "dry_run": True}, indent=2))
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Use --dry-run-prompts or add it to .env.")
    client = OpenAI()
    scored = []
    for i, row in enumerate(rows, 1):
        try:
            score = call_openai(client, args.model, row)
        except (OpenAIError, json.JSONDecodeError) as exc:
            write_jsonl(out, scored)
            raise SystemExit(f"Judge stopped after {i-1}/{len(rows)} rows: {exc}") from exc
        scored.append({**row, "judge_model": args.model, **score})
        if i % 20 == 0:
            print(f"judged {i}/{len(rows)}")
    write_jsonl(out, scored)
    df = pd.DataFrame(scored)
    summary = {
        "output": str(out),
        "rows": len(scored),
        "model": args.model,
        "by_generator": df.groupby("generator")[["event_faithfulness", "audience_fit", "visual_grounding", "naturalness"]].mean().round(3).to_dict("index"),
    }
    write_json(out.with_suffix(".summary.json"), summary)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
