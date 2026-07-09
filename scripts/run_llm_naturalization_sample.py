import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from pipeline_utils import ensure_dir, read_jsonl, repo_root, write_jsonl


SYSTEM_PROMPT = """You rewrite soccer commentary to sound natural and broadcast-like.

Rules:
- Preserve the event meaning and audience target.
- Use only facts present in the input.
- Do not invent player names, teams, scores, exact positions, or details.
- Avoid repetitive report-like openings such as "The key moment is" and "The visible cue is".
- Keep beginner outputs clear and explanatory.
- Keep expert outputs tactical but concise.
- Keep low-vision outputs visually descriptive and concrete.
- Return only the rewritten commentary, no JSON and no bullet points.
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


def build_prompt(row: dict[str, Any]) -> str:
    payload = {
        "event_label": row.get("event_label"),
        "audience": row.get("audience"),
        "current_commentary": row.get("commentary"),
        "constraints": {
            "preserve_event": True,
            "preserve_audience": True,
            "do_not_add_unprovided_specifics": True,
        },
    }
    return "Rewrite this commentary more naturally while preserving the grounded content:\n\n" + json.dumps(payload, indent=2)


def call_openai(client: OpenAI, model: str, row: dict[str, Any], temperature: float) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(row)},
        ],
        temperature=temperature,
        max_output_tokens=120,
    )
    return response.output_text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis/full_dataset_20260625/ablation_generations.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/full_dataset_20260625/llm_naturalization_sample.jsonl"),
    )
    parser.add_argument("--source-generator", default="full_template_existing")
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--dry-run-prompts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = repo_root()
    load_env_file(base / ".env")
    load_env_file(base / ".env")
    input_path = args.input if args.input.is_absolute() else base / args.input
    out = args.out if args.out.is_absolute() else base / args.out
    ensure_dir(out.parent)

    rows = [r for r in read_jsonl(input_path) if r.get("generator") == args.source_generator]
    rows = rows[: args.limit] if args.limit else rows
    outputs = []
    if args.dry_run_prompts:
        for row in rows:
            outputs.append({**row, "model": args.model, "system_prompt": SYSTEM_PROMPT, "user_prompt": build_prompt(row)})
        write_jsonl(out, outputs)
        print(json.dumps({"output": str(out), "rows": len(outputs), "dry_run": True}, indent=2))
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Use --dry-run-prompts or add it to .env.")
    client = OpenAI()
    for i, row in enumerate(rows, 1):
        try:
            rewritten = call_openai(client, args.model, row, args.temperature)
        except OpenAIError as exc:
            write_jsonl(out, outputs)
            raise SystemExit(f"OpenAI stopped after {i-1}/{len(rows)} rows: {exc}") from exc
        outputs.append({**row, "generator": "llm_naturalized", "source_generator": args.source_generator, "model": args.model, "commentary": rewritten})
        if i % 10 == 0:
            print(f"naturalized {i}/{len(rows)}")
    write_jsonl(out, outputs)
    print(json.dumps({"output": str(out), "rows": len(outputs), "model": args.model}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
