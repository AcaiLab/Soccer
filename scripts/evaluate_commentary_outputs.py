import argparse
import json
from pathlib import Path

import pandas as pd

from commentary_eval_utils import score_commentary_row, summarize_scores
from pipeline_utils import ensure_dir, read_jsonl, repo_root, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/llm_commentary_samples_grounded.jsonl"))
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = repo_root()
    reports = ensure_dir(base / "reports")
    out_csv = args.out_csv or reports / f"{args.input.stem}_evaluation.csv"
    out_summary = args.out_summary or reports / f"{args.input.stem}_evaluation_summary.json"
    rows = read_jsonl(args.input)
    scored = [score_commentary_row(row) for row in rows if row.get("commentary")]
    pd.DataFrame(scored).to_csv(out_csv, index=False)
    summary = summarize_scores(scored)
    summary["input"] = str(args.input)
    summary["out_csv"] = str(out_csv)
    write_json(out_summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
