import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_utils import ensure_dir, read_jsonl, repo_root


RUBRIC_COLUMNS = [
    "factual_correctness_1_5",
    "audience_fit_1_5",
    "usefulness_1_5",
    "too_verbose",
    "unsupported_claims",
    "annotator_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/llm_commentary_samples_grounded.jsonl"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [r for r in read_jsonl(args.input) if r.get("commentary")]
    if args.limit:
        rows = rows[: args.limit]
    export_rows = []
    for i, row in enumerate(rows, start=1):
        item = {
            "annotation_id": f"ann_{i:05d}",
            "window_id": row["window_id"],
            "game_label": row.get("game_label"),
            "event_label": row["event_label"],
            "audience": row["audience"],
            "commentary": row["commentary"],
        }
        item.update({col: "" for col in RUBRIC_COLUMNS})
        export_rows.append(item)
    out = args.out or repo_root() / "reports" / f"{args.input.stem}_annotation_batch.csv"
    ensure_dir(out.parent)
    pd.DataFrame(export_rows).to_csv(out, index=False)
    print(json.dumps({"input": str(args.input), "output": str(out), "rows": len(export_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
