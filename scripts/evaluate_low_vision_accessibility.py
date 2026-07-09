import argparse
import json
from pathlib import Path

import pandas as pd

from commentary_eval_utils import low_vision_spatial_score, vague_visual_flags, word_count
from pipeline_utils import ensure_dir, read_jsonl, repo_root, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/llm_commentary_samples_grounded.jsonl"))
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [r for r in read_jsonl(args.input) if r.get("audience") == "low_vision" and r.get("commentary")]
    scored = []
    for row in rows:
        text = row["commentary"]
        scored.append(
            {
                "window_id": row["window_id"],
                "event_label": row["event_label"],
                "word_count": word_count(text),
                "spatial_motion_score": low_vision_spatial_score(text),
                "vague_visual_count": len(vague_visual_flags(text)),
                "vague_visual_flags": ";".join(vague_visual_flags(text)),
                "too_long_for_live_audio": word_count(text) > 55,
                "commentary": text,
            }
        )
    df = pd.DataFrame(scored)
    reports = ensure_dir(repo_root() / "reports")
    out_csv = args.out_csv or reports / f"{args.input.stem}_low_vision_audit.csv"
    out_summary = args.out_summary or reports / f"{args.input.stem}_low_vision_audit_summary.json"
    df.to_csv(out_csv, index=False)
    summary = {
        "input": str(args.input),
        "rows": int(len(df)),
        "spatial_motion_score_mean": float(df["spatial_motion_score"].mean()) if len(df) else None,
        "vague_visual_count_total": int(df["vague_visual_count"].sum()) if len(df) else 0,
        "too_long_rate": float(df["too_long_for_live_audio"].mean()) if len(df) else None,
        "out_csv": str(out_csv),
    }
    write_json(out_summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
