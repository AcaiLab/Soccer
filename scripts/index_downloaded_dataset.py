import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_utils import ensure_dir, write_json


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
ANNOTATION_EXTS = {".json", ".jsonl", ".pkl", ".pickle", ".csv"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".xz", ".zst"}


def classify(path: Path) -> str:
    suffixes = path.suffixes
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in ANNOTATION_EXTS:
        return "annotation_or_metadata"
    if suffix in ARCHIVE_EXTS or any(s.lower() in ARCHIVE_EXTS for s in suffixes):
        return "archive"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/data/SoccerReplay-1988"))
    parser.add_argument("--out-csv", type=Path, default=Path("manifests/downloaded_dataset_inventory.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("manifests/downloaded_dataset_inventory_summary.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in args.root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(args.root)),
                "kind": classify(path),
                "suffix": path.suffix.lower(),
                "size_bytes": stat.st_size,
            }
        )
    df = pd.DataFrame(rows)
    ensure_dir(args.out_csv.parent)
    df.to_csv(args.out_csv, index=False)
    summary = {
        "root": str(args.root),
        "files": int(len(df)),
        "size_bytes": int(df["size_bytes"].sum()) if len(df) else 0,
        "by_kind": df.groupby("kind")["size_bytes"].agg(["count", "sum"]).to_dict(orient="index") if len(df) else {},
        "out_csv": str(args.out_csv),
    }
    write_json(args.out_summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
