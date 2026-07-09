import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from pipeline_utils import ensure_dir, sanitize_name, write_json


def annotation_split_parts(path: Path, root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) < 5:
        raise ValueError(f"Unexpected annotation path depth: {path}")
    split = parts[0]
    season_folder = parts[-3]
    game_folder = parts[-2]
    return split, season_folder, game_folder


def parse_event_counts(annotation_path: Path) -> tuple[int, Counter[str]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for comment in data.get("comments", []):
        if not (comment.get("time_stamp") or "").strip():
            continue
        try:
            half = int(comment.get("half") or 0)
        except (TypeError, ValueError):
            continue
        if half not in {1, 2}:
            continue
        label = sanitize_name(comment.get("comments_type", "unknown"))
        counts[label] += 1
    return sum(counts.values()), counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=Path("/home/ubuntu/Soccer_Data/annotations"))
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("manifests/dropbox_full_dataset_manifest.csv"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("manifests/dropbox_full_dataset_manifest_summary.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.annotation_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Annotation root not found: {root}")

    rows = []
    skipped = []
    label_counts: Counter[str] = Counter()
    for path in sorted(root.rglob("*.json")):
        try:
            split, season_folder, game_folder = annotation_split_parts(path, root)
            event_count, counts = parse_event_counts(path)
        except Exception as exc:
            skipped.append({"path": str(path), "error": type(exc).__name__, "message": str(exc)})
            continue
        label_counts.update(counts)
        rows.append(
            {
                "split": split,
                "season_folder": season_folder,
                "game_folder": game_folder,
                "game_label": game_folder,
                "annotation_path": str(path),
                "dropbox_game_path": f"/{season_folder}/{game_folder}",
                "event_count": event_count,
                "label_counts_json": json.dumps(dict(counts), sort_keys=True),
            }
        )

    df = pd.DataFrame(rows)
    ensure_dir(args.out_csv.parent)
    df.to_csv(args.out_csv, index=False)
    summary = {
        "annotation_root": str(root),
        "out_csv": str(args.out_csv),
        "games": int(len(df)),
        "splits": df["split"].value_counts().to_dict() if len(df) else {},
        "season_folders": int(df["season_folder"].nunique()) if len(df) else 0,
        "events": int(df["event_count"].sum()) if len(df) else 0,
        "labels": dict(label_counts.most_common()),
        "skipped_count": len(skipped),
        "skipped": skipped[:20],
    }
    write_json(args.out_summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
