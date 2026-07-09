import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import ensure_dir, load_config, repo_root, resolve_path, write_json


def window_id_from_clip_id(clip_id: str) -> str:
    return re.sub(r"_frame\d{2}-\d{2}\.\d+$", "", clip_id)


def frame_time_from_clip_id(clip_id: str) -> float:
    match = re.search(r"_frame(\d{2})-(\d{2}\.\d+)$", clip_id)
    if not match:
        return 0.0
    return int(match.group(1)) * 60.0 + float(match.group(2))


def sample_evenly(paths: list[str], n: int) -> list[str]:
    if not paths:
        return []
    if len(paths) == 1:
        return paths * n
    positions = np.linspace(0, len(paths) - 1, n).round().astype(int)
    return [paths[i] for i in positions]


def build_manifest(
    frame_manifest_path: Path,
    out_jsonl: Path,
    sampled_frames_per_window: int,
    max_games: int | None,
    max_windows: int | None,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(frame_manifest_path)
    required = {"image_path", "event_label", "game_label", "clip_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Frame manifest is missing columns: {sorted(missing)}")

    df["window_id"] = df["clip_id"].map(window_id_from_clip_id)
    df["frame_time"] = df["clip_id"].map(frame_time_from_clip_id)

    games = sorted(df["game_label"].unique())
    if max_games:
        games = games[:max_games]
        df = df[df["game_label"].isin(games)].copy()

    rows: list[dict[str, object]] = []
    for window_id, group in df.sort_values("frame_time").groupby("window_id", sort=True):
        labels = group["event_label"].unique()
        games_in_window = group["game_label"].unique()
        if len(labels) != 1 or len(games_in_window) != 1:
            raise ValueError(f"Mixed label/game window: {window_id}")

        paths = group["image_path"].tolist()
        rows.append(
            {
                "window_id": window_id,
                "game_label": games_in_window[0],
                "event_label": labels[0],
                "frame_count": len(paths),
                "sampled_frame_count": sampled_frames_per_window,
                "image_paths": paths,
                "sampled_image_paths": sample_evenly(paths, sampled_frames_per_window),
            }
        )

    if max_windows and len(rows) > max_windows:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(len(rows), size=max_windows, replace=False).tolist())
        rows = [rows[i] for i in keep]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    flat = pd.DataFrame(
        [
            {
                "window_id": row["window_id"],
                "game_label": row["game_label"],
                "event_label": row["event_label"],
                "frame_count": row["frame_count"],
                "sampled_frame_count": row["sampled_frame_count"],
            }
            for row in rows
        ]
    )
    flat.to_csv(out_jsonl.with_suffix(".csv"), index=False)
    return flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--frame-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    base = repo_root()
    output_root = resolve_path(cfg["project"]["output_root"], base) or base
    manifest_dir = ensure_dir(output_root / "manifests")

    frame_manifest = args.frame_manifest or resolve_path(
        cfg["data"].get("frame_manifest", "artifacts/frame_manifest.csv"), base
    )
    out = args.out or manifest_dir / "local_windows.jsonl"
    flat = build_manifest(
        frame_manifest_path=frame_manifest,
        out_jsonl=out,
        sampled_frames_per_window=int(cfg["windows"]["sampled_frames_per_window"]),
        max_games=cfg["data"].get("max_games"),
        max_windows=args.max_windows or cfg["data"].get("max_windows"),
        seed=args.seed,
    )

    summary = {
        "frame_manifest": str(frame_manifest),
        "window_manifest": str(out),
        "windows": int(len(flat)),
        "games": int(flat["game_label"].nunique()),
        "labels": int(flat["event_label"].nunique()),
        "label_counts": dict(Counter(flat["event_label"])),
        "frame_count_summary": flat["frame_count"].describe().to_dict(),
    }
    write_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
