import argparse
import json
from pathlib import Path

import numpy as np

from pipeline_utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, default=Path("shards"))
    parser.add_argument("--pattern", default="shard_*/visual_embeddings.npz")
    parser.add_argument("--out", type=Path, default=Path("features/merged_shard_visual_embeddings.npz"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.shard_root.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No shard outputs found under {args.shard_root} with pattern {args.pattern}")
    arrays: dict[str, list[np.ndarray]] = {
        "window_embeddings": [],
        "window_ids": [],
        "event_labels": [],
        "game_labels": [],
    }
    frame_embeddings = []
    sampled_paths = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        for key in arrays:
            arrays[key].append(data[key])
        if "frame_embeddings" in data:
            frame_embeddings.append(data["frame_embeddings"])
        if "sampled_image_paths" in data:
            sampled_paths.append(data["sampled_image_paths"])
    ensure_dir(args.out.parent)
    payload = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    if frame_embeddings:
        payload["frame_embeddings"] = np.concatenate(frame_embeddings, axis=0)
    if sampled_paths:
        payload["sampled_image_paths"] = np.concatenate(sampled_paths, axis=0)
    np.savez_compressed(args.out, **payload)
    summary = {
        "out": str(args.out),
        "shard_files": [str(path) for path in paths],
        "windows": int(len(payload["window_ids"])),
        "window_embedding_shape": list(payload["window_embeddings"].shape),
    }
    write_json(args.out.with_suffix(".metadata.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
