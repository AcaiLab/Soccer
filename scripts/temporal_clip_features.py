"""
Temporal Feature Extraction for Soccer Events
==============================================
Groups extracted frames by event (game x half x label x event_tag), encodes
each frame with a frozen vision model (DINOv2 or CLIP, see soccer/encoders.py)
and saves per-event feature tensors of shape (n_frames, embed_dim).

With a +/-30 s window at 1 fps there are up to 60 frames per event.

Filename pattern expected:
    {label}_frame_{event_tag}_t{frame_tag}.jpg
    e.g.  corner_frame_09-00.000_t08-45.000.jpg

Only the manifest's games are encoded. Encodings are cached per game half in
<out>/cache/, so adding games encodes only the new ones.

Outputs (in --out, default features/temporal_<model name>)
----------------------------------------------------------
  clip_temporal_padded.npz    - padded features (E, T, D) + masks (E, T) + labels,
                                games, halves, event_tags (name kept for compatibility)
  temporal_metadata.csv       - one row per event, with match_id and split
  run_info.json               - model, frame root and frame counts

Usage
-----
    python scripts/temporal_clip_features.py
    python scripts/temporal_clip_features.py --model facebook/dinov2-base --batch-size 64 --device cuda
"""

import argparse
import json
import pathlib

import numpy as np

from soccer.config import data_path, load_config, repo_path
from soccer.encoding import encode_cached, pick_device, save_outputs
from soccer.frames import scan_events
from soccer.manifest import load_manifest


def main(argv=None):
    cfg = load_config()
    fcfg = cfg["features"]
    parser = argparse.ArgumentParser(description="Extract per-event temporal features from soccer frames.")
    parser.add_argument("--frames-root", default=str(data_path(cfg, "frames")),
                        help="Root dir containing game_XX_half_N/frames/<label>/*.jpg")
    parser.add_argument("--out", default=None, help="default: features/temporal_<model name>")
    parser.add_argument("--model", default=fcfg["model"], help="HuggingFace model name (DINOv2 or CLIP)")
    parser.add_argument("--max-frames", type=int, default=fcfg["max_frames"],
                        help="Max frames kept per event (uniform subsample if exceeded)")
    parser.add_argument("--batch-size", type=int, default=fcfg["batch_size"])
    parser.add_argument("--device", default="auto",
                        help="cuda / cpu / auto (auto: GPU only if it has enough free memory)")
    args = parser.parse_args(argv)

    from soccer.encoders import build_encoder

    frames_root = pathlib.Path(args.frames_root)
    out_dir = pathlib.Path(args.out or data_path(cfg, "features") / f"temporal_{args.model.split('/')[-1]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"Device: {device}  Model: {args.model}  Max frames: {args.max_frames}")
    print(f"Input: {frames_root}\nOutput: {out_dir}")

    games = load_manifest(repo_path(cfg, "manifest"))
    wanted = {g.game_num for g in games}
    events = {k: v for k, v in scan_events(frames_root).items() if k[0] in wanted}
    if not events:
        raise SystemExit(f"No frames found under {frames_root}")
    n_per = [len(v) for v in events.values()]
    print(f"Found {len(events)} events across {len({k[2] for k in events})} labels; "
          f"frames per event min={min(n_per)} max={max(n_per)} mean={np.mean(n_per):.1f}")

    padded, masks, keys, n_frames, reused, encoded = encode_cached(
        lambda: build_encoder(args.model, device), events, args.max_frames, args.batch_size,
        out_dir / "cache", {"model": args.model, "max_frames": args.max_frames})
    print(f"Game halves: {reused} reused from cache, {encoded} encoded")
    df = save_outputs(out_dir, padded, masks, keys, n_frames, games)
    (out_dir / "run_info.json").write_text(json.dumps({
        "model": args.model, "frames_root": str(frames_root), "max_frames": args.max_frames,
        "events": len(keys), "frames_encoded": int(masks.sum()), "frames_found": int(sum(n_per)),
    }, indent=2))

    print(f"\nSaved {out_dir / 'clip_temporal_padded.npz'}  features {padded.shape}")
    print(f"Frames encoded: {int(masks.sum())} / {sum(min(n, args.max_frames) for n in n_per)}")
    print("\nLabel distribution (events):")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
