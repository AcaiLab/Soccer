"""
DINOv2 Soccer Event Pipeline - entry point
==========================================
Runs the whole pipeline, or any subset of its stages, in order:

  frames       download videos, extract event frames, delete videos   (CPU; batch_pipeline.py)
  features     encode frames with the frozen model                     (GPU; temporal_clip_features.py,
                                                                        extract_frame_features.py)
  cluster      K-Means + t-SNE, within-game clustering                 (CPU)
  contrastive  triplet head on training games, held-out evaluation     (GPU or CPU)
  classify     pooled, sequence, annotation-only and multimodal        (CPU; sequence models prefer GPU)

Which games are used and how they are split comes from configs/games.csv;
paths, window and model from configs/pipeline.yaml. Each stage runs as its own
process, so GPU memory is released between stages, and the run stops at the
first failure.

Usage:
    python run_dino_pipeline.py                                  # every stage
    python run_dino_pipeline.py --stages features cluster        # a subset
    python run_dino_pipeline.py --stages frames --local-only     # extract from local videos
    python run_dino_pipeline.py --model facebook/dinov2-large --device cuda
"""

import argparse
import subprocess
import sys
from datetime import datetime

from soccer.config import data_path, load_config

STAGES = ["frames", "features", "cluster", "contrastive", "classify"]


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run(script, *args):
    cmd = [sys.executable, script, *[str(a) for a in args]]
    log("$ " + " ".join(cmd[1:]))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log(f"ERROR: {script} exited with {result.returncode}")
        sys.exit(result.returncode)


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Run the soccer event pipeline end to end.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--model", default=cfg["features"]["model"])
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    parser.add_argument("--batch-size", type=int, default=cfg["features"]["batch_size"])
    parser.add_argument("--games", nargs="+", default=None, help="frames stage: only these game numbers")
    parser.add_argument("--local-only", action="store_true", help="frames stage: use videos already on disk")
    parser.add_argument("--skip-flat", action="store_true", help="features stage: skip per-frame features")
    args = parser.parse_args(argv)

    device = ["--device", args.device] if args.device else []
    prefix = args.model.split("/")[-1]
    log(f"Stages: {' '.join(args.stages)} | model {args.model} | data root {cfg['data_root']}")

    if "frames" in args.stages:
        run("batch_pipeline.py", *(["--games", *args.games] if args.games else []),
            *(["--local-only"] if args.local_only else []))
    if "features" in args.stages:
        run("temporal_clip_features.py", "--model", args.model, "--batch-size", args.batch_size, *device)
        if not args.skip_flat:
            run("extract_frame_features.py", "--model", args.model, "--batch-size", args.batch_size, *device)
    if "cluster" in args.stages:
        run("cluster_analysis.py", "--model", args.model)
        run("within_game_clustering.py",
            "--features-dir", data_path(cfg, "features") / f"temporal_{prefix}")
    if "contrastive" in args.stages:
        run("contrastive_training.py", *device)
        run("evaluate_contrastive.py")
    if "classify" in args.stages:
        run("event_classification.py")
        run("sequence_classifier.py", *device)
        if data_path(cfg, "annotations").is_dir():
            run("game_flow_features.py")
            run("multimodal_classification.py")
        else:
            log("No annotation folder; skipping annotation-only and multimodal classification")
    log("Pipeline complete")


if __name__ == "__main__":
    main()
