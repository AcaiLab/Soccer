"""
DINOv2 Soccer Event Clustering Pipeline
========================================
Reproducible end-to-end pipeline for extracting DINOv2 embeddings
and clustering soccer events.

Three-stage pipeline:
  1. Temporal features from all 627 events (per-event tensors)
  2. Flat per-frame features (optional, for comprehensive analysis)
  3. K-Means + t-SNE clustering with multiple visualizations

Usage:
    python run_dino_pipeline.py
    python run_dino_pipeline.py --model facebook/dinov2-large --batch-size 64
    python run_dino_pipeline.py --skip-extraction  # re-run clustering only
"""

import argparse
import pathlib
import subprocess
import sys
from datetime import datetime


def log(msg):
    """Print with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def run_cmd(cmd, description):
    """Run a shell command and log status."""
    log(f"Starting: {description}")
    log(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log(f"ERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    log(f"Completed: {description}\n")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end DINOv2 extraction and clustering pipeline"
    )
    parser.add_argument(
        "--model",
        default="facebook/dinov2-base",
        help="Vision model to use (default: facebook/dinov2-base)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoder (default: 32)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda/cpu, auto-detected if omitted)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip feature extraction, only run clustering",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Max frames per event for temporal extraction (default: 60)",
    )
    args = parser.parse_args()

    log("DINOv2 Soccer Event Clustering Pipeline")
    log(f"Model: {args.model}")
    log(f"Batch size: {args.batch_size}")
    if args.device:
        log(f"Device: {args.device}")
    log("")

    # Extract model prefix for output directory naming
    model_prefix = args.model.split("/")[-1]
    temporal_out = f"features/temporal_{model_prefix}"
    flat_out = f"features/{model_prefix}_flat"
    cluster_out = f"clustering/{model_prefix}"

    if not args.skip_extraction:
        # Stage 1: Temporal per-event features
        log("="*70)
        log("STAGE 1: Temporal Feature Extraction (627 events)")
        log("="*70)
        cmd_temporal = [
            sys.executable, "temporal_clip_features.py",
            "--model", args.model,
            "--out", temporal_out,
            "--max-frames", str(args.max_frames),
            "--batch-size", str(args.batch_size),
        ]
        if args.device:
            cmd_temporal.extend(["--device", args.device])
        run_cmd(cmd_temporal, f"Extract temporal features -> {temporal_out}")

        # Stage 2: Flat per-frame features
        log("="*70)
        log("STAGE 2: Flat Per-Frame Feature Extraction (18,501 frames)")
        log("="*70)
        cmd_flat = [
            sys.executable, "extract_frame_features.py",
            "--model", args.model,
            "--out", flat_out,
            "--batch-size", str(args.batch_size),
        ]
        if args.device:
            cmd_flat.extend(["--device", args.device])
        run_cmd(cmd_flat, f"Extract flat features -> {flat_out}")
    else:
        log("Skipping feature extraction (--skip-extraction flag set)\n")

    # Stage 3: Clustering and visualization
    log("="*70)
    log("STAGE 3: K-Means Clustering & Visualization")
    log("="*70)
    flat_features_path = f"{flat_out}/{model_prefix}_features.npy"
    flat_meta_path = f"{flat_out}/metadata.csv"

    cmd_cluster = [
        sys.executable, "cluster_analysis.py",
        "--features-dir", temporal_out,
        "--flat-features", flat_features_path,
        "--flat-meta", flat_meta_path,
        "--out-dir", cluster_out,
    ]
    run_cmd(cmd_cluster, f"Run clustering -> {cluster_out}")

    # Summary
    log("="*70)
    log("PIPELINE COMPLETE")
    log("="*70)
    log(f"Outputs saved to:")
    log(f"  Temporal features: {temporal_out}/")
    log(f"  Flat features:     {flat_out}/")
    log(f"  Clustering:        {cluster_out}/")
    log(f"\nRandom seed used: 42 (for reproducibility)")
    log(f"To re-run clustering only, use: python run_dino_pipeline.py --skip-extraction")


if __name__ == "__main__":
    main()
