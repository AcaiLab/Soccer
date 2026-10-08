"""
CLIP Feature Extraction for Soccer Event Frames
================================================
Per-frame CLIP (openai/clip-vit-base-patch32) embeddings, the baseline that
DINOv2 is compared against.

Superseded by `python extract_frame_features.py --model openai/clip-vit-base-patch32`,
which this forwards to; kept until its removal is approved (docs/cleanup_log.md).

Usage:
    python clip_feature_extraction.py [--batch-size 64 --device cuda]
"""

import sys

import extract_frame_features

CLIP_MODEL = "openai/clip-vit-base-patch32"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extract_frame_features.main(["--model", CLIP_MODEL, *argv])


if __name__ == "__main__":
    main()
