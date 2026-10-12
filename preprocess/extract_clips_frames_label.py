"""Extract event-centred frames from one half's video, organised by event label.

Only the events of the half the video holds are extracted: comment timestamps
restart at 0:00 each half, so cutting the other half's events would attach
their labels to unrelated footage (docs/leak_audit.md, B3).

Output: <out>/frames/<label>/<label>_frame_<event MM-SS.sss>_t<frame MM-SS.sss>.jpg

Usage:
    python preprocess/extract_clips_frames_label.py \
        --video data/videos/<match>/<match>_1.mkv --json <match>.json \
        --out frames_v2/game_01_half_1

Window, sampling rate and image format default to configs/pipeline.yaml.
Requires the repo package to be importable (`pip install -e .`).
"""

import argparse

from soccer.annotations import detect_half, events_for_half, load_comments
from soccer.config import load_config
from soccer.extract import ExtractionSettings, extract_events


def main(argv=None):
    ex = load_config()["extraction"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, help="one half's video, named *_1.* or *_2.*")
    parser.add_argument("--json", required=True, help="the match's annotation JSON")
    parser.add_argument("--out", required=True, help="output folder for this half")
    parser.add_argument("--half", type=int, choices=[1, 2], default=None,
                        help="override the half detected from the video file name")
    parser.add_argument("--before", type=float, default=ex["before_sec"])
    parser.add_argument("--after", type=float, default=ex["after_sec"])
    parser.add_argument("--every", type=int, default=ex["sample_every_sec"])
    parser.add_argument("--format", choices=["jpg", "png"], default=ex["image_format"])
    parser.add_argument("--clips", action="store_true", default=ex["write_clips"],
                        help="also write an .mp4 clip per event")
    args = parser.parse_args(argv)

    half = args.half or detect_half(args.video)
    comments = load_comments(args.json)
    events = events_for_half(comments, half)
    other = sum(1 for c in comments if c.get("time_stamp") and c.get("half") in (1, 2) and c["half"] != half)
    print(f"Half {half}: {len(events)} events to extract, {other} timestamped comments from the other half skipped")

    settings = ExtractionSettings(args.before, args.after, args.every, args.format,
                                  ex["jpeg_quality"], args.clips)
    stats = extract_events(args.video, events, args.out, settings)
    print(f"Extracted {stats.frames} frames for {stats.events} events "
          f"({stats.skipped} outside the video) -> {args.out}")


if __name__ == "__main__":
    main()
