"""Finding extracted frames on disk and grouping them into events.

Layout: <frames root>/game_XX_half_N/frames/<label>/<label>_frame_<event>_t<frame>.jpg
"""

from __future__ import annotations

import pathlib
from collections import defaultdict

import numpy as np

from soccer.annotations import tag_to_seconds
from soccer.extract import FRAME_RE
from soccer.manifest import normalize_game_id


def parse_half_dir(name: str) -> tuple[str, str] | None:
    """"game_07_half_2" -> ("07", "2"); None for anything else."""
    parts = name.split("_")
    if len(parts) == 4 and parts[0] == "game" and parts[2] == "half":
        return normalize_game_id(parts[1]), parts[3]
    return None


def scan_events(frames_root: str | pathlib.Path) -> dict:
    """(game, half, label, event_tag) -> [(frame_seconds, path), ...]"""
    events = defaultdict(list)
    for half_dir in sorted(p for p in pathlib.Path(frames_root).iterdir() if p.is_dir()):
        ids = parse_half_dir(half_dir.name)
        frames_dir = half_dir / "frames"
        if ids is None or not frames_dir.is_dir():
            continue
        for label_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
            for img in sorted(label_dir.iterdir()):
                m = FRAME_RE.search(img.name)
                if m:
                    events[(*ids, label_dir.name, m.group(1))].append((tag_to_seconds(m.group(2)), str(img)))
    return events


def select_frames(frames: list, max_frames: int) -> list:
    """Time-ordered frames, uniformly subsampled to at most max_frames."""
    frames = sorted(frames)
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]
    return frames
