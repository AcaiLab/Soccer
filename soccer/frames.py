"""Finding extracted frames on disk and grouping them into events.

Layout: <frames root>/game_XX_half_N/frames/<label>/<label>_frame_<event>_t<frame>.jpg
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import numpy as np

from soccer.annotations import tag_to_seconds
from soccer.extract import DONE_MARKER, FRAME_RE, MAX_TIME_ERROR_SEC
from soccer.manifest import normalize_game_id


def parse_half_dir(name: str) -> tuple[str, str] | None:
    """"game_07_half_2" -> ("07", "2"); None for anything else."""
    parts = name.split("_")
    if len(parts) == 4 and parts[0] == "game" and parts[2] == "half":
        return normalize_game_id(parts[1]), parts[3]
    return None


def scan_events(frames_root: str | pathlib.Path, require_marker: bool = True) -> dict:
    """(game, half, label, event_tag) -> [(frame_seconds, path), ...]

    Halves without a completion marker are skipped (with a warning): their
    extraction was interrupted, so their frames are incomplete.
    """
    events = defaultdict(list)
    incomplete = []
    for half_dir in sorted(p for p in pathlib.Path(frames_root).iterdir() if p.is_dir()):
        ids = parse_half_dir(half_dir.name)
        frames_dir = half_dir / "frames"
        if ids is None or not frames_dir.is_dir():
            continue
        if require_marker and not (half_dir / DONE_MARKER).is_file():
            incomplete.append(half_dir.name)
            continue
        for label_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
            for img in sorted(label_dir.iterdir()):
                m = FRAME_RE.search(img.name)
                if m:
                    events[(*ids, label_dir.name, m.group(1))].append((tag_to_seconds(m.group(2)), str(img)))
    if incomplete:
        print(f"  WARNING: skipped {len(incomplete)} half folder(s) without {DONE_MARKER} "
              f"(interrupted extraction): {', '.join(incomplete[:5])}{' ...' if len(incomplete) > 5 else ''}",
              flush=True)
    return events


def time_check(frames_root: str | pathlib.Path, games: set[str] | None = None,
               tolerance: float = MAX_TIME_ERROR_SEC) -> dict:
    """What the extraction markers say about frame timing (see soccer.extract.MAX_TIME_ERROR_SEC).

    Returns halves (complete halves seen), events, unchecked (halves extracted before the
    marker recorded timing), flagged ({half: seconds} above the tolerance) and flagged_events.
    """
    out = {"halves": 0, "events": 0, "unchecked": [], "flagged": {}, "flagged_events": 0}
    for half_dir in sorted(p for p in pathlib.Path(frames_root).iterdir() if p.is_dir()):
        ids = parse_half_dir(half_dir.name)
        marker = half_dir / DONE_MARKER
        if ids is None or not marker.is_file() or (games is not None and ids[0] not in games):
            continue
        info = json.loads(marker.read_text(encoding="utf-8"))
        out["halves"] += 1
        out["events"] += info.get("events", 0)
        error = info.get("max_time_error_sec")
        if error is None:
            out["unchecked"].append(half_dir.name)
        elif error > tolerance:
            out["flagged"][half_dir.name] = error
            out["flagged_events"] += info.get("events", 0)
    return out


def describe_time_check(check: dict) -> str:
    checked = check["halves"] - len(check["unchecked"])
    text = f"{checked} of {check['halves']} halves have a frame-timing check"
    if check["unchecked"]:
        text += f" ({len(check['unchecked'])} extracted before the check existed)"
    flagged = check["flagged"]
    if not flagged:
        return text + "; none is above the tolerance"
    worst = ", ".join(f"{name} {seconds:.1f}s" for name, seconds in sorted(flagged.items()))
    return (text + f"; {len(flagged)} above the tolerance, holding {check['flagged_events']} of "
            f"{check['events']} events: {worst}")


def half_marker(frame_path: str | pathlib.Path) -> str:
    """Contents of the completion marker of the half a frame belongs to ('' if absent)."""
    marker = pathlib.Path(frame_path).parents[2] / DONE_MARKER  # <half>/frames/<label>/<file>
    return marker.read_text(encoding="utf-8") if marker.is_file() else ""


def select_frames(frames: list, max_frames: int) -> list:
    """Time-ordered frames, uniformly subsampled to at most max_frames."""
    frames = sorted(frames)
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]
    return frames
