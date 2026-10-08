"""Reading SoccerReplay-1988 match annotations.

Each match JSON has a `comments` list. A comment carries `half` (0 = pre-match,
1, 2), `time_stamp` ("MM:SS" or "MM+added:SS", restarting at 0:00 in each half)
and `comments_type`, the event label.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass

TIMESTAMP_RE = re.compile(r"(\d+)(?:\+(\d+))?:(\d{2})")
SPLITS = ("train", "valid", "test")
LEAGUE_DIR_RE = re.compile(r"^[a-z]+_[a-z0-9-]+_\d{4}-\d{4}$")  # e.g. england_epl_2021-2022


@dataclass(frozen=True)
class Event:
    half: int
    timestamp: str      # as written in the JSON, e.g. "45+2:30"
    seconds: int        # seconds into the half, stoppage time included
    label: str          # sanitised, e.g. "lead_to_corner"


def parse_timestamp(ts: str) -> int:
    """Seconds into the half: "45+2:30" -> (45 + 2) * 60 + 30 = 2850."""
    m = TIMESTAMP_RE.fullmatch(ts.strip()) if ts else None
    if not m:
        raise ValueError(f"Unrecognized time format: {ts!r}")
    return (int(m[1]) + int(m[2] or 0)) * 60 + int(m[3])


def seconds_to_tag(seconds: float) -> str:
    """Filename-safe time tag: 540 -> "09-00.000"."""
    m = int(seconds // 60)
    return f"{m:02d}-{seconds - 60 * m:06.3f}"


def tag_to_seconds(tag: str) -> float:
    """Inverse of seconds_to_tag: "09-00.000" -> 540.0."""
    mm, ss = tag.split("-", 1)
    return int(mm) * 60.0 + float(ss)


def sanitize_label(label: str | None) -> str:
    """Folder-safe label: "Lead to corner" -> "lead_to_corner"; empty -> "unknown"."""
    s = (label or "unknown").strip().lower().replace("&", "and")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "unknown"


def match_minute(seconds: int, half: int) -> float:
    """Minute in the match, second half offset by 45, rounded like the features."""
    return round(seconds / 60.0 + (45.0 if half == 2 else 0.0), 2)


def load_comments(json_path: str | pathlib.Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f).get("comments", [])


def events_for_half(comments: list[dict], half: int) -> list[Event]:
    """Timestamped events of one half, de-duplicated and sorted by time.

    Comments from the other half are excluded: their timestamps also start at
    0:00, so cutting them from this half's video would attach their labels to
    unrelated footage (docs/leak_audit.md, B3).
    """
    seen = set()
    events = []
    for c in comments:
        ts = c.get("time_stamp") or ""
        if c.get("half") != half or not ts:
            continue
        label = sanitize_label(c.get("comments_type"))
        key = (ts, label)
        if key in seen:
            continue
        seen.add(key)
        try:
            events.append(Event(half, ts, parse_timestamp(ts), label))
        except ValueError:
            continue
    events.sort(key=lambda e: e.seconds)
    return events


def find_annotation_file(annotations_root: str | pathlib.Path, league: str,
                         match_id: str) -> tuple[pathlib.Path, str] | None:
    """Locate a match's JSON under <root>/<split>/<league>/<match_id>/.

    Returns (path, official_split) so callers can check the match is in the
    split they expect, or None if it is missing.
    """
    root = pathlib.Path(annotations_root)
    for split in SPLITS:
        folder = root / split / league / match_id
        # Hidden files are skipped: zips made on macOS carry "._<name>.json"
        # metadata stubs that would otherwise sort before the real file.
        jsons = sorted(p for p in folder.glob("*.json") if not p.name.startswith(".")) if folder.is_dir() else []
        if jsons:
            return jsons[0], split
    return None


def detect_half(video_path: str | pathlib.Path) -> int:
    """Half held by a video named like "<match>_1.mkv" or "<match>_2.mkv"."""
    m = re.search(r"_([12])\.[A-Za-z0-9]+$", str(video_path))
    if not m:
        raise ValueError(f"Cannot tell which half {str(video_path)!r} holds; pass the half explicitly")
    return int(m.group(1))


def install_split_zip(zip_path: str | pathlib.Path, split: str, annotations_root: str | pathlib.Path) -> int:
    """Unpack a SoccerReplay split zip into <root>/<split>/<league>/<match>/*.json.

    The league folders are located wherever they sit inside the archive, so a
    zip with or without a top-level "train/" folder installs the same way.
    Returns the number of JSON files installed.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    dest = pathlib.Path(annotations_root) / split
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        leagues = [d for d in pathlib.Path(tmp).rglob("*") if d.is_dir() and LEAGUE_DIR_RE.match(d.name)
                   and "__MACOSX" not in d.parts]
        if not leagues:
            raise ValueError(f"{zip_path}: no league folders (like england_epl_2021-2022) inside")
        for league in leagues:
            shutil.copytree(league, dest / league.name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("._*", ".DS_Store"))
    return sum(1 for p in dest.glob("*/*/*.json") if not p.name.startswith("."))
