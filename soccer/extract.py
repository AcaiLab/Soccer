"""Cut event-centred frames out of one half's broadcast video.

Output layout (unchanged from the original scripts, so existing frames and the
feature extractors keep working):

    <out_dir>/frames/<label>/<label>_frame_<event_tag>_t<frame_tag>.<ext>

`event_tag` is the commentary timestamp and `frame_tag` the saved frame's own
time in the half (its decoder timestamp), both as "MM-SS.sss".

Frames are chosen by the video's timestamps, not by counting frames: some of
these recordings have a variable frame rate, and their nominal fps is wrong
(docs/leak_audit.md, B19). Counting frames in such a video drifts by seconds
within one event window and by minutes over a half.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from soccer.annotations import Event, seconds_to_tag

FRAME_RE = re.compile(r"_frame_(\d{2}-\d{2}\.\d{3})_t(\d{2}-\d{2}\.\d{3})\.(?:png|jpg)$")
DONE_MARKER = "extraction.json"  # written into a half's folder once its extraction is complete
VIDEO_EXTS = (".mkv", ".mp4")
FRAME_SELECTION = "timestamp"  # recorded in each marker; markers without it chose frames by index
# A saved frame more than this far from the second it stands for is reported:
# the video has a gap there.
MAX_TIME_ERROR_SEC = 0.5


@dataclass(frozen=True)
class ExtractionSettings:
    before_sec: float = 30
    after_sec: float = 30
    sample_every_sec: int = 1
    image_format: str = "jpg"
    jpeg_quality: int = 90
    write_clips: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> "ExtractionSettings":
        return cls(**cfg["extraction"])


@dataclass
class ExtractionStats:
    events: int = 0
    frames: int = 0
    skipped: int = 0
    # Largest gap, over all saved frames, between a frame's timestamp and the
    # second it was chosen for (about half a frame interval in a gap-free video).
    max_time_error_sec: float = 0.0


def frame_filename(label: str, event: Event, frame_seconds: float, ext: str) -> str:
    return f"{label}_frame_{seconds_to_tag(event.seconds)}_t{seconds_to_tag(frame_seconds)}.{ext}"


def extract_events(video_path: str | pathlib.Path, events: list[Event],
                   out_dir: str | pathlib.Path,
                   settings: ExtractionSettings | None = None) -> ExtractionStats:
    """Save the frame nearest each second from `before_sec` before to `after_sec` after each event.

    `events` must all belong to the half this video holds; use
    annotations.events_for_half() to get them.
    """
    import cv2  # imported here so constants above load without OpenCV (e.g. in the transfer Lambda)

    settings = settings or ExtractionSettings()
    out_dir = pathlib.Path(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError(f"{video_path} reports no frame rate")
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        stats = ExtractionStats()
        for event in events:
            n, err = _extract_one(cap, fps, size, event, out_dir, settings)
            stats.max_time_error_sec = max(stats.max_time_error_sec, err)
            if n:
                stats.events += 1
                stats.frames += n
            else:
                stats.skipped += 1
        if stats.max_time_error_sec > MAX_TIME_ERROR_SEC:
            print(f"  WARNING: {video_path}: some saved frames are up to {stats.max_time_error_sec:.2f}s "
                  f"from the second they stand for (a gap in the video)", flush=True)
        return stats
    finally:
        cap.release()


def _read(cap):
    """(frame, its timestamp in seconds), or (None, None) at the end of the video."""
    import cv2

    ok, frame = cap.read()
    return (frame, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) if ok else (None, None)


def _seek_before(cap, t: float):
    """Position so the next frame read is at or before t; returns that first (frame, timestamp).

    A seek can land a little late (seconds, in some of these files), so step back
    until the decoder's own timestamp is not past t.
    """
    import cv2

    back = 1.0
    while True:
        target = max(0.0, t - back)
        cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
        frame, ts = _read(cap)
        if frame is None or ts <= t or target == 0.0:
            return frame, ts
        back *= 4


def _extract_one(cap, fps, size, event, out_dir, settings) -> tuple[int, float]:
    """Returns (frames saved, largest timestamp-vs-target gap among them)."""
    import cv2

    start_sec = max(0.0, event.seconds - settings.before_sec)
    end_sec = event.seconds + settings.after_sec  # exclusive; the video's end stops reading anyway
    step = settings.sample_every_sec
    targets = [start_sec + k * step for k in range(int((end_sec - start_sec) / step) + 1)
               if start_sec + k * step < end_sec]
    if not targets:
        return 0, 0.0

    frame, ts = _seek_before(cap, start_sec)
    if frame is None:
        return 0, 0.0  # the event lies past the end of the video
    frames_dir = out_dir / "frames" / event.label
    frames_dir.mkdir(parents=True, exist_ok=True)
    clip = None
    if settings.write_clips:
        clips_dir = out_dir / "clips" / event.label
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clips_dir / f"{event.label}_clip_{seconds_to_tag(event.seconds)}.mp4"
        clip = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    params = ([cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
              if settings.image_format == "jpg" else [])

    saved, worst, i = 0, 0.0, 0
    written = set()
    prev = None  # (frame, timestamp) read just before the current one
    try:
        while frame is not None and i < len(targets):
            # Every target the video has now passed gets the nearer of the two frames around it.
            while i < len(targets) and ts >= targets[i]:
                best = (frame, ts) if prev is None or ts - targets[i] <= targets[i] - prev[1] else prev
                if best[1] not in written:  # a gap can make two seconds share a frame: save it once
                    path = frames_dir / frame_filename(event.label, event, best[1], settings.image_format)
                    if not cv2.imwrite(str(path), best[0], params):
                        raise RuntimeError(f"Failed to write {path}")
                    written.add(best[1])
                    saved += 1
                worst = max(worst, abs(best[1] - targets[i]))
                i += 1
            if clip is not None and start_sec <= ts < end_sec:
                clip.write(frame)
            prev = (frame, ts)
            frame, ts = _read(cap)
    finally:
        if clip is not None:
            clip.release()
    return saved, worst
