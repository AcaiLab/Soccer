"""Cut event-centred frames out of one half's broadcast video.

Output layout (unchanged from the original scripts, so existing frames and the
feature extractors keep working):

    <out_dir>/frames/<label>/<label>_frame_<event_tag>_t<frame_tag>.<ext>

`event_tag` is the commentary timestamp and `frame_tag` the frame's own time in
the half, both as "MM-SS.sss".
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from soccer.annotations import Event, seconds_to_tag

FRAME_RE = re.compile(r"_frame_(\d{2}-\d{2}\.\d{3})_t(\d{2}-\d{2}\.\d{3})\.(?:png|jpg)$")
DONE_MARKER = "extraction.json"  # written into a half's folder once its extraction is complete
VIDEO_EXTS = (".mkv", ".mp4")
# A frame whose decoded timestamp differs from its index-based time by more
# than this is reported: it would mean seeking or frame counting went wrong.
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
    # Largest gap, over all saved frames, between the decoder's timestamp and
    # the time implied by the frame index (0 for exact seeking, constant fps).
    max_time_error_sec: float = 0.0


def frame_filename(label: str, event: Event, frame_seconds: float, ext: str) -> str:
    return f"{label}_frame_{seconds_to_tag(event.seconds)}_t{seconds_to_tag(frame_seconds)}.{ext}"


def extract_events(video_path: str | pathlib.Path, events: list[Event],
                   out_dir: str | pathlib.Path,
                   settings: ExtractionSettings | None = None) -> ExtractionStats:
    """Save one frame every `sample_every_sec` seconds around each event.

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
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        stats = ExtractionStats()
        for event in events:
            n, err = _extract_one(cap, fps, duration, size, event, out_dir, settings)
            stats.max_time_error_sec = max(stats.max_time_error_sec, err)
            if n:
                stats.events += 1
                stats.frames += n
            else:
                stats.skipped += 1
        if stats.max_time_error_sec > MAX_TIME_ERROR_SEC:
            print(f"  WARNING: {video_path}: decoded frame times differ from their index by up to "
                  f"{stats.max_time_error_sec:.2f}s (inexact seeking or variable frame rate)", flush=True)
        return stats
    finally:
        cap.release()


def _extract_one(cap, fps, duration, size, event, out_dir, settings) -> tuple[int, float]:
    """Returns (frames saved, largest decoder-vs-index time error among them)."""
    import cv2

    start_sec = max(0.0, event.seconds - settings.before_sec)
    end_sec = min(duration, event.seconds + settings.after_sec)
    start_f, end_f = round(start_sec * fps), round(end_sec * fps)
    if end_f <= start_f:
        return 0, 0.0

    frames_dir = out_dir / "frames" / event.label
    frames_dir.mkdir(parents=True, exist_ok=True)
    clip = None
    if settings.write_clips:
        clips_dir = out_dir / "clips" / event.label
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clips_dir / f"{event.label}_clip_{seconds_to_tag(event.seconds)}.mp4"
        clip = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    save_idxs = {round((start_sec + t) * fps)
                 for t in range(0, round(end_sec - start_sec) + 1, settings.sample_every_sec)}
    params = ([cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
              if settings.image_format == "jpg" else [])

    # Frames are counted from start_f after a frame-index seek. That was checked
    # to be exact on these videos (docs/leak_audit.md, B3); every saved frame is
    # still compared with the decoder's own timestamp, and the worst gap is
    # returned so each half's marker records how exact its frame times are.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    saved, worst = 0, 0.0
    try:
        for idx in range(start_f, end_f):
            ok, frame = cap.read()
            if not ok:
                break
            if clip is not None:
                clip.write(frame)
            if idx in save_idxs:
                decoded = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if decoded > 0 or idx == 0:  # some backends report 0 when unknown
                    worst = max(worst, abs(decoded - idx / fps))
                path = frames_dir / frame_filename(event.label, event, idx / fps, settings.image_format)
                if not cv2.imwrite(str(path), frame, params):
                    raise RuntimeError(f"Failed to write {path}")
                saved += 1
    finally:
        if clip is not None:
            clip.release()
    return saved, worst
