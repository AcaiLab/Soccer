"""How well do a video's own timestamps agree with its frame count?

soccer.extract picks frames by index (time x fps), which is exact only when the video
has a constant frame rate and no gaps. profile_video measures the difference between the
decoder's timestamp of a frame and the time its index implies:

  * samples: that difference at regular points, found with the same seek the extraction uses;
  * jumps (scan=True): every place where consecutive frames are further apart, or closer,
    than 1/fps by more than `jump_sec`, found by decoding the whole video, with the size of
    the jump and how much the picture changes across it.

A jump across which the picture barely changes is a timestamp glitch in a continuous
recording. A jump with a large picture change may be missing footage (or an ordinary
camera cut), which thumbnails of the two frames settle.
"""

from __future__ import annotations

import collections
import pathlib
from dataclasses import dataclass, field

THUMB_SIZE = (160, 90)
MAX_THUMBS = 20  # keep pictures only for the first jumps of a video


@dataclass
class Jump:
    index: int                              # frame whose timestamp jumps relative to the previous frame
    index_time: float                       # index / fps
    seconds: float                          # timestamp step minus 1/fps (positive: time is missing)
    picture_change: float | None = None     # mean absolute pixel difference (0-255) across the jump
    typical_change: float | None = None     # median difference between neighbouring frames just before
    before: object = None                   # small images of the two frames, for a human check
    after: object = None


@dataclass
class TimingProfile:
    path: str
    fps: float
    frames: int
    scanned: bool = False
    samples: list[tuple[float, float]] = field(default_factory=list)  # (index time, decoded - index time)
    jumps: list[Jump] = field(default_factory=list)
    final_error: float | None = None        # decoded - index time at the last frame (scan only)

    @property
    def max_error(self) -> float:
        errors = [abs(e) for _, e in self.samples]
        if self.final_error is not None:
            errors.append(abs(self.final_error))
        return max(errors, default=0.0)


def profile_video(path: str | pathlib.Path, step_sec: float = 30.0, scan: bool = False,
                  jump_sec: float = 0.2) -> TimingProfile:
    """Timing profile of one video; `scan` decodes all of it (minutes) to locate every jump."""
    import cv2  # imported here so the module loads without OpenCV

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError(f"{path} reports no frame rate")
        profile = TimingProfile(str(path), fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), scanned=scan)
        t = 0.0
        while t * fps < profile.frames:
            idx = round(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, _ = cap.read()
            if not ok:
                break
            profile.samples.append((idx / fps, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - idx / fps))
            t += step_sec
        if scan:
            _scan(cap, profile, jump_sec)
        return profile
    finally:
        cap.release()


def _scan(cap, profile: TimingProfile, jump_sec: float) -> None:
    import cv2
    import numpy as np

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    step = 1.0 / profile.fps
    recent = collections.deque(maxlen=25)
    prev_pts = prev_thumb = prev_gray = last = None
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        known = pts > 0 or i == 0  # some backends report 0 when the time is unknown
        thumb = cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev_gray is not None:
            change = float(np.abs(gray - prev_gray).mean())
            if known and prev_pts is not None and abs(pts - prev_pts - step) > jump_sec:
                keep = len(profile.jumps) < MAX_THUMBS
                profile.jumps.append(Jump(
                    index=i, index_time=i / profile.fps, seconds=pts - prev_pts - step,
                    picture_change=change, typical_change=float(np.median(recent)) if recent else None,
                    before=prev_thumb if keep else None, after=thumb if keep else None))
            else:
                recent.append(change)
        if known:
            prev_pts, last = pts, (i, pts)
        prev_thumb, prev_gray = thumb, gray
    if last is not None:
        profile.final_error = last[1] - last[0] / profile.fps
