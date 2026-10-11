import cv2
import numpy as np
import pytest

from soccer.annotations import detect_half, events_for_half, tag_to_seconds
from soccer.extract import FRAME_RE, ExtractionSettings, extract_events

FPS = 10
SECONDS = 20


@pytest.fixture
def video(tmp_path):
    """A 20 s, 10 fps video whose frame i has brightness 10 * (i // 10).

    Brightness therefore encodes the second, so a saved frame can be checked
    against the time in its filename.
    """
    path = tmp_path / "match_1.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    assert writer.isOpened()
    for i in range(FPS * SECONDS):
        writer.write(np.full((48, 64, 3), 10 * (i // FPS), dtype=np.uint8))
    writer.release()
    return path


def saved_frames(out_dir):
    return sorted(p for p in (out_dir / "frames").rglob("*.jpg"))


def test_frames_come_from_the_right_seconds(video, tmp_path):
    comments = [{"half": 1, "time_stamp": "00:10", "comments_type": "corner"}]
    events = events_for_half(comments, detect_half(video))
    settings = ExtractionSettings(before_sec=3, after_sec=3)
    stats = extract_events(video, events, tmp_path / "out", settings)

    frames = saved_frames(tmp_path / "out")
    assert stats.events == 1 and stats.frames == len(frames) == 6  # seconds 7..12, end exclusive
    # The decoder's own timestamps agree with the index-based times (exact seeking).
    assert stats.max_time_error_sec < 0.2
    for p in frames:
        m = FRAME_RE.search(p.name)
        assert m and m.group(1) == "00-10.000"
        second = tag_to_seconds(m.group(2))
        brightness = cv2.imread(str(p)).mean()
        assert abs(brightness - 10 * second) < 3, (p.name, brightness)


def test_other_half_events_are_never_extracted(video, tmp_path):
    # The second-half goal shares no footage with this first-half video.
    comments = [
        {"half": 1, "time_stamp": "00:05", "comments_type": "corner"},
        {"half": 2, "time_stamp": "00:15", "comments_type": "goal"},
    ]
    events = events_for_half(comments, detect_half(video))
    extract_events(video, events, tmp_path / "out", ExtractionSettings(before_sec=2, after_sec=2))
    labels = {p.parent.name for p in saved_frames(tmp_path / "out")}
    assert labels == {"corner"}


class VariableRateCapture:
    """Like the recordings in docs/leak_audit.md B19: claims 30 fps but has 25 frames a second,
    and a seek lands 2 s late. Frame brightness is 10 x its true second."""

    true_fps, claimed_fps, late, seconds = 25, 30.0, 2.0, 20

    def __init__(self, path):
        self.pos, self.last = 0, 0

    def isOpened(self):
        return True

    def release(self):
        pass

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.claimed_fps, cv2.CAP_PROP_FRAME_WIDTH: 64.0, cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
                cv2.CAP_PROP_POS_MSEC: 1000.0 * self.last / self.true_fps}[prop]

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_MSEC
        t = value / 1000.0 + (self.late if value > 0 else 0.0)
        self.pos = round(t * self.true_fps)

    def read(self):
        if self.pos >= self.true_fps * self.seconds:
            return False, None
        self.last, self.pos = self.pos, self.pos + 1
        return True, np.full((48, 64, 3), 10 * (self.last // self.true_fps), dtype=np.uint8)


def test_frames_follow_the_video_clock_when_the_frame_rate_is_wrong(monkeypatch, tmp_path):
    monkeypatch.setattr(cv2, "VideoCapture", VariableRateCapture)
    events = events_for_half([{"half": 1, "time_stamp": "00:10", "comments_type": "corner"}], 1)
    stats = extract_events("match_1.mkv", events, tmp_path / "out", ExtractionSettings(before_sec=3, after_sec=3))
    frames = saved_frames(tmp_path / "out")
    assert stats.frames == len(frames) == 6 and stats.max_time_error_sec < 1e-9
    for p in frames:
        second = tag_to_seconds(FRAME_RE.search(p.name).group(2))
        assert abs(cv2.imread(str(p)).mean() - 10 * second) < 3, p.name  # the picture is from that second


def test_events_past_the_end_are_skipped(video, tmp_path):
    events = events_for_half([{"half": 1, "time_stamp": "05:00", "comments_type": "goal"}], 1)
    stats = extract_events(video, events, tmp_path / "out", ExtractionSettings(before_sec=2, after_sec=2))
    assert stats.events == 0 and stats.skipped == 1
