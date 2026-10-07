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


def test_events_past_the_end_are_skipped(video, tmp_path):
    events = events_for_half([{"half": 1, "time_stamp": "05:00", "comments_type": "goal"}], 1)
    stats = extract_events(video, events, tmp_path / "out", ExtractionSettings(before_sec=2, after_sec=2))
    assert stats.events == 0 and stats.skipped == 1
