"""Timing audit: does the decoder's clock agree with the frame index? Real and simulated videos."""

import json

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from audit_video_timing import main as audit_main  # noqa: E402

from soccer.frames import describe_time_check, time_check  # noqa: E402
from soccer.timing import profile_video  # noqa: E402


def write_marker(root, name, events, error=None):
    """A finished half as the extraction leaves it; old markers have no timing field."""
    (root / name / "frames").mkdir(parents=True)
    info = {"events": events, "frames": events * 60}
    if error is not None:
        info["max_time_error_sec"] = error
    (root / name / "extraction.json").write_text(json.dumps(info))


def test_time_check_counts_flagged_and_unchecked_halves(tmp_path):
    write_marker(tmp_path, "game_01_half_1", 30)               # extracted before the check existed
    write_marker(tmp_path, "game_02_half_1", 31, error=0.04)
    write_marker(tmp_path, "game_02_half_2", 33, error=13.9)
    write_marker(tmp_path, "game_99_half_1", 40, error=20.0)   # not one of the games asked about
    (tmp_path / "game_03_half_1" / "frames").mkdir(parents=True)  # interrupted: no marker
    check = time_check(tmp_path, {"01", "02", "03"})
    assert check == {"halves": 3, "events": 94, "unchecked": ["game_01_half_1"],
                     "flagged": {"game_02_half_2": 13.9}, "flagged_events": 33}
    assert describe_time_check(check) == (
        "2 of 3 halves have a frame-timing check (1 extracted before the check existed); "
        "1 above the tolerance, holding 33 of 94 events: game_02_half_2 13.9s")
    assert describe_time_check(time_check(tmp_path, {"02"})).endswith("game_02_half_2 13.9s")
    assert describe_time_check(time_check(tmp_path, {"01"})) == (
        "0 of 1 halves have a frame-timing check (1 extracted before the check existed), "
        "so their frame timing is unverified")


def write_video(path, frames=100, fps=25.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 72))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG video")
    for i in range(frames):
        image = np.zeros((72, 96, 3), np.uint8)
        image[:, (i * 2) % 96:(i * 2) % 96 + 8] = 255  # a bar moving across, so neighbouring frames differ a little
        writer.write(image)
    writer.release()


class FakeCapture:
    """A decoder whose timestamps jump by 13.8 s at frame 50, where the picture also cuts from black to white."""

    jump_at, jump_sec, drift = 50, 13.8, 1.0

    def __init__(self, path):
        self.n = 100
        self.pos = 0
        self.last = 0

    def isOpened(self):
        return True

    def release(self):
        pass

    def pts(self, i):
        return i * 0.04 * self.drift + (self.jump_sec if i >= self.jump_at else 0.0)

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 25.0, cv2.CAP_PROP_FRAME_COUNT: float(self.n),
                cv2.CAP_PROP_POS_MSEC: self.pts(self.last) * 1000.0}[prop]

    def set(self, prop, value):
        self.pos = int(value)

    def read(self):
        if self.pos >= self.n:
            return False, None
        self.last = self.pos
        self.pos += 1
        value = 255 if self.last >= self.jump_at else 0
        return True, np.full((72, 96, 3), value, np.uint8)


def test_a_constant_frame_rate_video_has_no_timing_error(tmp_path):
    write_video(tmp_path / "steady.avi")
    profile = profile_video(tmp_path / "steady.avi", step_sec=1.0, scan=True)
    assert profile.fps == 25.0 and profile.frames == 100
    assert profile.max_error < 1e-6 and not profile.jumps
    assert len(profile.samples) == 4  # t = 0, 1, 2, 3 s


def test_a_timestamp_jump_is_located_and_measured(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    profile = profile_video("any.mkv", step_sec=1.0, scan=True)
    assert [round(e, 1) for _, e in profile.samples] == [0.0, 0.0, 13.8, 13.8]
    (jump,) = profile.jumps
    assert jump.index == 50 and jump.index_time == 2.0
    assert jump.seconds == pytest.approx(13.8)
    assert jump.picture_change == 255 and jump.typical_change == 0  # black -> white across the jump
    assert jump.before is not None and jump.after is not None
    assert profile.final_error == pytest.approx(13.8) and profile.max_error == pytest.approx(13.8)


def test_slow_drift_shows_in_the_error_but_is_not_a_jump(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(FakeCapture, "jump_sec", 0.0)
    monkeypatch.setattr(FakeCapture, "drift", 1.02)  # the decoder's clock runs 2% slow
    profile = profile_video("any.mkv", step_sec=1.0, scan=True)
    assert not profile.jumps
    assert profile.final_error == pytest.approx(99 * 0.04 * 0.02)


def test_without_a_scan_only_the_seek_samples_are_taken(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    profile = profile_video("any.mkv", step_sec=1.0, scan=False)
    assert profile.final_error is None and not profile.jumps and not profile.scanned
    assert profile.max_error == pytest.approx(13.8)


def test_a_file_that_cannot_be_opened_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="Could not open"):
        profile_video(tmp_path / "missing.mkv")


def test_the_audit_script_writes_a_profile_and_the_jump_picture(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    audit_main(["--video", "half1.mkv", "--scan", "--step-sec", "1", "--out-dir", str(tmp_path)])
    record = json.loads((tmp_path / "half1.json").read_text())
    assert record["max_error"] == pytest.approx(13.8) and record["jumps"][0]["seconds"] == pytest.approx(13.8)
    assert (tmp_path / "half1_jump0.png").is_file()
    assert "FLAGGED" in capsys.readouterr().out
    audit_main(["--summary", "--out-dir", str(tmp_path)])
    assert "1 halves audited, 1 above 0.5s" in capsys.readouterr().out


def test_the_audit_script_needs_exactly_one_source(tmp_path):
    with pytest.raises(SystemExit):
        audit_main(["--out-dir", str(tmp_path)])
