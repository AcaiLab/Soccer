import json

import pytest

from soccer.annotations import (
    detect_half,
    events_for_half,
    find_annotation_file,
    match_minute,
    parse_timestamp,
    sanitize_label,
    seconds_to_tag,
    tag_to_seconds,
)


@pytest.mark.parametrize("ts, seconds", [
    ("00:01", 1),
    ("09:00", 540),
    ("45+2:30", 2850),
    (" 46:53 ", 2813),
])
def test_parse_timestamp(ts, seconds):
    assert parse_timestamp(ts) == seconds


@pytest.mark.parametrize("ts", ["", "9", "9:5", "abc", "1:2:3"])
def test_parse_timestamp_rejects_bad_input(ts):
    with pytest.raises(ValueError):
        parse_timestamp(ts)


def test_tag_round_trip():
    assert seconds_to_tag(540) == "09-00.000"
    assert seconds_to_tag(2850.5) == "47-30.500"
    assert tag_to_seconds(seconds_to_tag(2850.5)) == 2850.5


@pytest.mark.parametrize("raw, clean", [
    ("lead to corner", "lead_to_corner"),
    ("saved by goal-keeper", "saved_by_goal-keeper"),
    ("Yellow Card", "yellow_card"),
    ("", "unknown"),
    (None, "unknown"),
])
def test_sanitize_label(raw, clean):
    assert sanitize_label(raw) == clean


def test_match_minute_offsets_second_half():
    assert match_minute(90, 1) == 1.5
    assert match_minute(90, 2) == 46.5


def comments():
    # Second-half timestamps restart at 0:00, so both halves have a "07:09".
    return [
        {"half": 0, "time_stamp": "", "comments_type": "statistics and summary"},
        {"half": 1, "time_stamp": "07:09", "comments_type": "corner"},
        {"half": 1, "time_stamp": "01:06", "comments_type": "foul with no card"},
        {"half": 1, "time_stamp": "01:06", "comments_type": "foul with no card"},  # duplicate
        {"half": 2, "time_stamp": "07:09", "comments_type": "corner"},
        {"half": 2, "time_stamp": "20:00", "comments_type": "goal"},
        {"half": 2, "time_stamp": "bad", "comments_type": "goal"},
    ]


def test_events_for_half_keeps_only_that_half():
    first = events_for_half(comments(), 1)
    second = events_for_half(comments(), 2)
    assert [(e.timestamp, e.label) for e in first] == [("01:06", "foul_with_no_card"), ("07:09", "corner")]
    assert [(e.timestamp, e.label) for e in second] == [("07:09", "corner"), ("20:00", "goal")]
    assert all(e.half == 1 for e in first) and all(e.half == 2 for e in second)


def test_events_for_half_ignores_pre_match():
    assert events_for_half(comments(), 0) == []


@pytest.mark.parametrize("name, half", [("x/game_1.mkv", 1), ("game-2021_2.mp4", 2), ("C:\\v\\a_b_1.avi", 1)])
def test_detect_half(name, half):
    assert detect_half(name) == half


def test_detect_half_refuses_unknown_names():
    with pytest.raises(ValueError):
        detect_half("full_match.mkv")


def test_find_annotation_file_reports_split(tmp_path):
    folder = tmp_path / "valid" / "england_epl_2021-2022" / "m1"
    folder.mkdir(parents=True)
    (folder / "m1.json").write_text(json.dumps({"comments": []}))
    path, split = find_annotation_file(tmp_path, "england_epl_2021-2022", "m1")
    assert path.name == "m1.json" and split == "valid"
    assert find_annotation_file(tmp_path, "england_epl_2021-2022", "missing") is None


def test_events_are_deduplicated_by_instant_not_text():
    comments = [
        {"half": 1, "time_stamp": "45+2:30", "comments_type": "corner"},
        {"half": 1, "time_stamp": "47:30", "comments_type": "corner"},   # same second
        {"half": 1, "time_stamp": "47:30", "comments_type": "lead to corner"},  # same second, other label
    ]
    events = events_for_half(comments, 1)
    assert [(e.seconds, e.label) for e in events] == [(2850, "corner"), (2850, "lead_to_corner")]


def test_half_stored_as_text_is_understood():
    comments = [{"half": "2", "time_stamp": "10:00", "comments_type": "goal"},
                {"half": None, "time_stamp": "11:00", "comments_type": "goal"}]
    assert [e.seconds for e in events_for_half(comments, 2)] == [600]
