"""Annotation features may only describe the past: strictly earlier events, in true time order."""

from game_flow_features import features_for_match, match_events


def comment(half, ts, label):
    return {"half": half, "time_stamp": ts, "comments_type": label}


def rows_by_label(comments):
    return {r["label"]: r for r in features_for_match(match_events(comments))}


def test_first_half_stoppage_time_never_sees_the_second_half():
    # 46:30 of the first half happens before 00:30 of the second (match minute 45.5).
    rows = rows_by_label([comment(1, "46:30", "yellow_card"), comment(2, "00:30", "corner"),
                          comment(1, "40:00", "clearance")])
    assert rows["yellow_card"]["prev_1"] == "clearance"
    assert rows["yellow_card"]["recent_corners"] == 0
    assert rows["corner"]["prev_1"] == "yellow_card"  # the second half follows the whole first half


def test_events_at_the_same_second_are_not_history():
    rows = rows_by_label([comment(1, "10:00", "lead_to_corner"), comment(1, "10:00", "corner"),
                          comment(1, "09:00", "clearance")])
    assert rows["corner"]["prev_1"] == "clearance"
    assert rows["corner"]["recent_corners"] == 0  # the simultaneous lead_to_corner does not count


def test_repeated_comments_are_kept_once_and_halves_may_be_text():
    events = match_events([comment("1", "12:00", "substitution"), comment(1, "12:00", "substitution"),
                           comment(1, "11:00", "foul_with_no_card")])
    assert [e["label"] for e in events] == ["foul_with_no_card", "substitution"]
    rows = features_for_match(events)
    assert rows[1]["prev_1"] == "foul_with_no_card"  # never its own duplicate


def test_the_recent_window_is_five_minutes_of_the_same_half():
    rows = rows_by_label([comment(1, "40:00", "corner"), comment(1, "44:59", "shot_off_target"),
                          comment(1, "45:01", "goal"), comment(2, "01:00", "free_kick")])
    assert rows["goal"]["recent_corners"] == 0 and rows["goal"]["recent_shots"] == 1  # 40:00 is 5:01 earlier
    assert rows["free_kick"]["recent_shots"] == 0  # the first half is not in the second half's window
