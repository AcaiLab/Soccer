from make_league_manifest import build_rows


def folders():
    return {"england_epl_2018-2019": ["2018-08-11_b-c", "2018-08-10_a-b"],
            "england_epl_2017-2018": ["2017-08-12_c-d"]}


def test_games_are_numbered_from_start_in_league_then_date_order():
    rows, _ = build_rows(folders(), lambda lg, m: "test", start=66, excluded={})
    assert [(r["game_num"], r["match_id"]) for r in rows] == [
        ("66", "2017-08-12_c-d"), ("67", "2018-08-10_a-b"), ("68", "2018-08-11_b-c")]


def test_official_valid_becomes_project_val():
    rows, _ = build_rows(folders(), lambda lg, m: "valid", start=1, excluded={})
    assert {(r["official_split"], r["split"]) for r in rows} == {("valid", "val")}


def test_missing_annotation_is_listed_as_train_and_reported():
    split = {"2018-08-10_a-b": None}.get
    rows, missing = build_rows(folders(), lambda lg, m: split(m, "test"), start=1, excluded={})
    assert missing == ["england_epl_2018-2019/2018-08-10_a-b"]
    assert next(r for r in rows if r["match_id"] == "2018-08-10_a-b")["official_split"] == "train"


def test_excluded_matches_are_left_out_without_using_a_number():
    excluded = {("england_epl_2018-2019", "2018-08-10_a-b"): "duplicate"}
    rows, _ = build_rows(folders(), lambda lg, m: "train", start=1, excluded=excluded)
    assert [r["game_num"] for r in rows] == ["01", "02"]
    assert "2018-08-10_a-b" not in {r["match_id"] for r in rows}
