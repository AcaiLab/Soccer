import pytest

from soccer.config import load_config, repo_path
from soccer.manifest import assert_disjoint, game_nums, load_manifest, normalize_game_id


@pytest.fixture
def games():
    cfg = load_config()
    return load_manifest(repo_path(cfg, "manifest"), repo_path(cfg, "excluded_matches"))


def test_committed_manifest_is_valid(games):
    assert len(games) == 30
    assert len(game_nums(games, "train")) == 20
    assert game_nums(games, "val") == ["21", "22", "23", "24"]
    assert game_nums(games, "test") == ["25", "26", "27", "28", "29", "30"]


def test_no_match_is_in_two_splits(games):
    by_split = {s: {g.match_id for g in games if g.split == s} for s in ("train", "val", "test")}
    assert_disjoint(**by_split)
    assert sum(len(v) for v in by_split.values()) == len(games)


@pytest.mark.parametrize("raw, norm", [("7", "07"), ("07", "07"), ("0031", "31"), (31, "31")])
def test_normalize_game_id(raw, norm):
    assert normalize_game_id(raw) == norm


def write_manifest(tmp_path, rows):
    p = tmp_path / "games.csv"
    p.write_text("game_num,match_id,league,official_split,split\n" + "\n".join(rows) + "\n")
    return p


def test_duplicate_match_is_rejected(tmp_path):
    p = write_manifest(tmp_path, ["01,m1,l,train,train", "02,m1,l,train,test"])
    with pytest.raises(ValueError, match="appears twice"):
        load_manifest(p)


def test_unknown_split_is_rejected(tmp_path):
    p = write_manifest(tmp_path, ["01,m1,l,train,holdout"])
    with pytest.raises(ValueError, match="unknown split"):
        load_manifest(p)


def test_excluded_match_is_rejected(tmp_path):
    p = write_manifest(tmp_path, ["01,m1,l,train,train"])
    ex = tmp_path / "excluded.csv"
    ex.write_text("league,match_id,reason\nl,m1,duplicate\n")
    with pytest.raises(ValueError, match="excluded"):
        load_manifest(p, ex)


def test_assert_disjoint_names_the_overlap():
    with pytest.raises(ValueError, match="'07' is in both train and test"):
        assert_disjoint(train=["01", "07"], test=["07"])
