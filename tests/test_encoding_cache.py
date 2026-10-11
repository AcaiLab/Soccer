import numpy as np
from PIL import Image

from soccer.encoding import encode_cached
from soccer.evaluation import split_by_game
from soccer.frames import scan_events
from soccer.manifest import Game


class FakeEncoder:
    embed_dim = 3

    def __init__(self):
        self.frames = 0

    def encode_batch(self, images):
        self.frames += len(images)
        return np.array([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in images])


def write_frames(root, game, half, label, tag, seconds, marker=True):
    d = root / f"game_{game}_half_{half}" / "frames" / label
    d.mkdir(parents=True, exist_ok=True)
    if marker:
        (root / f"game_{game}_half_{half}" / "extraction.json").write_text('{"settings": {"before_sec": 30}}')
    for s in seconds:
        Image.new("RGB", (4, 4), (s, s, s)).save(d / f"{label}_frame_{tag}_t00-{s:02d}.000.jpg")


def run(root, cache, made):
    def factory():
        made.append(FakeEncoder())
        return made[-1]
    return encode_cached(factory, scan_events(root), max_frames=60, batch_size=4,
                         cache_dir=cache, meta={"model": "fake", "max_frames": 60})


def test_cache_reuses_unchanged_halves_and_reencodes_changed_ones(tmp_path):
    frames, cache = tmp_path / "frames", tmp_path / "cache"
    write_frames(frames, "01", 1, "corner", "00-10.000", [8, 9, 10])
    write_frames(frames, "02", 1, "goal", "00-20.000", [19, 20])

    made = []
    padded, masks, keys, n, reused, encoded = run(frames, cache, made)
    assert (reused, encoded) == (0, 2) and made[0].frames == 5
    assert keys == [("01", "1", "corner", "00-10.000"), ("02", "1", "goal", "00-20.000")]

    made = []
    padded2, masks2, keys2, _, reused, encoded = run(frames, cache, made)
    assert (reused, encoded) == (2, 0) and made == []  # nothing encoded, encoder never built
    np.testing.assert_array_equal(padded, padded2)
    assert keys2 == keys

    write_frames(frames, "02", 1, "goal", "00-20.000", [21])  # one half gains a frame
    made = []
    _, masks3, _, _, reused, encoded = run(frames, cache, made)
    assert (reused, encoded) == (1, 1) and made[0].frames == 3
    assert masks3[1].sum() == 3


def test_split_ignores_games_outside_the_manifest():
    manifest = [Game("01", "m1", "l", "train", "train"), Game("02", "m2", "l", "train", "val"),
                Game("03", "m3", "l", "train", "test")]
    games = np.array(["01", "02", "03", "99"])  # 99 is in the features but not in the manifest
    labels = np.array(["a", "a", "a", "a"])
    split = split_by_game(games, labels, manifest)
    assert split.train.tolist() == [True, False, False, False]
    assert split.val.tolist() == [False, True, False, False]
    assert split.test.tolist() == [False, False, True, False]


def test_re_extracted_pictures_invalidate_the_cache_even_with_the_same_names(tmp_path):
    frames, cache = tmp_path / "frames", tmp_path / "cache"
    write_frames(frames, "01", 1, "corner", "00-10.000", [8, 9, 10])
    run(frames, cache, [])
    # Same file name, different picture: e.g. re-extracted by timestamp from a variable-rate video.
    path = next((frames / "game_01_half_1" / "frames" / "corner").glob("*t00-09.000.jpg"))
    Image.new("RGB", (4, 4), (200, 0, 0)).save(path)
    *_, reused, encoded = run(frames, cache, [])
    assert (reused, encoded) == (0, 1)


def test_identical_frames_keep_their_cache_after_re_extraction(tmp_path):
    frames, cache = tmp_path / "frames", tmp_path / "cache"
    write_frames(frames, "01", 1, "corner", "00-10.000", [8, 9, 10])
    run(frames, cache, [])
    (frames / "game_01_half_1" / "extraction.json").write_text('{"frame_selection": "timestamp"}')
    *_, reused, encoded = run(frames, cache, [])
    assert (reused, encoded) == (1, 0)


def test_halves_without_a_marker_are_not_encoded(tmp_path, capsys):
    frames = tmp_path / "frames"
    write_frames(frames, "01", 1, "corner", "00-10.000", [8, 9])
    write_frames(frames, "02", 1, "goal", "00-20.000", [19], marker=False)  # interrupted extraction
    assert {k[0] for k in scan_events(frames)} == {"01"}
    assert "without extraction.json" in capsys.readouterr().out
    assert {k[0] for k in scan_events(frames, require_marker=False)} == {"01", "02"}
