import io
import zipfile

import boto3
import pytest
from moto import mock_aws

import soccer.transfer as transfer
from soccer.annotations import find_annotation_file, install_split_zip
from soccer.config import load_config, s3_uri
from soccer.manifest import Game
from soccer.pipeline import fetch_videos_from_s3
from soccer.storage import upload_dir

GAME = Game("01", "2021-08-22_arsenal-fc-chelsea-fc", "england_epl_2021-2022", "train", "train")


class FakeStream:
    def __init__(self, data):
        self.raw = io.BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDropbox:
    """Lists two halves; can claim a larger size than it serves, like a truncated download."""

    def __init__(self, files, claimed_extra=0):
        self.files, self.extra, self.opened = files, claimed_extra, []

    def list_folder(self, path):
        return [{".tag": "file", "name": n, "size": len(d) + self.extra} for n, d in self.files.items()]

    def open_file(self, path):
        self.opened.append(path)
        return FakeStream(self.files[path.rsplit("/", 1)[1]])


@pytest.fixture
def s3(monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test",
                 "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(k, v)
    with mock_aws():
        client = boto3.client("s3")
        client.create_bucket(Bucket="soccer-test")
        yield client


def files(size_1=1000, size_2=2000):
    return {"m_1.mkv": b"a" * size_1, "m_2.mkv": b"b" * size_2}


def test_copy_game_streams_both_halves_and_resumes(s3):
    dbx = FakeDropbox(files())
    out = transfer.copy_game(dbx, GAME, "s3://soccer-test/videos", s3)
    assert [o for _, _, o in out] == ["copied", "copied"]
    key = f"videos/{GAME.league}/{GAME.match_id}/m_1.mkv"
    assert s3.head_object(Bucket="soccer-test", Key=key)["ContentLength"] == 1000

    dbx.opened.clear()
    again = transfer.copy_game(dbx, GAME, "s3://soccer-test/videos", s3)
    assert [o for _, _, o in again] == ["already in S3", "already in S3"] and dbx.opened == []


def test_multipart_upload_path(s3, monkeypatch):
    monkeypatch.setattr(transfer, "CHUNK", 5 * 1024 * 1024)  # S3's minimum part size
    big = files(size_1=11 * 1024 * 1024)
    transfer.copy_game(FakeDropbox(big), GAME, "s3://soccer-test/videos", s3)
    key = f"videos/{GAME.league}/{GAME.match_id}/m_1.mkv"
    assert s3.head_object(Bucket="soccer-test", Key=key)["ContentLength"] == 11 * 1024 * 1024


def test_truncated_copy_is_deleted_and_fails(s3):
    with pytest.raises(RuntimeError, match="deleted"):
        transfer.copy_game(FakeDropbox(files(), claimed_extra=10), GAME, "s3://soccer-test/videos", s3)
    keys = s3.list_objects_v2(Bucket="soccer-test").get("Contents", [])
    assert keys == []


def test_videos_come_back_from_s3_for_extraction(s3, tmp_path):
    transfer.copy_game(FakeDropbox(files()), GAME, "s3://soccer-test/videos", s3)
    out = fetch_videos_from_s3("s3://soccer-test/videos", GAME, tmp_path)
    assert sorted(p.name for p in out.iterdir()) == ["m_1.mkv", "m_2.mkv"]
    assert (out / "m_2.mkv").stat().st_size == 2000


def test_upload_dir_keeps_relative_paths(s3, tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.csv").write_text("1")
    assert upload_dir(tmp_path, "s3://soccer-test/results") == 1
    assert s3.head_object(Bucket="soccer-test", Key="results/a/b/x.csv")["ContentLength"] == 1


def test_s3_uri_requires_a_root(monkeypatch):
    monkeypatch.delenv("SOCCER_S3_ROOT", raising=False)
    cfg = load_config()
    with pytest.raises(RuntimeError):
        s3_uri(cfg, "videos")
    monkeypatch.setenv("SOCCER_S3_ROOT", "s3://bucket/prefix/")
    assert s3_uri(cfg, "videos") == "s3://bucket/prefix/videos"
    monkeypatch.setenv("SOCCER_S3_ROOT", "my-bucket")
    with pytest.raises(ValueError):
        s3_uri(cfg, "videos")


def test_install_split_zip_drops_macos_metadata(tmp_path):
    """The real valid/test zips carry '._<name>.json' stubs and .DS_Store files."""
    zip_path = tmp_path / "test.zip"
    match = "test/england_epl_2021-2022/2021-08-22_x"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{match}/2021-08-22_A_1-1_B_abc.json", '{"comments": []}')
        z.writestr(f"{match}/._2021-08-22_A_1-1_B_abc.json", b"\x00\x05\x16\x07 macOS stub")
        z.writestr("test/england_epl_2021-2022/.DS_Store", b"\x00")
        z.writestr(f"__MACOSX/{match}/._2021-08-22_A_1-1_B_abc.json", b"\x00")
    assert install_split_zip(zip_path, "test", tmp_path / "json") == 1
    folder = tmp_path / "json" / "test" / "england_epl_2021-2022"
    assert sorted(p.name for p in folder.rglob("*")) == ["2021-08-22_A_1-1_B_abc.json", "2021-08-22_x"]


def test_annotation_lookup_ignores_hidden_files(tmp_path):
    folder = tmp_path / "train" / "england_epl_2021-2022" / "m1"
    folder.mkdir(parents=True)
    (folder / "._m1.json").write_bytes(b"\x00 stub")  # sorts first
    (folder / "m1.json").write_text('{"comments": []}')
    path, split = find_annotation_file(tmp_path, "england_epl_2021-2022", "m1")
    assert path.name == "m1.json" and split == "train"


@pytest.mark.parametrize("top", ["", "train/", "SoccerReplay/train/"])
def test_install_split_zip_any_layout(tmp_path, top):
    zip_path = tmp_path / "train.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(f"{top}england_epl_2021-2022/2021-08-22_x/2021-08-22_A_0-2_B_abc.json", '{"comments": []}')
        z.writestr(f"{top}spain_laliga_2021-2022/2021-09-01_y/2021-09-01_C_1-1_D_def.json", '{"comments": []}')
    n = install_split_zip(zip_path, "train", tmp_path / "json")
    assert n == 2
    assert (tmp_path / "json" / "train" / "england_epl_2021-2022" / "2021-08-22_x").is_dir()
