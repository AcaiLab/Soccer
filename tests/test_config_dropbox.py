import pytest

from soccer.config import data_path, dropped_labels, load_config
from soccer.dropbox import DropboxSharedLink, TokenExpired


def test_data_root_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("SOCCER_DATA_ROOT", str(tmp_path))
    cfg = load_config()
    assert data_path(cfg, "frames") == tmp_path / "results_by_label_epl_2021_2022"


def test_dropped_labels_cover_junk_and_graphics():
    labels = dropped_labels(load_config())
    assert {"unknown", "wcl-icon-settings-info-rounded", "statistics_and_summary"} <= labels
    assert "corner" not in labels


@pytest.fixture
def no_dropbox_env(monkeypatch):
    for name in ("DROPBOX_TOKEN", "DROPBOX_TOKEN_FILE", "DROPBOX_SHARED_URL",
                 "DROPBOX_LINK_PASSWORD", "SOCCER_SSM_PREFIX"):
        monkeypatch.delenv(name, raising=False)


def test_missing_credentials_fail_clearly(no_dropbox_env):
    with pytest.raises(RuntimeError, match="DROPBOX_SHARED_URL"):
        DropboxSharedLink.from_env()


def test_token_can_come_from_a_file(no_dropbox_env, monkeypatch, tmp_path):
    token = tmp_path / "token.txt"
    token.write_text("abc\n")
    monkeypatch.setenv("DROPBOX_SHARED_URL", "https://example.invalid/link")
    client = DropboxSharedLink.from_env(token_file=token)
    assert client._token == "abc"


class FakeResponse:
    def __init__(self, status, text=""):
        self.status_code = status
        self._text = text

    @property
    def text(self):
        if self.status_code == 200:
            raise AssertionError("a successful streamed body must not be read")
        return self._text


def test_check_never_reads_successful_bodies():
    DropboxSharedLink("t", "u")._check(FakeResponse(200))


def test_check_detects_expired_tokens():
    with pytest.raises(TokenExpired):
        DropboxSharedLink("t", "u")._check(FakeResponse(401, '{"error": "expired_access_token"}'))
    with pytest.raises(RuntimeError, match="HTTP 409"):
        DropboxSharedLink("t", "u")._check(FakeResponse(409, "path/not_found"))


def test_token_file_env_is_used_when_no_option_given(no_dropbox_env, monkeypatch, tmp_path):
    token = tmp_path / "server_token"
    token.write_text("from-env\n")
    monkeypatch.setenv("DROPBOX_SHARED_URL", "https://example.invalid/link")
    monkeypatch.setenv("DROPBOX_TOKEN_FILE", str(token))
    assert DropboxSharedLink.from_env()._token == "from-env"
    assert DropboxSharedLink.from_env(token_file=None)._token == "from-env"
