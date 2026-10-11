"""Refresh-token support: access tokens are fetched, renewed and never leaked. No network is used."""

import json

import pytest
import requests
from fake_http import FakeResponse, FakeSession, token_reply

from soccer.dropbox import (
    EXPIRY_MARGIN_SEC,
    TOKEN_URL,
    DropboxSharedLink,
    MissingScope,
    RefreshToken,
    StaticToken,
    TokenExpired,
)

LINK = "https://www.dropbox.com/scl/fo/abc"
REFRESH = "refresh-token-value"
EXPIRED = '{"error_summary": "expired_access_token/", "error": {".tag": "expired_access_token"}}'


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr("soccer.dropbox.time.sleep", lambda seconds: None)


def client_with(session, tokens):
    return DropboxSharedLink(tokens, LINK, "pw", session=session)


def bearer(call):
    return call[1]["headers"]["Authorization"]


# --- fetching and renewing access tokens ---------------------------------------------------

def test_access_token_is_fetched_once_and_reused():
    session = FakeSession(token_reply("A"))
    tokens = RefreshToken("key", REFRESH, session=session)
    assert tokens.get() == "A" and tokens.get() == "A"
    (url, sent), = session.calls
    assert url == TOKEN_URL
    assert sent["data"] == {"grant_type": "refresh_token", "refresh_token": REFRESH, "client_id": "key"}
    assert sent["auth"] is None


def test_access_token_is_replaced_shortly_before_it_expires():
    clock = Clock()
    session = FakeSession(token_reply("A", expires_in=3600), token_reply("B"))
    tokens = RefreshToken("key", REFRESH, session=session, clock=clock)
    assert tokens.get() == "A"
    clock.now += 3600 - EXPIRY_MARGIN_SEC - 1
    assert tokens.get() == "A"  # still comfortably valid
    clock.now += 1
    assert tokens.get() == "B"  # inside the margin: a new one was fetched
    assert len(session.calls) == 2


def test_app_secret_goes_in_basic_auth_instead_of_the_client_id():
    session = FakeSession(token_reply("A"))
    RefreshToken("key", REFRESH, app_secret="app-secret", session=session).get()
    sent = session.calls[0][1]
    assert sent["auth"] == ("key", "app-secret") and "client_id" not in sent["data"]


def test_revoked_refresh_token_is_reported_once_and_not_leaked():
    session = FakeSession(FakeResponse(400, {"error": "invalid_grant",
                                             "error_description": "refresh token is invalid or revoked"}))
    with pytest.raises(TokenExpired, match="invalid_grant") as info:
        RefreshToken("key", REFRESH, session=session).get()
    assert REFRESH not in str(info.value)
    assert len(session.calls) == 1  # asking again could not help


def test_temporary_failures_are_retried():
    session = FakeSession(FakeResponse(503, {}), requests.ConnectionError("down"), token_reply("A"))
    assert RefreshToken("key", REFRESH, session=session).get() == "A"
    assert len(session.calls) == 3


def test_gives_up_after_three_failed_attempts():
    session = FakeSession(*[FakeResponse(503, {}) for _ in range(3)])
    with pytest.raises(RuntimeError, match="HTTP 503"):
        RefreshToken("key", REFRESH, session=session).get()
    assert len(session.calls) == 3


def test_threads_refused_the_same_token_share_one_renewal():
    session = FakeSession(token_reply("A"), token_reply("B"))
    tokens = RefreshToken("key", REFRESH, session=session)
    assert tokens.get() == "A"
    assert tokens.renew("A") == "B"  # the first thread to notice
    assert tokens.renew("A") == "B"  # a second one holding the same stale token: no new request
    assert len(session.calls) == 2


# --- the client renews when Dropbox says the token has expired ----------------------------

def test_expired_token_is_renewed_and_the_request_repeated():
    expired = FakeResponse(401, text=EXPIRED)
    session = FakeSession(token_reply("A"), expired, token_reply("B"),
                          FakeResponse(200, {"entries": [{"name": "x"}]}))
    client = client_with(session, RefreshToken("key", REFRESH, session=session))
    assert client.list_folder("") == [{"name": "x"}]
    assert [url == TOKEN_URL for url, _ in session.calls] == [True, False, True, False]
    assert bearer(session.calls[1]) == "Bearer A" and bearer(session.calls[3]) == "Bearer B"
    assert expired.closed


def test_paging_goes_on_with_the_renewed_token():
    session = FakeSession(token_reply("A"),
                          FakeResponse(200, {"entries": [1], "has_more": True, "cursor": "c"}),
                          FakeResponse(401, text=EXPIRED), token_reply("B"),
                          FakeResponse(200, {"entries": [2]}))
    client = client_with(session, RefreshToken("key", REFRESH, session=session))
    assert client.list_folder("") == [1, 2]
    url, sent = session.calls[-1]
    assert url.endswith("/files/list_folder/continue") and sent["json"] == {"cursor": "c"}
    assert bearer(session.calls[-1]) == "Bearer B"


def test_a_streamed_download_is_renewed_too():
    ok = FakeResponse(200, {})
    session = FakeSession(token_reply("A"), FakeResponse(401, text=EXPIRED), token_reply("B"), ok)
    client = client_with(session, RefreshToken("key", REFRESH, session=session))
    assert client.open_file("/a/b.mkv") is ok
    retried = session.calls[3][1]
    assert retried["stream"] is True and retried["headers"]["Authorization"] == "Bearer B"
    assert json.loads(retried["headers"]["Dropbox-API-Arg"]) == {"url": LINK, "path": "/a/b.mkv",
                                                                 "link_password": "pw"}


def test_a_second_refusal_is_reported_instead_of_looping():
    session = FakeSession(token_reply("A"), FakeResponse(401, text=EXPIRED),
                          token_reply("B"), FakeResponse(401, text=EXPIRED))
    client = client_with(session, RefreshToken("key", REFRESH, session=session))
    with pytest.raises(TokenExpired, match="HTTP 401"):
        client.list_folder("")
    assert len(session.calls) == 4


def test_a_fixed_token_cannot_be_renewed():
    session = FakeSession(FakeResponse(401, text=EXPIRED))
    client = client_with(session, StaticToken("fixed"))
    with pytest.raises(TokenExpired, match="a refresh token avoids this"):
        client.list_folder("")
    assert len(session.calls) == 1


def test_missing_scope_is_not_mistaken_for_an_expired_token():
    body = ('{"error_summary": "missing_scope/", "error": {".tag": "missing_scope", '
            '"required_scope": "files.content.read"}}')
    session = FakeSession(token_reply("A"), FakeResponse(401, text=body))
    client = client_with(session, RefreshToken("key", REFRESH, session=session))
    with pytest.raises(MissingScope, match="files.content.read"):
        client.list_folder("")
    assert len(session.calls) == 2  # one token and one call: no pointless renewal


# --- where the credentials come from -------------------------------------------------------

def source_of():
    return DropboxSharedLink.from_env()._tokens


def test_refresh_credentials_from_the_environment(no_dropbox_env, monkeypatch):
    monkeypatch.setenv("DROPBOX_SHARED_URL", LINK)
    monkeypatch.setenv("DROPBOX_APP_KEY", "key")
    monkeypatch.setenv("DROPBOX_REFRESH_TOKEN", REFRESH)
    tokens = source_of()
    assert isinstance(tokens, RefreshToken) and tokens._app_key == "key"


def test_refresh_credentials_from_the_oauth_file(no_dropbox_env, monkeypatch, tmp_path):
    saved = tmp_path / "elsewhere" / "oauth.json"
    saved.parent.mkdir()
    saved.write_text(json.dumps({"app_key": "key", "refresh_token": REFRESH}))
    monkeypatch.setenv("DROPBOX_SHARED_URL", LINK)
    monkeypatch.setenv("DROPBOX_OAUTH_FILE", str(saved))
    assert isinstance(source_of(), RefreshToken)


def test_oauth_file_is_found_in_tmp_dbx_by_default(no_dropbox_env, monkeypatch, tmp_path):
    (tmp_path / "tmp_dbx").mkdir()
    (tmp_path / "tmp_dbx" / "oauth.json").write_text(json.dumps({"app_key": "key", "refresh_token": REFRESH}))
    monkeypatch.setenv("DROPBOX_SHARED_URL", LINK)
    assert isinstance(source_of(), RefreshToken)


def test_refresh_credentials_from_ssm(no_dropbox_env, monkeypatch):
    values = {"/soccer/dropbox/app_key": "key", "/soccer/dropbox/refresh_token": REFRESH,
              "/soccer/dropbox/shared_url": LINK}
    monkeypatch.setenv("SOCCER_SSM_PREFIX", "/soccer")
    monkeypatch.setattr("soccer.dropbox._ssm_parameter", values.get)
    assert isinstance(source_of(), RefreshToken)


def test_a_token_given_for_this_run_beats_the_refresh_token(no_dropbox_env, monkeypatch, tmp_path):
    monkeypatch.setenv("DROPBOX_SHARED_URL", LINK)
    monkeypatch.setenv("DROPBOX_APP_KEY", "key")
    monkeypatch.setenv("DROPBOX_REFRESH_TOKEN", REFRESH)
    chosen = tmp_path / "chosen"
    chosen.write_text("from-option\n")
    assert DropboxSharedLink.from_env(token_file=chosen)._tokens.get() == "from-option"
    monkeypatch.setenv("DROPBOX_TOKEN", "from-env")
    assert source_of().get() == "from-env"


def test_refresh_token_beats_a_stale_token_file(no_dropbox_env, monkeypatch, tmp_path):
    stale = tmp_path / "old_token"
    stale.write_text("stale\n")
    monkeypatch.setenv("DROPBOX_SHARED_URL", LINK)
    monkeypatch.setenv("DROPBOX_TOKEN_FILE", str(stale))
    assert source_of().get() == "stale"
    monkeypatch.setenv("DROPBOX_APP_KEY", "key")
    monkeypatch.setenv("DROPBOX_REFRESH_TOKEN", REFRESH)
    assert isinstance(source_of(), RefreshToken)


def test_refresh_token_without_an_app_key_is_explained(no_dropbox_env, monkeypatch):
    monkeypatch.setenv("DROPBOX_SHARED_URL", LINK)
    monkeypatch.setenv("DROPBOX_REFRESH_TOKEN", REFRESH)
    with pytest.raises(RuntimeError, match="DROPBOX_APP_KEY"):
        source_of()


def test_a_token_file_named_on_the_command_line_must_exist(no_dropbox_env, monkeypatch, tmp_path):
    monkeypatch.setenv("DROPBOX_SHARED_URL", LINK)
    with pytest.raises(RuntimeError, match="not found"):
        DropboxSharedLink.from_env(token_file=tmp_path / "nope")


def test_a_broken_oauth_file_is_explained(no_dropbox_env, monkeypatch, tmp_path):
    broken = tmp_path / "oauth.json"
    broken.write_text("{not json")
    monkeypatch.setenv("DROPBOX_SHARED_URL", LINK)
    monkeypatch.setenv("DROPBOX_OAUTH_FILE", str(broken))
    with pytest.raises(RuntimeError, match="not valid JSON"):
        source_of()


# --- transient failures are retried ----------------------------------------------------------

def test_dropped_connections_and_server_errors_are_retried(monkeypatch):
    waits = []
    monkeypatch.setattr("soccer.dropbox.time.sleep", waits.append)
    session = FakeSession(requests.ConnectionError("reset"), FakeResponse(503, {}),
                          FakeResponse(429, {}, headers={"Retry-After": "42"}),
                          FakeResponse(200, {"entries": [{"name": "x"}]}))
    client = client_with(session, StaticToken("fixed"))
    assert client.list_folder("") == [{"name": "x"}]
    assert len(session.calls) == 4
    assert waits == [5.0, 10.0, 42.0]  # backoff, then what Dropbox asked for


def test_retries_give_up_and_report_the_error(monkeypatch):
    session = FakeSession(*[FakeResponse(503, {}, text="busy") for _ in range(5)])
    with pytest.raises(RuntimeError, match="HTTP 503"):
        client_with(session, StaticToken("fixed")).list_folder("")
    assert len(session.calls) == 5


def test_errors_that_will_not_go_away_are_not_retried():
    session = FakeSession(FakeResponse(409, text="path/not_found"))
    with pytest.raises(RuntimeError, match="HTTP 409"):
        client_with(session, StaticToken("fixed")).list_folder("/nope")
    assert len(session.calls) == 1
