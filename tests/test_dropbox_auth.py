"""The one-time authorisation script: PKCE link, code exchange, and keeping secrets out of the output."""

import json
import os
import stat
import string
import urllib.parse

import pytest
from dropbox_auth import authorize_url, challenge_for, main, new_verifier
from fake_http import FakeResponse, FakeSession, token_reply

from soccer.dropbox import TOKEN_URL

REFRESH = "REFRESH-VALUE"
ACCESS = "ACCESS-VALUE"


def query_of(url):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


def issued_tokens():
    return FakeResponse(200, {"access_token": ACCESS, "refresh_token": REFRESH, "expires_in": 14400,
                              "scope": "files.content.read sharing.read"})


def test_challenge_matches_the_example_in_rfc_7636():
    assert challenge_for("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == \
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_verifiers_are_random_and_within_the_allowed_length():
    a, b = new_verifier(), new_verifier()
    assert a != b and 43 <= len(a) <= 128
    assert set(a) <= set(string.ascii_letters + string.digits + "-_")


def test_link_asks_for_an_offline_token_with_pkce():
    url = authorize_url("KEY", "CHALLENGE")
    parts = urllib.parse.urlsplit(url)
    assert (parts.netloc, parts.path) == ("www.dropbox.com", "/oauth2/authorize")
    assert query_of(url) == {"client_id": "KEY", "response_type": "code", "code_challenge": "CHALLENGE",
                             "code_challenge_method": "S256", "token_access_type": "offline"}
    assert query_of(authorize_url("KEY", "C", ["files.content.read", "sharing.read"]))["scope"] == \
        "files.content.read sharing.read"


def test_start_then_finish_saves_a_refresh_token_and_prints_no_secret(tmp_path, capsys):
    oauth = tmp_path / "oauth.json"
    main(["start", "--app-key", "KEY", "--oauth-file", str(oauth)])
    started = capsys.readouterr().out
    state = json.loads((tmp_path / "pkce_state.json").read_text())
    assert state["app_key"] == "KEY"
    assert challenge_for(state["verifier"]) in started  # the link carries the challenge ...
    assert state["verifier"] not in started             # ... and never the verifier

    session = FakeSession(issued_tokens())
    main(["finish", "--code", " CODE-FROM-BROWSER ", "--oauth-file", str(oauth)], session=session)
    finished = capsys.readouterr().out

    (url, sent), = session.calls
    assert url == TOKEN_URL
    assert sent["data"] == {"code": "CODE-FROM-BROWSER", "grant_type": "authorization_code",
                            "code_verifier": state["verifier"], "client_id": "KEY"}
    assert json.loads(oauth.read_text()) == {"app_key": "KEY", "refresh_token": REFRESH}
    assert not (tmp_path / "pkce_state.json").exists()
    assert REFRESH not in finished and ACCESS not in finished
    assert "files.content.read" in finished


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX file modes")
def test_saved_files_are_readable_by_their_owner_only(tmp_path):
    oauth = tmp_path / "oauth.json"
    main(["start", "--app-key", "KEY", "--oauth-file", str(oauth)])
    assert stat.S_IMODE((tmp_path / "pkce_state.json").stat().st_mode) == 0o600
    main(["finish", "--code", "c", "--oauth-file", str(oauth)], session=FakeSession(issued_tokens()))
    assert stat.S_IMODE(oauth.stat().st_mode) == 0o600


def test_the_code_can_be_read_from_a_file(tmp_path):
    oauth = tmp_path / "oauth.json"
    code = tmp_path / "code.txt"
    code.write_text("CODE-FROM-FILE\n")
    main(["start", "--app-key", "KEY", "--oauth-file", str(oauth)])
    session = FakeSession(issued_tokens())
    main(["finish", "--code-file", str(code), "--oauth-file", str(oauth)], session=session)
    assert session.calls[0][1]["data"]["code"] == "CODE-FROM-FILE"


def test_a_rejected_code_explains_how_to_retry(tmp_path):
    oauth = tmp_path / "oauth.json"
    main(["start", "--app-key", "KEY", "--oauth-file", str(oauth)])
    session = FakeSession(FakeResponse(400, {"error": "invalid_grant", "error_description": "code has expired"}))
    with pytest.raises(SystemExit, match="invalid_grant.*run 'start' again"):
        main(["finish", "--code", "old", "--oauth-file", str(oauth)], session=session)
    assert not oauth.exists()


def test_a_reply_without_a_refresh_token_is_refused(tmp_path):
    oauth = tmp_path / "oauth.json"
    main(["start", "--app-key", "KEY", "--oauth-file", str(oauth)])
    session = FakeSession(FakeResponse(200, {"access_token": ACCESS, "expires_in": 14400}))
    with pytest.raises(SystemExit, match="no refresh token"):
        main(["finish", "--code", "c", "--oauth-file", str(oauth)], session=session)
    assert not oauth.exists()


def test_finish_before_start_says_what_to_do(tmp_path):
    with pytest.raises(SystemExit, match="run 'start' first"):
        main(["finish", "--code", "c", "--oauth-file", str(tmp_path / "oauth.json")])


def test_start_needs_an_app_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DROPBOX_APP_KEY", raising=False)
    with pytest.raises(SystemExit, match="app key"):
        main(["start", "--oauth-file", str(tmp_path / "oauth.json")])


def test_check_fetches_a_token_and_lists_the_shared_link(tmp_path, monkeypatch, capsys):
    oauth = tmp_path / "oauth.json"
    oauth.write_text(json.dumps({"app_key": "KEY", "refresh_token": REFRESH}))
    monkeypatch.setenv("DROPBOX_SHARED_URL", "https://www.dropbox.com/scl/fo/abc")
    monkeypatch.delenv("DROPBOX_LINK_PASSWORD", raising=False)
    session = FakeSession(token_reply(ACCESS), FakeResponse(200, {"entries": [{}, {}, {}]}))
    main(["check", "--oauth-file", str(oauth)], session=session)
    out = capsys.readouterr().out
    assert "3 entries" in out and REFRESH not in out and ACCESS not in out


def test_check_reports_a_revoked_refresh_token(tmp_path):
    oauth = tmp_path / "oauth.json"
    oauth.write_text(json.dumps({"app_key": "KEY", "refresh_token": REFRESH}))
    session = FakeSession(FakeResponse(400, {"error": "invalid_grant"}))
    with pytest.raises(SystemExit, match="invalid_grant"):
        main(["check", "--oauth-file", str(oauth)], session=session)


def test_check_without_saved_credentials_says_what_to_do(tmp_path):
    with pytest.raises(SystemExit, match="run 'start' and 'finish' first"):
        main(["check", "--oauth-file", str(tmp_path / "missing.json")])
