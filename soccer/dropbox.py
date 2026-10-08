"""Download SoccerReplay-1988 videos from the password-protected Dropbox link.

Credentials never live in code. They come from, in order:

  1. environment variables
       DROPBOX_SHARED_URL, DROPBOX_LINK_PASSWORD   the shared link
     and one way to get access tokens, the first that is set:
       DROPBOX_TOKEN, or the --token-file option   one fixed access token; Dropbox ends it
                                                   after about four hours
       DROPBOX_APP_KEY + DROPBOX_REFRESH_TOKEN     a refresh token, which does not expire:
         or the JSON file DROPBOX_OAUTH_FILE       new access tokens are fetched as needed
         (default tmp_dbx/oauth.json)              (scripts/dropbox_auth.py writes the file)
       DROPBOX_TOKEN_FILE, else tmp_dbx/token.txt  one fixed access token, kept in a file
  2. AWS SSM Parameter Store, if SOCCER_SSM_PREFIX is set:
       <prefix>/dropbox/{shared_url,link_password,app_key,refresh_token,token}  (SecureString)
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time

import requests

API = "https://api.dropboxapi.com/2"
CONTENT = "https://content.dropboxapi.com/2"
TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"
DEFAULT_TOKEN_FILE = "tmp_dbx/token.txt"
DEFAULT_OAUTH_FILE = "tmp_dbx/oauth.json"
EXPIRY_MARGIN_SEC = 300  # fetch a new access token this long before the old one runs out


class TokenExpired(RuntimeError):
    """The access token was rejected and could not be renewed; get new credentials and resume."""


class MissingScope(TokenExpired):
    """The app was not granted a permission the call needs; a new token would not help."""


class StaticToken:
    """One fixed access token. It cannot be renewed, so it ends when Dropbox expires it."""

    renewable = False

    def __init__(self, token: str) -> None:
        self._token = token

    def get(self) -> str:
        return self._token


class RefreshToken:
    """Access tokens fetched with a refresh token, which stays valid until it is revoked.

    A token is fetched on first use and again shortly before it expires, so a job can run
    for days. Several threads may share one instance.
    """

    renewable = True

    def __init__(self, app_key: str, refresh_token: str, app_secret: str | None = None,
                 session: requests.Session | None = None, clock=time.time) -> None:
        self._app_key = app_key
        self._refresh_token = refresh_token
        self._app_secret = app_secret
        self._session = session or requests.Session()
        self._clock = clock
        self._access: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if self._access is None or self._clock() >= self._expires_at - EXPIRY_MARGIN_SEC:
                self._fetch()
            return self._access

    def renew(self, rejected: str) -> str:
        """A new access token after Dropbox refused `rejected`, unless another thread already got one."""
        with self._lock:
            if self._access in (None, rejected):
                self._fetch()
            return self._access

    def _fetch(self, attempts: int = 3) -> None:
        data = {"grant_type": "refresh_token", "refresh_token": self._refresh_token}
        auth = None
        if self._app_secret:
            auth = (self._app_key, self._app_secret)
        else:
            data["client_id"] = self._app_key  # an app authorised with PKCE needs no secret
        for attempt in range(1, attempts + 1):
            try:
                r = self._session.post(TOKEN_URL, data=data, auth=auth, timeout=30)
            except requests.RequestException as e:
                problem = type(e).__name__
            else:
                if r.status_code == 200:
                    body = r.json()
                    self._access = body["access_token"]
                    self._expires_at = self._clock() + float(body.get("expires_in", 4 * 3600))
                    return
                problem = f"HTTP {r.status_code} {oauth_error(r)}".strip()
                if r.status_code in (400, 401):  # revoked or mismatched credentials: waiting cannot help
                    raise TokenExpired(f"Dropbox would not issue an access token ({problem}). The refresh "
                                       "token was revoked or does not belong to this app key; run "
                                       "scripts/dropbox_auth.py again")
            if attempt == attempts:
                raise RuntimeError(f"Could not get a new Dropbox access token ({problem})")
            time.sleep(5 * attempt)


def oauth_error(r: requests.Response) -> str:
    """The error code and description of a token-endpoint reply, and nothing else from it."""
    try:
        body = r.json()
        return f"{body.get('error', '')}: {body.get('error_description', '')}".strip(": ")[:200]
    except (ValueError, AttributeError):
        return ""


def _required_scope(body: str) -> str:
    try:
        return json.loads(body)["error"]["required_scope"]
    except (ValueError, KeyError, TypeError):
        return "scope not named"


def _ssm_parameter(name: str) -> str | None:
    import boto3  # optional dependency, only needed when SSM is used

    try:
        resp = boto3.client("ssm").get_parameter(Name=name, WithDecryption=True)
    except Exception:
        return None
    return resp["Parameter"]["Value"]


def _setting(env_name: str, ssm_suffix: str) -> str | None:
    value = os.environ.get(env_name)
    if value:
        return value
    prefix = os.environ.get("SOCCER_SSM_PREFIX")
    return _ssm_parameter(f"{prefix.rstrip('/')}/dropbox/{ssm_suffix}") if prefix else None


def _read_token_file(path: str | os.PathLike) -> str | None:
    p = pathlib.Path(path).expanduser()
    return (p.read_text(encoding="utf-8").strip() or None) if p.is_file() else None


def _saved_oauth() -> dict:
    """app_key and refresh_token (and app_secret, if any) as saved by scripts/dropbox_auth.py."""
    path = pathlib.Path(os.environ.get("DROPBOX_OAUTH_FILE") or DEFAULT_OAUTH_FILE).expanduser()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise RuntimeError(f"{path} is not valid JSON; run scripts/dropbox_auth.py again") from e
    return data if isinstance(data, dict) else {}


def _token_source(token_file: str | os.PathLike | None = None) -> StaticToken | RefreshToken | None:
    """Where access tokens come from; the order is in the module docstring."""
    if token_file and not pathlib.Path(token_file).expanduser().is_file():
        raise RuntimeError(f"Token file not found: {token_file}")
    fixed = os.environ.get("DROPBOX_TOKEN") or (_read_token_file(token_file) if token_file else None)
    if fixed:
        return StaticToken(fixed)

    saved = _saved_oauth()
    refresh_token = _setting("DROPBOX_REFRESH_TOKEN", "refresh_token") or saved.get("refresh_token")
    if refresh_token:
        app_key = _setting("DROPBOX_APP_KEY", "app_key") or saved.get("app_key")
        if not app_key:
            raise RuntimeError("A Dropbox refresh token is set but DROPBOX_APP_KEY is not (see README)")
        app_secret = _setting("DROPBOX_APP_SECRET", "app_secret") or saved.get("app_secret")
        return RefreshToken(app_key, refresh_token, app_secret)

    fixed = (_setting("DROPBOX_TOKEN", "token")
             or _read_token_file(os.environ.get("DROPBOX_TOKEN_FILE") or DEFAULT_TOKEN_FILE))
    return StaticToken(fixed) if fixed else None


class DropboxSharedLink:
    def __init__(self, token: str | StaticToken | RefreshToken, url: str, password: str | None = None,
                 session: requests.Session | None = None) -> None:
        self._tokens = StaticToken(token) if isinstance(token, str) else token
        self._link = {"url": url, **({"password": password} if password else {})}
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls, token_file: str | os.PathLike | None = None) -> "DropboxSharedLink":
        tokens = _token_source(token_file)
        url = _setting("DROPBOX_SHARED_URL", "shared_url")
        password = _setting("DROPBOX_LINK_PASSWORD", "link_password")
        missing = [n for n, v in [("a Dropbox token (DROPBOX_TOKEN, a token file or a refresh token)", tokens),
                                  ("DROPBOX_SHARED_URL", url)] if not v]
        if missing:
            raise RuntimeError("Missing Dropbox settings: " + ", ".join(missing) + " (see README)")
        return cls(tokens, url, password)

    def _post(self, url: str, headers: dict | None = None, **kwargs) -> requests.Response:
        """POST with the current access token.

        If Dropbox says the token has expired and it can be renewed, renew it and send the
        request once more; a second refusal is reported by _check.
        """
        token = self._tokens.get()
        r = self._session.post(url, headers={**(headers or {}), "Authorization": f"Bearer {token}"}, **kwargs)
        if r.status_code == 401 and self._tokens.renewable and "missing_scope" not in r.text[:500]:
            r.close()
            token = self._tokens.renew(token)
            r = self._session.post(url, headers={**(headers or {}), "Authorization": f"Bearer {token}"},
                                   **kwargs)
        return r

    def _check(self, r: requests.Response) -> None:
        if r.status_code == 200:
            return  # never touch r.text here: it would read a streamed video into memory
        body = r.text[:500]
        if "missing_scope" in body:
            raise MissingScope(f"The Dropbox app was not granted a permission this call needs "
                               f"({_required_scope(body)}): tick it on the app's Permissions tab, press "
                               f"Submit, then authorise again (scripts/dropbox_auth.py)")
        if r.status_code == 401 or "access_token" in body:
            hint = "" if self._tokens.renewable else "; a refresh token avoids this (see README)"
            raise TokenExpired(f"Dropbox rejected the access token (HTTP {r.status_code}){hint}")
        raise RuntimeError(f"Dropbox HTTP {r.status_code}: {body[:300]}")

    def list_folder(self, path: str) -> list[dict]:
        """Entries of a folder inside the shared link, following pagination."""
        r = self._post(f"{API}/files/list_folder", timeout=60, json={"path": path, "shared_link": self._link})
        self._check(r)
        data = r.json()
        entries = data["entries"]
        while data.get("has_more"):
            r = self._post(f"{API}/files/list_folder/continue", timeout=60, json={"cursor": data["cursor"]})
            self._check(r)
            data = r.json()
            entries += data["entries"]
        return entries

    def open_file(self, remote_path: str) -> requests.Response:
        """Streaming response for a file inside the shared link; use as a context manager."""
        arg = {"url": self._link["url"], "path": remote_path}
        if "password" in self._link:
            arg["link_password"] = self._link["password"]
        r = self._post(f"{CONTENT}/sharing/get_shared_link_file", stream=True, timeout=120,
                       headers={"Dropbox-API-Arg": json.dumps(arg)})
        try:
            self._check(r)
        except Exception:
            r.close()
            raise
        return r

    def download(self, remote_path: str, local_path: str | os.PathLike,
                 expected_size: int | None = None, retries: int = 3) -> int:
        """Download to local_path via a .part file; returns bytes written."""
        local_path = pathlib.Path(local_path)
        if expected_size is not None and local_path.is_file() and local_path.stat().st_size == expected_size:
            return 0
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_suffix(local_path.suffix + ".part")
        for attempt in range(1, retries + 1):
            try:
                with self.open_file(remote_path) as r:
                    written = 0
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                            written += len(chunk)
                if expected_size is not None and written != expected_size:
                    raise RuntimeError(f"size mismatch: got {written}, expected {expected_size}")
                tmp.replace(local_path)
                return written
            except TokenExpired:
                raise
            except (requests.RequestException, RuntimeError) as e:
                if attempt == retries:
                    raise
                print(f"    attempt {attempt} failed: {str(e)[:100]}; retrying", flush=True)
                time.sleep(5)
        return 0
