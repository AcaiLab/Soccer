"""Download SoccerReplay-1988 videos from the password-protected Dropbox link.

Credentials never live in code. They come from, in order:

  1. environment variables
       DROPBOX_TOKEN, or a token file: the --token-file option if given,
       else DROPBOX_TOKEN_FILE, else tmp_dbx/token.txt
       DROPBOX_SHARED_URL
       DROPBOX_LINK_PASSWORD
  2. AWS SSM Parameter Store, if SOCCER_SSM_PREFIX is set:
       <prefix>/dropbox/token, <prefix>/dropbox/shared_url,
       <prefix>/dropbox/link_password  (SecureString)
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import requests

API = "https://api.dropboxapi.com/2"
DEFAULT_TOKEN_FILE = "tmp_dbx/token.txt"
CONTENT = "https://content.dropboxapi.com/2"


class TokenExpired(RuntimeError):
    """The access token was rejected; get a new one and resume."""


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


class DropboxSharedLink:
    def __init__(self, token: str, url: str, password: str | None = None,
                 session: requests.Session | None = None) -> None:
        self._token = token
        self._link = {"url": url, **({"password": password} if password else {})}
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls, token_file: str | os.PathLike | None = None) -> "DropboxSharedLink":
        token = _setting("DROPBOX_TOKEN", "token")
        token_file = token_file or os.environ.get("DROPBOX_TOKEN_FILE") or DEFAULT_TOKEN_FILE
        if not token and token_file and pathlib.Path(token_file).is_file():
            token = pathlib.Path(token_file).read_text(encoding="utf-8").strip()
        url = _setting("DROPBOX_SHARED_URL", "shared_url")
        password = _setting("DROPBOX_LINK_PASSWORD", "link_password")
        missing = [n for n, v in [("DROPBOX_TOKEN or a token file", token),
                                  ("DROPBOX_SHARED_URL", url)] if not v]
        if missing:
            raise RuntimeError("Missing Dropbox settings: " + ", ".join(missing) + " (see README)")
        return cls(token, url, password)

    def _check(self, r: requests.Response) -> None:
        if r.status_code == 200:
            return  # never touch r.text here: it would read a streamed video into memory
        body = r.text[:500]
        if r.status_code == 401 or "access_token" in body:
            raise TokenExpired(f"Dropbox rejected the token (HTTP {r.status_code})")
        raise RuntimeError(f"Dropbox HTTP {r.status_code}: {body[:300]}")

    def list_folder(self, path: str) -> list[dict]:
        """Entries of a folder inside the shared link, following pagination."""
        headers = {"Authorization": f"Bearer {self._token}"}
        r = self._session.post(f"{API}/files/list_folder", headers=headers, timeout=60,
                               json={"path": path, "shared_link": self._link})
        self._check(r)
        data = r.json()
        entries = data["entries"]
        while data.get("has_more"):
            r = self._session.post(f"{API}/files/list_folder/continue", headers=headers,
                                   timeout=60, json={"cursor": data["cursor"]})
            self._check(r)
            data = r.json()
            entries += data["entries"]
        return entries

    def open_file(self, remote_path: str) -> requests.Response:
        """Streaming response for a file inside the shared link; use as a context manager."""
        arg = {"url": self._link["url"], "path": remote_path}
        if "password" in self._link:
            arg["link_password"] = self._link["password"]
        r = self._session.post(f"{CONTENT}/sharing/get_shared_link_file", stream=True, timeout=120,
                               headers={"Authorization": f"Bearer {self._token}",
                                        "Dropbox-API-Arg": json.dumps(arg)})
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
        if expected_size is not None and local_path.is_file()                 and local_path.stat().st_size == expected_size:
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
