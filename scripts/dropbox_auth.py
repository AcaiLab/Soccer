"""
One-time Dropbox authorisation that gives the pipeline a credential which does not expire.

A Dropbox access token lasts about four hours. A refresh token lasts until you revoke it,
and soccer.dropbox uses it to fetch a new access token whenever one is needed, so a
download that takes days needs no one to paste a token.

The script uses PKCE, the flow for programs that cannot keep a secret. There is no app
secret to protect, and the code Dropbox shows in the browser is useless without the
verifier, which stays on this machine.

Before you start, in the Dropbox App Console (https://www.dropbox.com/developers/apps):
  - open your app and copy its "App key" (it is not a secret);
  - Settings: "Allow public clients (Implicit Grant & PKCE)" must be Allow;
  - Permissions: tick the scopes the downloader needs (files.metadata.read,
    files.content.read, sharing.read) and press Submit.

Usage:
    python scripts/dropbox_auth.py start --app-key APP_KEY   # prints a link: open it, press Allow
    python scripts/dropbox_auth.py finish                    # asks for the code the page shows
    python scripts/dropbox_auth.py check                     # fetches a token, lists the shared link

`finish` writes tmp_dbx/oauth.json (app key and refresh token; owner-only, gitignored).
Copy it to the machine that runs the pipeline and point DROPBOX_OAUTH_FILE at it; see
docs/remote_gpu_server.md. No token is ever printed.
"""

import argparse
import base64
import getpass
import hashlib
import json
import os
import pathlib
import secrets
import urllib.parse

import requests

from soccer.dropbox import (
    DEFAULT_OAUTH_FILE,
    TOKEN_URL,
    DropboxSharedLink,
    RefreshToken,
    TokenExpired,
    oauth_error,
)

AUTHORIZE_URL = "https://www.dropbox.com/oauth2/authorize"
STATE_NAME = "pkce_state.json"


def new_verifier() -> str:
    """A random PKCE verifier: 86 URL-safe characters (the allowed range is 43 to 128)."""
    return secrets.token_urlsafe(64)


def challenge_for(verifier: str) -> str:
    """The S256 challenge of a verifier (RFC 7636): unpadded base64url of its SHA-256."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def authorize_url(app_key: str, challenge: str, scopes: list[str] | None = None) -> str:
    """The link to open; token_access_type=offline is what makes Dropbox issue a refresh token."""
    params = {"client_id": app_key, "response_type": "code", "code_challenge": challenge,
              "code_challenge_method": "S256", "token_access_type": "offline"}
    if scopes:
        params["scope"] = " ".join(scopes)
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


def write_private(path: pathlib.Path, text: str) -> None:
    """Write a file only its owner can read (on Windows the folder's permissions apply)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o600)


def exchange_code(app_key: str, code: str, verifier: str, session=None) -> dict:
    """Trade the code from the browser for tokens; the verifier proves we started the request."""
    r = (session or requests).post(TOKEN_URL, timeout=30, data={
        "code": code.strip(), "grant_type": "authorization_code",
        "code_verifier": verifier, "client_id": app_key})
    if r.status_code != 200:
        raise SystemExit(f"Dropbox refused the code (HTTP {r.status_code} {oauth_error(r)}). A code works "
                         f"once and expires within minutes: run 'start' again for a fresh link.")
    body = r.json()
    if not body.get("refresh_token"):
        raise SystemExit("Dropbox sent no refresh token: authorise with the link that 'start' prints "
                         "(it asks for token_access_type=offline).")
    return body


def start(oauth_file: pathlib.Path, app_key: str | None, scopes: list[str] | None) -> None:
    app_key = app_key or os.environ.get("DROPBOX_APP_KEY")
    if not app_key:
        raise SystemExit("Give the app key: --app-key APP_KEY (Dropbox App Console > your app > Settings)")
    verifier = new_verifier()
    write_private(oauth_file.parent / STATE_NAME, json.dumps({"app_key": app_key, "verifier": verifier}))
    print("1. Open this link in a browser signed in to Dropbox and press Allow:\n")
    print(authorize_url(app_key, challenge_for(verifier), scopes))
    print("\n2. Dropbox then shows a code. Run:  python scripts/dropbox_auth.py finish")


def finish(oauth_file: pathlib.Path, code: str | None, code_file: str | None, session=None) -> None:
    state_file = oauth_file.parent / STATE_NAME
    if not state_file.is_file():
        raise SystemExit(f"{state_file} not found: run 'start' first (and 'finish' on the same machine).")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if code_file:
        code = pathlib.Path(code_file).read_text(encoding="utf-8")
    code = code or getpass.getpass("Code shown by Dropbox (input is hidden): ")
    body = exchange_code(state["app_key"], code, state["verifier"], session)
    write_private(oauth_file, json.dumps({"app_key": state["app_key"],
                                          "refresh_token": body["refresh_token"]}, indent=2) + "\n")
    state_file.unlink()
    print(f"Saved {oauth_file} (the refresh token is in that file only, not shown here).")
    print(f"Permissions granted: {body.get('scope', 'not reported')}")
    print("Next: python scripts/dropbox_auth.py check")


def check(oauth_file: pathlib.Path, session=None) -> int:
    """Prove the saved refresh token works and that the shared link can be read with it."""
    if not oauth_file.is_file():
        raise SystemExit(f"{oauth_file} not found: run 'start' and 'finish' first.")
    saved = json.loads(oauth_file.read_text(encoding="utf-8"))
    tokens = RefreshToken(saved["app_key"], saved["refresh_token"], saved.get("app_secret"), session=session)
    tokens.get()
    print("Dropbox issued an access token from the saved refresh token.")
    url = os.environ.get("DROPBOX_SHARED_URL")
    if not url:
        print("DROPBOX_SHARED_URL is not set, so the shared link itself was not tested.")
        return 0
    client = DropboxSharedLink(tokens, url, os.environ.get("DROPBOX_LINK_PASSWORD"), session=session)
    entries = client.list_folder("")
    print(f"The shared link lists {len(entries)} entries: the app has the permissions it needs.")
    return len(entries)


def main(argv=None, session=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--oauth-file", default=os.environ.get("DROPBOX_OAUTH_FILE") or DEFAULT_OAUTH_FILE,
                        help="where the app key and refresh token are kept (default: $DROPBOX_OAUTH_FILE, "
                             f"else {DEFAULT_OAUTH_FILE})")
    parser = argparse.ArgumentParser(description="One-time Dropbox authorisation (refresh token).")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("start", parents=[common], help="print the link to open and press Allow")
    p.add_argument("--app-key", default=None, help="the app's key (default: $DROPBOX_APP_KEY)")
    p.add_argument("--scope", nargs="+", default=None,
                   help="ask for only these scopes (default: every scope the app has)")
    p = commands.add_parser("finish", parents=[common], help="trade the code Dropbox shows for a refresh token")
    p.add_argument("--code", default=None, help="the code (default: asked for, hidden); the code alone is "
                                                "useless without the verifier saved by 'start'")
    p.add_argument("--code-file", default=None, help="read the code from this file instead")
    commands.add_parser("check", parents=[common], help="fetch a token and list the shared link")
    args = parser.parse_args(argv)

    oauth_file = pathlib.Path(args.oauth_file).expanduser()
    try:
        if args.command == "start":
            start(oauth_file, args.app_key, args.scope)
        elif args.command == "finish":
            finish(oauth_file, args.code, args.code_file, session)
        else:
            check(oauth_file, session)
    except TokenExpired as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
