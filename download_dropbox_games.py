"""
Download game videos from the SoccerReplay-1988 Dropbox shared folder.

Uses the Dropbox API (sharing/get_shared_link_file) with the shared link
password. Requires a short-lived access token with sharing.read +
files.content.read scopes.

Usage:
    python download_dropbox_games.py --token-file <path> \
        --games-file tmp_dbx/selected_games.json
"""

import argparse
import json
import pathlib
import sys
import time

import requests

SHARED_URL = ("https://www.dropbox.com/scl/fo/x7klm2zbd5tnw1qn4ilbs/"
              "ABe03XYpBBPtolxrsIf0lK4?rlkey=ftc9m99fzegazg5vhnpik7nmk"
              "&st=vbbm0p2c&dl=0")
LINK_PASSWORD = "sjtu-ai4sports4ever"
LEAGUE_DIR = "/england_epl_2021-2022"
OUT_BASE = pathlib.Path("data/videos")

parser = argparse.ArgumentParser()
parser.add_argument("--token-file", required=True)
parser.add_argument("--games-file", required=True)
args = parser.parse_args()

token = pathlib.Path(args.token_file).read_text().strip()
games = json.load(open(args.games_file))

session = requests.Session()


def list_folder(path):
    r = session.post(
        "https://api.dropboxapi.com/2/files/list_folder",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        json={"path": path,
              "shared_link": {"url": SHARED_URL,
                              "password": LINK_PASSWORD}},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["entries"]


def download_file(remote_path, local_path, size):
    tmp = local_path.with_suffix(local_path.suffix + ".part")
    with session.post(
        "https://content.dropboxapi.com/2/sharing/get_shared_link_file",
        headers={
            "Authorization": f"Bearer {token}",
            "Dropbox-API-Arg": json.dumps(
                {"url": SHARED_URL, "path": remote_path,
                 "link_password": LINK_PASSWORD}),
        },
        stream=True, timeout=120,
    ) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        done = 0
        t0 = time.time()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
        speed = done / max(time.time() - t0, 1e-6) / 1e6
        tmp.rename(local_path)
        print(f"    {local_path.name}  {done/1e9:.2f} GB  ({speed:.1f} MB/s)",
              flush=True)


total_bytes = 0
for gi, game in enumerate(games, 1):
    out_dir = OUT_BASE / game
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{gi}/{len(games)}] {game}", flush=True)
    try:
        entries = list_folder(f"{LEAGUE_DIR}/{game}")
    except Exception as e:
        print(f"    LIST FAILED: {e}", flush=True)
        continue
    for e in entries:
        if e[".tag"] != "file" or not e["name"].endswith(".mkv"):
            continue
        local = out_dir / e["name"]
        if local.exists() and local.stat().st_size == e["size"]:
            print(f"    {e['name']}  already downloaded, skipping", flush=True)
            continue
        remote = f"{LEAGUE_DIR}/{game}/{e['name']}"
        for attempt in range(3):
            try:
                download_file(remote, local, e["size"])
                total_bytes += e["size"]
                break
            except Exception as ex:
                print(f"    attempt {attempt+1} failed: {ex}", flush=True)
                time.sleep(5)
        else:
            print(f"    GIVING UP on {e['name']}", flush=True)

print(f"\nDone. Downloaded {total_bytes/1e9:.1f} GB total.")
