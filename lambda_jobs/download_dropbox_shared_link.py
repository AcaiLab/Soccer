import argparse
import os
from pathlib import Path

import dropbox


def clean_lines(path: Path) -> list[str]:
    if not path:
        return [""]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(stripped)
    return rows


def output_path(out_dir: Path, rel_path: str, fallback_name: str) -> Path:
    if not rel_path:
        return out_dir / fallback_name
    rel = rel_path.lstrip("/")
    return out_dir / rel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("DROPBOX_SHARED_LINK"))
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ.get("OUT_DIR", "/workspace/data/SoccerReplay-1988/dropbox")))
    parser.add_argument("--paths-file", type=Path, help="Relative paths inside a shared folder link.")
    parser.add_argument("--fallback-name", default="dropbox_shared_file.bin")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("DROPBOX_ACCESS_TOKEN")
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")
    password = os.environ.get("DROPBOX_SHARED_LINK_PASSWORD")
    if not args.url:
        raise SystemExit("Missing --url or DROPBOX_SHARED_LINK")
    if refresh_token:
        if not app_key or not app_secret:
            raise SystemExit("DROPBOX_REFRESH_TOKEN requires DROPBOX_APP_KEY and DROPBOX_APP_SECRET")
        dbx = dropbox.Dropbox(oauth2_refresh_token=refresh_token, app_key=app_key, app_secret=app_secret)
    elif token:
        dbx = dropbox.Dropbox(token)
    else:
        raise SystemExit(
            "Missing Dropbox credentials. Set DROPBOX_REFRESH_TOKEN with DROPBOX_APP_KEY/DROPBOX_APP_SECRET, "
            "or set DROPBOX_ACCESS_TOKEN."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rel_paths = clean_lines(args.paths_file) if args.paths_file else [""]
    for rel_path in rel_paths:
        dest = output_path(args.out_dir, rel_path, args.fallback_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading shared link path={rel_path or '<root file>'} -> {dest}")
        metadata = dbx.sharing_get_shared_link_file_to_file(
            str(dest),
            args.url,
            path=rel_path or None,
            link_password=password,
        )
        print(f"  saved {dest} ({dest.stat().st_size} bytes), metadata={metadata.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
