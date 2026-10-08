"""
Download the SoccerReplay-1988 annotation JSONs onto this machine.

Sources (configs/pipeline.yaml, annotations_source):
  drive  the Google Drive copy, by file id (no login)
  hf     the gated Hugging Face repo (needs HF_TOKEN with approved access)
  s3     a copy previously uploaded with --upload (<s3_root>/json)

Installs to <annotations path>/<split>/<league>/<match>/*.json, the layout
every script expects. --upload also copies the installed JSONs to S3 so other
machines can use `--source s3`.

Usage:
    python scripts/fetch_annotations.py                         # train split from Google Drive
    python scripts/fetch_annotations.py --splits train valid test --upload
    python scripts/fetch_annotations.py --source s3
"""

import argparse
import os
import pathlib
import tempfile

from soccer.annotations import install_split_zip
from soccer.config import data_path, load_config, s3_uri
from soccer.storage import download_prefix, upload_dir


def fetch_zip(source: str, split: str, cfg: dict, tmp: pathlib.Path) -> pathlib.Path:
    src = cfg["annotations_source"]
    if source == "drive":
        import gdown

        out = tmp / f"{split}.zip"
        gdown.download(id=src["google_drive_files"][split], output=str(out), quiet=False)
        return out
    from huggingface_hub import hf_hub_download

    return pathlib.Path(hf_hub_download(src["huggingface_repo"], f"{split}.zip", repo_type="dataset",
                                        token=os.environ.get("HF_TOKEN"), local_dir=tmp))


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Download SoccerReplay-1988 annotations.")
    parser.add_argument("--source", choices=["drive", "hf", "s3"], default="drive")
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["train"])
    parser.add_argument("--upload", action="store_true", help="also copy the installed JSONs to <s3_root>/json")
    args = parser.parse_args(argv)

    root = data_path(cfg, "annotations")
    if args.source == "s3":
        for split in args.splits:
            n = download_prefix(f"{s3_uri(cfg, 'json')}/{split}/", root / split)
            print(f"{split}: {n} files from S3")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            for split in args.splits:
                zip_path = fetch_zip(args.source, split, cfg, pathlib.Path(tmp))
                print(f"{split}: {install_split_zip(zip_path, split, root)} match JSONs installed in {root / split}")
    if args.upload:
        for split in args.splits:
            n = upload_dir(root / split, f"{s3_uri(cfg, 'json')}/{split}")
            print(f"{split}: uploaded {n} files to {s3_uri(cfg, 'json')}/{split}")


if __name__ == "__main__":
    main()
