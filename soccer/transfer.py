"""Copy match videos from the Dropbox shared link straight into S3.

Each file is streamed from Dropbox into an S3 multipart upload: nothing is
written to local disk, so this runs the same on a server, an EC2 instance or
an AWS Lambda function (see `handler`). Layout in S3 mirrors Dropbox:

    <dest>/<league>/<match_id>/<match>_1.mkv, <match>_2.mkv

Integrity: S3 verifies a SHA-256 checksum of every uploaded part, and the
object's final size must equal the size Dropbox lists, or it is deleted and
the copy fails. A file already in S3 with the right size is skipped, so a
rerun resumes where the last one stopped.
"""

from __future__ import annotations

from soccer.dropbox import DropboxSharedLink
from soccer.extract import VIDEO_EXTS
from soccer.manifest import Game
from soccer.storage import split_s3

CHUNK = 64 * 1024 * 1024


def _s3():
    import boto3

    return boto3.client("s3")


def _size_in_s3(s3, bucket: str, key: str) -> int | None:
    from botocore.exceptions import ClientError

    try:
        return s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def copy_file(client: DropboxSharedLink, remote_path: str, bucket: str, key: str, size: int, s3=None) -> str:
    """Stream one Dropbox file into s3://bucket/key; returns "copied" or "already in S3"."""
    from boto3.s3.transfer import TransferConfig

    s3 = s3 or _s3()
    if _size_in_s3(s3, bucket, key) == size:
        return "already in S3"
    with client.open_file(remote_path) as r:
        r.raw.decode_content = True
        s3.upload_fileobj(
            r.raw, bucket, key,
            ExtraArgs={"ChecksumAlgorithm": "SHA256", "Metadata": {"source-size": str(size)}},
            Config=TransferConfig(multipart_threshold=CHUNK, multipart_chunksize=CHUNK, max_concurrency=4),
        )
    got = _size_in_s3(s3, bucket, key)
    if got != size:
        s3.delete_object(Bucket=bucket, Key=key)
        raise RuntimeError(f"{key}: S3 has {got} bytes, Dropbox listed {size}; deleted, retry")
    return "copied"


def copy_game(client: DropboxSharedLink, game: Game, dest: str, s3=None) -> list[tuple[str, int, str]]:
    """Copy both halves of a game; returns (file name, bytes, outcome) per file."""
    bucket, prefix = split_s3(dest.rstrip("/"))
    entries = client.list_folder(f"/{game.league}/{game.match_id}")
    videos = sorted((e for e in entries if e[".tag"] == "file" and e["name"].lower().endswith(VIDEO_EXTS)),
                    key=lambda e: e["name"])
    if not videos:
        raise FileNotFoundError(f"No video files in /{game.league}/{game.match_id}")
    results = []
    for e in videos:
        key = "/".join(p for p in (prefix, game.league, game.match_id, e["name"]) if p)
        outcome = copy_file(client, f"/{game.league}/{game.match_id}/{e['name']}", bucket, key, e["size"], s3)
        results.append((e["name"], e["size"], outcome))
    return results


def handler(event, context):
    """AWS Lambda entry point: copy one game.

    event = {"league": "...", "match_id": "...", "dest": "s3://bucket/videos"}
    Dropbox settings come from SSM (SOCCER_SSM_PREFIX) via the function's role.
    """
    game = Game("00", event["match_id"], event["league"], "", "")
    results = copy_game(DropboxSharedLink.from_env(), game, event["dest"])
    return {"match_id": game.match_id,
            "files": [{"name": n, "bytes": b, "outcome": o} for n, b, o in results]}
