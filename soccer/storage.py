"""Read data from a local path or an s3://bucket/prefix URI.

Credentials come from the standard AWS chain (instance or task role, AWS_PROFILE,
...). Nothing here reads keys from code or config files.
"""

from __future__ import annotations

import pathlib

S3_PREFIX = "s3://"


def is_s3(uri: str | pathlib.Path) -> bool:
    return str(uri).startswith(S3_PREFIX)


def split_s3(uri: str) -> tuple[str, str]:
    """"s3://bucket/a/b" -> ("bucket", "a/b")."""
    bucket, _, key = uri[len(S3_PREFIX):].partition("/")
    return bucket, key


def _client():
    import boto3  # optional dependency, only needed for s3:// paths

    return boto3.client("s3")


def list_subdirs(uri: str | pathlib.Path) -> list[str]:
    """Names of the immediate sub-folders of a local directory or S3 prefix."""
    if not is_s3(uri):
        return sorted(p.name for p in pathlib.Path(uri).iterdir() if p.is_dir())
    bucket, prefix = split_s3(str(uri).rstrip("/") + "/")
    names = []
    for page in _client().get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix, Delimiter="/"):
        names += [cp["Prefix"][len(prefix):].rstrip("/") for cp in page.get("CommonPrefixes", [])]
    return sorted(names)


def download_prefix(uri: str, local_dir: str | pathlib.Path) -> int:
    """Copy every object under an S3 prefix into local_dir; returns file count."""
    bucket, prefix = split_s3(uri.rstrip("/") + "/")
    local_dir = pathlib.Path(local_dir)
    s3, n = _client(), 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(dest))
            n += 1
    return n


def local_file(uri: str | pathlib.Path, cache_dir: str | pathlib.Path) -> pathlib.Path:
    """A local path for `uri`, downloading it into cache_dir first if it is on S3."""
    if not is_s3(uri):
        return pathlib.Path(uri)
    bucket, key = split_s3(str(uri))
    dest = pathlib.Path(cache_dir) / bucket / key
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _client().download_file(bucket, key, str(dest))
    return dest
