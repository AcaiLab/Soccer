"""Load configs/pipeline.yaml and resolve the paths in it."""

from __future__ import annotations

import os
import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read the pipeline config.

    The file is `path`, else $SOCCER_CONFIG, else configs/pipeline.yaml. The data
    root is $SOCCER_DATA_ROOT if set, else `data_root` from the file; a relative
    data root is taken relative to the repository.
    """
    path = pathlib.Path(path or os.environ.get("SOCCER_CONFIG") or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = pathlib.Path(os.environ.get("SOCCER_DATA_ROOT") or cfg.get("data_root", "."))
    cfg["data_root"] = root if root.is_absolute() else (REPO_ROOT / root).resolve()
    return cfg


def data_path(cfg: dict, key: str) -> pathlib.Path:
    """A path from the `paths:` section, resolved against the data root."""
    p = pathlib.Path(cfg["paths"][key])
    return p if p.is_absolute() else cfg["data_root"] / p


def repo_path(cfg: dict, key: str) -> pathlib.Path:
    """A top-level path that lives in the repository (manifest, exclusions)."""
    p = pathlib.Path(cfg[key])
    return p if p.is_absolute() else REPO_ROOT / p


def dropped_labels(cfg: dict) -> set[str]:
    """Labels removed before clustering and classification."""
    labels = cfg.get("labels", {})
    return set(labels.get("junk", [])) | set(labels.get("non_visual", []))
