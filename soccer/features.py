"""Loading per-event temporal features and pooling them into event vectors."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, fields

import numpy as np

from soccer.manifest import normalize_game_id

TEMPORAL_FILE = "clip_temporal_padded.npz"


@dataclass
class TemporalFeatures:
    features: np.ndarray    # (E, T, D) frame embeddings, zero-padded
    masks: np.ndarray       # (E, T) True for real frames
    labels: np.ndarray      # (E,) event label
    games: np.ndarray       # (E,) game number, normalised to "07"
    halves: np.ndarray      # (E,) "1" or "2"
    event_tags: np.ndarray  # (E,) commentary timestamp "MM-SS.sss"

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, keep: np.ndarray) -> "TemporalFeatures":
        return TemporalFeatures(**{f.name: getattr(self, f.name)[keep] for f in fields(self)})

    def drop_labels(self, labels: set[str]) -> "TemporalFeatures":
        return self.subset(~np.isin(self.labels, list(labels)))


def load_temporal(path: str | pathlib.Path) -> TemporalFeatures:
    """Read clip_temporal_padded.npz (a file, or the directory holding it)."""
    path = pathlib.Path(path)
    if path.is_dir():
        path = path / TEMPORAL_FILE
    npz = np.load(path, allow_pickle=False)
    return TemporalFeatures(
        features=npz["features"].astype(np.float32),
        masks=npz["masks"].astype(bool),
        labels=npz["labels"].astype(str),
        games=np.array([normalize_game_id(g) for g in npz["games"]]),
        halves=npz["halves"].astype(str),
        event_tags=npz["event_tags"].astype(str),
    )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)


def mean_pool(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Average of the real frames of each event, L2-normalised."""
    n = masks.sum(axis=1, keepdims=True).clip(min=1)
    return l2_normalize((features * masks[:, :, None]).sum(axis=1) / n)


def max_pool(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    masked = np.where(masks[:, :, None], features, -np.inf)
    out = masked.max(axis=1)
    return l2_normalize(np.where(np.isfinite(out), out, 0.0))


def first_frame(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return l2_normalize(features[:, 0, :])


def last_frame(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    last = (masks.sum(axis=1) - 1).clip(min=0)
    return l2_normalize(features[np.arange(len(features)), last, :])


# Note: a softmax over per-frame L2 norms ("attention pooling") equals the mean
# here, because the encoders already normalise every frame to length 1. It was
# removed rather than reported as a separate method.
POOLING = {
    "mean_pool": mean_pool,
    "max_pool": max_pool,
    "first_frame": first_frame,
    "last_frame": last_frame,
}


def center_per_group(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean (e.g. per game), then re-normalise."""
    out = x.astype(np.float32).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(axis=0)
    return l2_normalize(out)
