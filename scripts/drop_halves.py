"""
Copy a clip_temporal_padded.npz without the events of given game halves.

Used to leave out halves whose frame timing failed the check in the extraction
markers (soccer.frames.time_check), without touching the original features.

Usage:
    python scripts/drop_halves.py --features features/temporal_dinov2-base --out other/temporal_dinov2-base \
        --flagged                      # every half above the timing tolerance, from the frames' markers
    python scripts/drop_halves.py --features ... --out ... --halves game_288_half_2 game_205_half_1
"""

import argparse
import pathlib

import numpy as np

from soccer.config import data_path, load_config
from soccer.features import TEMPORAL_FILE
from soccer.frames import parse_half_dir, time_check
from soccer.manifest import normalize_game_id


def keep_mask(games: np.ndarray, halves: np.ndarray, drop: set[tuple[str, str]]) -> np.ndarray:
    return np.array([(g, h) not in drop for g, h in zip(games, halves, strict=True)])


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Copy features without some game halves.")
    parser.add_argument("--features", required=True, help="directory holding clip_temporal_padded.npz")
    parser.add_argument("--out", required=True, help="directory to write the filtered copy")
    parser.add_argument("--halves", nargs="*", default=[], help="half folders to drop, e.g. game_288_half_2")
    parser.add_argument("--flagged", action="store_true", help="also drop every half that failed the timing check")
    args = parser.parse_args(argv)

    names = set(args.halves)
    if args.flagged:
        names |= set(time_check(data_path(cfg, "frames"))["flagged"])
    drop = {parse_half_dir(n) for n in names}
    if None in drop:
        raise SystemExit(f"not half folder names: {sorted(n for n in names if parse_half_dir(n) is None)}")

    npz = np.load(pathlib.Path(args.features) / TEMPORAL_FILE, allow_pickle=False)
    games = np.array([normalize_game_id(g) for g in npz["games"]])
    keep = keep_mask(games, npz["halves"].astype(str), drop)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / TEMPORAL_FILE, **{k: npz[k][keep] for k in npz.files})
    print(f"dropped {len(drop)} halves ({', '.join(sorted(names))}): {(~keep).sum()} of {len(keep)} events; "
          f"wrote {out / TEMPORAL_FILE}")


if __name__ == "__main__":
    main()
