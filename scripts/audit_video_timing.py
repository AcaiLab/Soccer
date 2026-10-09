"""
Audit whether a video's timestamps agree with its frame count.

Frame extraction picks frames by index (time x fps). The extraction marker of a half
records the largest gap between the decoder's timestamp and that index time; a gap above
0.5 s means the saved frames may not be the ones at the annotated moments. This script
finds out what such a gap is: a constant offset, a jump at one place (and what the
picture does there), or a steady drift.

For every audited half it writes <out-dir>/<name>.json (the profile) and, with --scan,
<out-dir>/<name>_jump<k>.png (the frame before | the frame after each jump).

Usage:
    python scripts/audit_video_timing.py --video half1.mkv half2.mkv --scan
    python scripts/audit_video_timing.py --games 35 63 --scan     # download from Dropbox, audit, delete
    python scripts/audit_video_timing.py --games 01 02 03         # quick: seek samples only (seconds per half)
    python scripts/audit_video_timing.py --summary                # table of everything audited so far
"""

import argparse
import json
import pathlib
import shutil
import sys

from soccer.config import data_path, load_config, repo_path
from soccer.dropbox import DropboxSharedLink, TokenExpired
from soccer.extract import MAX_TIME_ERROR_SEC
from soccer.manifest import load_manifest, normalize_game_id
from soccer.pipeline import download_videos, half_video
from soccer.timing import profile_video


def audit(video: pathlib.Path, name: str, out_dir: pathlib.Path, step_sec: float, scan: bool) -> dict:
    import cv2
    import numpy as np

    profile = profile_video(video, step_sec=step_sec, scan=scan)
    out_dir.mkdir(parents=True, exist_ok=True)
    jumps = []
    for k, j in enumerate(profile.jumps):
        entry = {"index": j.index, "index_time": round(j.index_time, 2), "seconds": round(j.seconds, 3),
                 "picture_change": None if j.picture_change is None else round(j.picture_change, 2),
                 "typical_change": None if j.typical_change is None else round(j.typical_change, 2)}
        if j.before is not None:
            image = f"{name}_jump{k}.png"
            cv2.imwrite(str(out_dir / image), np.hstack([j.before, j.after]))
            entry["image"] = image
        jumps.append(entry)
    record = {"name": name, "video": video.name, "fps": profile.fps, "frames": profile.frames,
              "scanned": profile.scanned, "max_error": round(profile.max_error, 3),
              "final_error": None if profile.final_error is None else round(profile.final_error, 3),
              "samples": [(round(t, 1), round(e, 3)) for t, e in profile.samples], "jumps": jumps}
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
    print(describe(record), flush=True)
    return record


def describe(r: dict) -> str:
    verdict = "ok" if r["max_error"] <= MAX_TIME_ERROR_SEC else "FLAGGED"
    line = f"{r['name']:16s} {verdict:8s} max error {r['max_error']:6.2f}s  fps {r['fps']:g}  {r['frames']} frames"
    if r["scanned"]:
        line += f"  final error {r['final_error']:.2f}s  {len(r['jumps'])} jump(s)"
        for j in r["jumps"][:3]:
            line += (f"\n    at {j['index_time']:.1f}s the timestamp jumps by {j['seconds']:+.2f}s; picture change "
                     f"{j['picture_change']} (typical {j['typical_change']})" + (f"  [{j['image']}]" if "image" in j else ""))
    elif r["max_error"] > MAX_TIME_ERROR_SEC:
        first_bad = next(t for t, e in r["samples"] if abs(e) > MAX_TIME_ERROR_SEC)
        line += f"  first sample off at {first_bad:.0f}s, samples: " + " ".join(f"{e:+.1f}" for _, e in r["samples"])[:160]
    return line


def summary(out_dir: pathlib.Path) -> None:
    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not records:
        raise SystemExit(f"No audits found in {out_dir}")
    flagged = [r for r in records if r["max_error"] > MAX_TIME_ERROR_SEC]
    print(f"{len(records)} halves audited, {len(flagged)} above {MAX_TIME_ERROR_SEC}s\n")
    for r in records:
        print(describe(r))


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Audit video timestamps against frame counts.")
    parser.add_argument("--video", nargs="+", default=None, help="local video files")
    parser.add_argument("--games", nargs="+", default=None,
                        help="game numbers of the manifest: download both halves from Dropbox, audit, delete")
    parser.add_argument("--scan", action="store_true",
                        help="decode each video completely to locate every timestamp jump (minutes per half)")
    parser.add_argument("--step-sec", type=float, default=30.0, help="spacing of the seek samples")
    parser.add_argument("--summary", action="store_true", help="print the audits found in --out-dir and exit")
    parser.add_argument("--keep-videos", action="store_true", help="keep downloaded videos")
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--out-dir", default=str(data_path(cfg, "results") / "timing_audit"))
    args = parser.parse_args(argv)
    out_dir = pathlib.Path(args.out_dir)

    if args.summary:
        return summary(out_dir)
    if bool(args.video) == bool(args.games):
        parser.error("give exactly one of --video, --games or --summary")

    if args.video:
        for v in map(pathlib.Path, args.video):
            audit(v, v.stem, out_dir, args.step_sec, args.scan)
        return

    games = load_manifest(repo_path(cfg, "manifest"), repo_path(cfg, "excluded_matches"))
    wanted = {normalize_game_id(g) for g in args.games}
    selected = [g for g in games if g.game_num in wanted]
    if len(selected) != len(wanted):
        parser.error(f"not in the manifest: {sorted(wanted - {g.game_num for g in selected})}")
    client = DropboxSharedLink.from_env(token_file=args.token_file)
    videos_root = data_path(cfg, "videos")
    failed = []
    for game in selected:
        print(f"game_{game.game_num}: {game.match_id}", flush=True)
        try:
            video_dir = download_videos(client, game, videos_root)
            for half in (1, 2):
                audit(half_video(video_dir, half), f"game_{game.game_num}_half_{half}", out_dir,
                      args.step_sec, args.scan)
        except TokenExpired as e:
            print(f"\n*** {e}. Renew the Dropbox credentials (README, Credentials) and rerun. ***")
            sys.exit(2)
        except Exception as e:  # one bad game should not stop the rest
            print(f"    FAILED: {e}", flush=True)
            failed.append(game.game_num)
            continue
        if not args.keep_videos:
            shutil.rmtree(video_dir, ignore_errors=True)
    if failed:
        print(f"Failed: {' '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
