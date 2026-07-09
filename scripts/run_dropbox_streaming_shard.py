import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import dropbox
import pandas as pd
import requests
from dropbox.files import FileMetadata, SharedLink

from pipeline_utils import ensure_dir, read_jsonl, sanitize_name, write_json, write_jsonl

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


@dataclass(frozen=True)
class Event:
    half: int
    timestamp: str
    event_label: str
    event_seconds: float
    text: str


def parse_timestamp(timestamp: str) -> float:
    match = re.fullmatch(r"(\d+)(?:\+(\d+))?:(\d{2})", timestamp.strip())
    if not match:
        raise ValueError(f"Unrecognized timestamp: {timestamp!r}")
    minutes = int(match.group(1))
    added = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3))
    return (minutes + added) * 60 + seconds


def seconds_label(seconds: float) -> str:
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}-{rem:06.3f}"


def read_events(annotation_path: Path, max_events: int | None = None) -> list[Event]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    events = []
    seen = set()
    for comment in data.get("comments", []):
        timestamp = (comment.get("time_stamp") or "").strip()
        if not timestamp:
            continue
        try:
            half = int(comment.get("half") or 0)
            event_seconds = parse_timestamp(timestamp)
        except (TypeError, ValueError):
            continue
        if half not in {1, 2}:
            continue
        event_label = sanitize_name(comment.get("comments_type", "unknown"))
        key = (half, timestamp, event_label)
        if key in seen:
            continue
        seen.add(key)
        events.append(
            Event(
                half=half,
                timestamp=timestamp,
                event_label=event_label,
                event_seconds=event_seconds,
                text=comment.get("comments_text_anonymized") or comment.get("comments_text") or "",
            )
        )
    events = sorted(events, key=lambda item: (item.half, item.event_seconds, item.event_label))
    return events[:max_events] if max_events else events


def dropbox_client() -> tuple[dropbox.Dropbox, SharedLink]:
    token = os.environ.get("DROPBOX_ACCESS_TOKEN")
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")
    url = os.environ.get("DROPBOX_SHARED_LINK")
    password = os.environ.get("DROPBOX_SHARED_LINK_PASSWORD") or None
    if refresh_token:
        if not app_key or not app_secret:
            raise SystemExit(
                "DROPBOX_REFRESH_TOKEN requires DROPBOX_APP_KEY and DROPBOX_APP_SECRET. "
                "Source .env first."
            )
        client = dropbox.Dropbox(oauth2_refresh_token=refresh_token, app_key=app_key, app_secret=app_secret)
    elif token:
        client = dropbox.Dropbox(token)
    else:
        raise SystemExit(
            "Missing Dropbox credentials. Set DROPBOX_REFRESH_TOKEN with DROPBOX_APP_KEY/DROPBOX_APP_SECRET, "
            "or set DROPBOX_ACCESS_TOKEN."
        )
    if not url:
        raise SystemExit("Missing DROPBOX_SHARED_LINK. Source .env first.")
    return client, SharedLink(url=url, password=password)


def list_game_video_paths(dbx: dropbox.Dropbox, shared: SharedLink, dropbox_game_path: str) -> list[str]:
    result = dbx.files_list_folder(path=dropbox_game_path, recursive=False, shared_link=shared)
    entries = list(result.entries)
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries.extend(result.entries)
    paths = []
    for entry in entries:
        if isinstance(entry, FileMetadata) and Path(entry.name).suffix.lower() in VIDEO_EXTS:
            paths.append(f"{dropbox_game_path.rstrip('/')}/{entry.name}")
    return sorted(paths)


def infer_half(path: str, order_index: int) -> int:
    stem = Path(path).stem
    match = re.search(r"(?:^|_)([12])$", stem)
    if match:
        return int(match.group(1))
    return order_index + 1


def download_video(dbx: dropbox.Dropbox, shared_url: str, shared_password: str | None, rel_path: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    ensure_dir(dest.parent)
    dbx.check_and_refresh_access_token()
    token = dbx._oauth2_access_token
    api_arg = {"url": shared_url, "path": rel_path}
    if shared_password:
        api_arg["link_password"] = shared_password
    with requests.post(
        "https://content.dropboxapi.com/2/sharing/get_shared_link_file",
        headers={
            "Authorization": f"Bearer {token}",
            "Dropbox-API-Arg": json.dumps(api_arg),
        },
        stream=True,
        timeout=(30, 120),
    ) as response:
        response.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    return dest


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"Could not read video metadata: {path}")
    return cap, fps, frames / fps


def sample_evenly(paths: list[str], n: int) -> list[str]:
    if not paths:
        return []
    if len(paths) == 1:
        return paths * n
    if len(paths) <= n:
        return paths + [paths[-1]] * (n - len(paths))
    positions = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]
    return [paths[i] for i in positions]


def extract_event_frames(
    cap: cv2.VideoCapture,
    fps: float,
    duration: float,
    event: Event,
    game_label: str,
    frame_root: Path,
    before_sec: float,
    after_sec: float,
    sample_every_sec: float,
    sampled_frames_per_window: int,
    image_size: int | None,
    overwrite: bool,
) -> dict | None:
    start_sec = max(0.0, event.event_seconds - before_sec)
    end_sec = min(duration, event.event_seconds + after_sec)
    if end_sec <= start_sec:
        return None
    event_time_label = seconds_label(event.event_seconds)
    window_id = f"{game_label}_half{event.half}_{event.event_label}_event{event_time_label}"
    frame_dir = frame_root / event.event_label / game_label / window_id
    ensure_dir(frame_dir)
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    sample_frames = {
        int(round((start_sec + idx * sample_every_sec) * fps))
        for idx in range(int((end_sec - start_sec) // sample_every_sec) + 1)
    }
    paths = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx not in sample_frames:
            continue
        frame_time = frame_idx / fps
        out = frame_dir / (
            f"{game_label}_half{event.half}_{event.event_label}_"
            f"event{event_time_label}_frame{seconds_label(frame_time)}.png"
        )
        if overwrite or not out.exists():
            if image_size:
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), frame)
        paths.append(str(out))
    if not paths:
        return None
    return {
        "window_id": window_id,
        "game_label": game_label,
        "event_label": event.event_label,
        "half": event.half,
        "timestamp": event.timestamp,
        "commentary_reference": event.text,
        "frame_count": len(paths),
        "sampled_frame_count": sampled_frames_per_window,
        "image_paths": paths,
        "sampled_image_paths": sample_evenly(paths, sampled_frames_per_window),
    }


def run(cmd: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def run_existing_pipeline(config: Path, shard_dir: Path, device: str, force: bool) -> dict[str, str]:
    manifest = shard_dir / "windows.jsonl"
    outputs = {
        "embeddings": shard_dir / "visual_embeddings.npz",
        "memory": shard_dir / "memory_mean_std.npz",
        "basic_cues": shard_dir / "basic_visual_cues.jsonl",
        "plans": shard_dir / "content_plans.jsonl",
        "plans_visual": shard_dir / "content_plans_with_visual_facts.jsonl",
        "natural": shard_dir / "natural_commentary_with_visual_facts.jsonl",
    }
    log_path = shard_dir / "pipeline.log"
    steps = [
        (
            "embeddings",
            outputs["embeddings"],
            [
                "python3",
                "scripts/extract_visual_embeddings.py",
                "--config",
                str(config),
                "--manifest",
                str(manifest),
                "--out",
                str(outputs["embeddings"]),
                "--device",
                device,
            ],
        ),
        (
            "memory",
            outputs["memory"],
            [
                "python3",
                "scripts/build_retrieval_memory.py",
                "--config",
                str(config),
                "--embeddings",
                str(outputs["embeddings"]),
                "--out",
                str(outputs["memory"]),
                "--feature-mode",
                "mean_std",
            ],
        ),
        (
            "basic_cues",
            outputs["basic_cues"],
            [
                "python3",
                "scripts/extract_basic_visual_cues.py",
                "--config",
                str(config),
                "--manifest",
                str(manifest),
                "--out",
                str(outputs["basic_cues"]),
            ],
        ),
        (
            "plans",
            outputs["plans"],
            [
                "python3",
                "scripts/build_content_plans.py",
                "--config",
                str(config),
                "--manifest",
                str(manifest),
                "--memory",
                str(outputs["memory"]),
                "--out",
                str(outputs["plans"]),
            ],
        ),
        (
            "plans_visual",
            outputs["plans_visual"],
            [
                "python3",
                "scripts/merge_visual_facts_into_plans.py",
                "--plans",
                str(outputs["plans"]),
                "--basic-cues",
                str(outputs["basic_cues"]),
                "--out",
                str(outputs["plans_visual"]),
            ],
        ),
        (
            "natural",
            outputs["natural"],
            [
                "python3",
                "scripts/generate_natural_commentary.py",
                "--config",
                str(config),
                "--plans",
                str(outputs["plans_visual"]),
                "--out",
                str(outputs["natural"]),
            ],
        ),
    ]
    statuses = {}
    for name, output, cmd in steps:
        if output.exists() and output.stat().st_size > 0 and not force:
            statuses[name] = "skipped_existing"
            continue
        run(cmd, log_path)
        statuses[name] = "completed"
    return statuses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("manifests/dropbox_full_dataset_manifest.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/lambda_smoke.yaml"))
    parser.add_argument("--out-root", type=Path, default=Path("dropbox_shards"))
    parser.add_argument("--tmp-video-root", type=Path, default=Path("/home/ubuntu/Soccer_Data/dropbox_tmp_videos"))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-events-per-game", type=int)
    parser.add_argument("--before-sec", type=float, default=15.0)
    parser.add_argument("--after-sec", type=float, default=15.0)
    parser.add_argument("--sample-every-sec", type=float, default=1.0)
    parser.add_argument("--sampled-frames-per-window", type=int, default=8)
    parser.add_argument("--resize-frames", type=int, default=0, help="Optional square resize before writing PNG frames.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-videos", action="store_true")
    parser.add_argument("--cleanup-frames", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    start = time.time()
    df = pd.read_csv(args.dataset_manifest)
    if args.split != "all":
        df = df[df["split"] == args.split].copy()
    df = df.sort_values(["split", "season_folder", "game_folder"]).reset_index(drop=True)
    df = df[df.index % args.num_shards == args.shard_id].copy()
    if args.max_games:
        df = df.head(args.max_games).copy()

    shard_dir = ensure_dir(args.out_root / f"shard_{args.shard_id:04d}")
    frame_root = ensure_dir(shard_dir / "frames")
    manifest_out = shard_dir / "windows.jsonl"
    extraction_summary_path = shard_dir / "extraction_summary.json"

    dbx, shared = dropbox_client()
    shared_url = os.environ["DROPBOX_SHARED_LINK"]
    shared_password = os.environ.get("DROPBOX_SHARED_LINK_PASSWORD") or None

    windows = []
    games_done = []
    failures = []
    for row in df.to_dict("records"):
        annotation_path = Path(row["annotation_path"])
        game_label = row["game_label"]
        game_path = row["dropbox_game_path"]
        try:
            events = read_events(annotation_path, args.max_events_per_game)
            video_paths = list_game_video_paths(dbx, shared, game_path)
            half_to_remote = {infer_half(path, i): path for i, path in enumerate(video_paths)}
            if args.dry_run:
                games_done.append(
                    {
                        "game_label": game_label,
                        "events": len(events),
                        "dropbox_game_path": game_path,
                        "video_count": len(video_paths),
                        "video_paths": video_paths,
                    }
                )
                continue
            local_videos = {}
            for half, remote_path in sorted(half_to_remote.items()):
                local_path = args.tmp_video_root / game_path.strip("/") / Path(remote_path).name
                local_videos[half] = download_video(dbx, shared_url, shared_password, remote_path, local_path)
            game_windows = []
            for half, video_path in sorted(local_videos.items()):
                half_events = [event for event in events if event.half == half]
                if not half_events:
                    continue
                cap, fps, duration = open_video(video_path)
                try:
                    for event in half_events:
                        item = extract_event_frames(
                            cap=cap,
                            fps=fps,
                            duration=duration,
                            event=event,
                            game_label=game_label,
                            frame_root=frame_root,
                            before_sec=args.before_sec,
                            after_sec=args.after_sec,
                            sample_every_sec=args.sample_every_sec,
                            sampled_frames_per_window=args.sampled_frames_per_window,
                            image_size=args.resize_frames or None,
                            overwrite=args.force,
                        )
                        if item:
                            game_windows.append(item)
                finally:
                    cap.release()
            windows.extend(game_windows)
            games_done.append(
                {
                    "game_label": game_label,
                    "events": len(events),
                    "windows": len(game_windows),
                    "dropbox_game_path": game_path,
                    "video_count": len(video_paths),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "game_label": game_label,
                    "dropbox_game_path": game_path,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )
        finally:
            if not args.keep_videos and not args.dry_run:
                shutil.rmtree(args.tmp_video_root / game_path.strip("/"), ignore_errors=True)

    if not args.dry_run:
        write_jsonl(manifest_out, windows)
    statuses = {}
    if not args.dry_run and not args.extract_only and windows:
        statuses = run_existing_pipeline(args.config, shard_dir, args.device, args.force)
    if args.cleanup_frames and not args.dry_run:
        shutil.rmtree(frame_root, ignore_errors=True)

    summary = {
        "dataset_manifest": str(args.dataset_manifest),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "split": args.split,
        "selected_games": int(len(df)),
        "games_done": games_done,
        "failures": failures,
        "window_manifest": str(manifest_out),
        "windows": len(windows),
        "pipeline_statuses": statuses,
        "runtime_sec": round(time.time() - start, 3),
        "dry_run": args.dry_run,
        "extract_only": args.extract_only,
        "cleanup_frames": args.cleanup_frames,
    }
    write_json(extraction_summary_path, summary)
    print(json.dumps(summary, indent=2))
    if failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
