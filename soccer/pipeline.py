"""Per-game steps shared by the download and extraction scripts."""

from __future__ import annotations

import json
import pathlib
import shutil

from soccer.annotations import events_for_half, find_annotation_file, load_comments
from soccer.dropbox import DropboxSharedLink
from soccer.extract import ExtractionSettings, extract_events
from soccer.manifest import Game

DONE_MARKER = "extraction.json"


def half_dir(frames_root: pathlib.Path, game: Game, half: int) -> pathlib.Path:
    return frames_root / f"game_{game.game_num}_half_{half}"


def is_extracted(frames_root: pathlib.Path, game: Game) -> bool:
    """Both halves finished. A half counts only once its marker is written, so an
    interrupted extraction is redone instead of being mistaken for complete."""
    return all((half_dir(frames_root, game, h) / DONE_MARKER).is_file() for h in (1, 2))


def annotation_for(annotations_root: pathlib.Path, game: Game) -> pathlib.Path:
    found = find_annotation_file(annotations_root, game.league, game.match_id)
    if found is None:
        raise FileNotFoundError(f"No annotation JSON for {game.league}/{game.match_id} under {annotations_root}")
    path, split = found
    if split != game.official_split:
        raise ValueError(f"{game.match_id} is in the official {split!r} split, "
                         f"but the manifest says {game.official_split!r}")
    return path


def download_videos(client: DropboxSharedLink, game: Game, videos_root: pathlib.Path) -> pathlib.Path:
    """Download the game's .mkv halves into videos_root/<match_id>/."""
    out = videos_root / game.match_id
    entries = client.list_folder(f"/{game.league}/{game.match_id}")
    mkvs = [e for e in entries if e[".tag"] == "file" and e["name"].endswith(".mkv")]
    if not mkvs:
        raise FileNotFoundError(f"No .mkv files in /{game.league}/{game.match_id}")
    for e in mkvs:
        n = client.download(f"/{game.league}/{game.match_id}/{e['name']}", out / e["name"], e["size"])
        print(f"    {e['name']}  {'already present' if n == 0 else f'{n / 1e9:.2f} GB'}", flush=True)
    return out


def extract_game(game: Game, video_dir: pathlib.Path, annotation: pathlib.Path,
                 frames_root: pathlib.Path, settings: ExtractionSettings) -> dict:
    """Extract both halves and write a marker recording what was done."""
    comments = load_comments(annotation)
    summary = {}
    for half in (1, 2):
        videos = sorted(video_dir.glob(f"*_{half}.mkv"))
        if not videos:
            raise FileNotFoundError(f"No *_{half}.mkv in {video_dir}")
        out = half_dir(frames_root, game, half)
        if (out / DONE_MARKER).is_file():
            summary[half] = json.loads((out / DONE_MARKER).read_text())
            continue
        if out.exists():
            shutil.rmtree(out)  # partial output from an interrupted run
        events = events_for_half(comments, half)
        stats = extract_events(videos[0], events, out, settings)
        summary[half] = {
            "match_id": game.match_id, "half": half, "video": videos[0].name,
            "annotation": annotation.name, "events": stats.events, "frames": stats.frames,
            "skipped_outside_video": stats.skipped, "settings": settings.__dict__,
        }
        (out / DONE_MARKER).write_text(json.dumps(summary[half], indent=2))
        print(f"    half {half}: {stats.events} events, {stats.frames} frames", flush=True)
    return summary
