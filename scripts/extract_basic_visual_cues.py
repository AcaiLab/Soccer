import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline_utils import ensure_dir, load_config, read_jsonl, repo_root, resolve_path, write_jsonl


def load_rgb(path: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size)), dtype=np.uint8)


def green_field_ratio(rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Broad soccer-pitch green range. This is a heuristic, not segmentation.
    mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 35)
    return float(mask.mean())


def white_line_ratio(rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] <= 45) & (hsv[:, :, 2] >= 170)
    return float(mask.mean())


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float((edges > 0).mean())


def frame_features(rgb: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    top = rgb[: max(1, rgb.shape[0] // 5), :, :]
    top_gray = gray[: max(1, gray.shape[0] // 5), :]
    return {
        "green_ratio": green_field_ratio(rgb),
        "white_line_ratio": white_line_ratio(rgb),
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "edge_density": edge_density(gray),
        "top_band_green_ratio": green_field_ratio(top),
        "top_band_edge_density": edge_density(top_gray),
    }


def motion_features(frames: list[np.ndarray]) -> dict[str, float]:
    if len(frames) < 2:
        return {"motion_intensity": 0.0, "shot_change_rate": 0.0}
    diffs = []
    for a, b in zip(frames[:-1], frames[1:], strict=False):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        diffs.append(float(np.mean(np.abs(ga - gb))))
    diffs_arr = np.array(diffs, dtype=np.float32)
    return {
        "motion_intensity": float(diffs_arr.mean()),
        "shot_change_rate": float((diffs_arr > 0.22).mean()),
    }


def classify_shot_type(mean_green: float, mean_edge: float, top_green: float) -> str:
    if mean_green >= 0.45:
        return "wide_field_view"
    if mean_green >= 0.25:
        return "medium_field_view"
    if top_green < 0.08 and mean_edge > 0.12:
        return "graphic_or_closeup"
    return "closeup_or_nonfield_view"


def motion_bucket(value: float) -> str:
    if value >= 0.16:
        return "high"
    if value >= 0.08:
        return "medium"
    return "low"


def extract_window_cues(row: dict, image_size: int) -> dict[str, object]:
    frames = [load_rgb(path, image_size) for path in row["sampled_image_paths"]]
    per_frame = [frame_features(frame) for frame in frames]
    motion = motion_features(frames)
    mean = {
        key: float(np.mean([features[key] for features in per_frame]))
        for key in per_frame[0]
    }
    maxes = {
        f"{key}_max": float(np.max([features[key] for features in per_frame]))
        for key in per_frame[0]
    }
    shot_type = classify_shot_type(mean["green_ratio"], mean["edge_density"], mean["top_band_green_ratio"])
    likely_overlay = bool(mean["top_band_green_ratio"] < 0.15 and mean["top_band_edge_density"] > 0.08)
    field_visible = mean["green_ratio"] > 0.25
    return {
        "window_id": row["window_id"],
        "game_label": row["game_label"],
        "event_label": row["event_label"],
        "visual_facts": {
            "camera": {
                "shot_type": shot_type,
                "field_visible": field_visible,
                "field_visible_ratio": round(mean["green_ratio"], 4),
                "likely_graphic_overlay": likely_overlay,
                "likely_replay": None,
            },
            "scene": {
                "motion_intensity": motion_bucket(motion["motion_intensity"]),
                "motion_intensity_score": round(motion["motion_intensity"], 4),
                "shot_change_rate": round(motion["shot_change_rate"], 4),
                "white_line_ratio": round(mean["white_line_ratio"], 4),
                "brightness": round(mean["brightness"], 4),
                "contrast": round(mean["contrast"], 4),
                "edge_density": round(mean["edge_density"], 4),
            },
            "objects": {
                "ball_visible": None,
                "referee_visible": None,
                "card_visible": None,
                "substitution_board_visible": None,
                "player_count_estimate": None,
            },
            "field": {
                "zone": "unknown",
                "confidence": 0.0,
            },
            "raw_basic_cues": {**mean, **maxes, **motion},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    base = repo_root()
    output_root = resolve_path(cfg["project"]["output_root"], base) or base
    manifest = args.manifest or output_root / "manifests" / "local_windows.jsonl"
    out = args.out or output_root / "features" / "basic_visual_cues.jsonl"
    rows = read_jsonl(manifest)
    if args.limit:
        rows = rows[: args.limit]
    output = [extract_window_cues(row, args.image_size) for row in rows]
    ensure_dir(out.parent)
    write_jsonl(out, output)
    print(json.dumps({"output": str(out), "windows": len(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
