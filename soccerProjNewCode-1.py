import json, re, pathlib, cv2
import argparse

# ----------------------------
# Command line input
# ----------------------------
parser = argparse.ArgumentParser(description="Extract clips and frames from multiple soccer matches.")
parser.add_argument("--project-folder", required=True, help="Path to the main project folder containing match subfolders")
parser.add_argument("--out", default="extracts_output", help="Output folder name")

args = parser.parse_args()

project_folder = pathlib.Path(args.project_folder)
base_out = pathlib.Path(args.out)

CLIP_BEFORE_SEC = 15
CLIP_AFTER_SEC = 15
FRAME_SAMPLING_EVERY_SEC = 1

# ----------------------------
# Timestamp helpers
# ----------------------------
def parse_time_stamp(mm_ss: str) -> float:
    if not mm_ss:
        raise ValueError("Empty timestamp")
    m = re.fullmatch(r'(\d+)(?:\+(\d+))?:(\d{2})', mm_ss.strip())
    if not m:
        raise ValueError(f"Unrecognized time format: {mm_ss!r}")
    mm = int(m.group(1))
    added = int(m.group(2)) if m.group(2) else 0
    ss = int(m.group(3))
    return (mm + added) * 60 + ss

def norm_ts_for_filename(ts: str) -> str:
    sec = parse_time_stamp(ts)
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}-{s:06.3f}"

def seek_frame(cap, frame_idx: int) -> bool:
    return cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

# ----------------------------
# Main extraction function
# ----------------------------
def write_clip_and_frames(cap, fps, duration_sec, w, h, center_sec, before_sec, after_sec, tag, clips_dir, frames_dir):
    start_sec = max(0.0, center_sec - before_sec)
    end_sec = min(duration_sec, center_sec + after_sec)

    start_f = int(round(start_sec * fps))
    end_f = int(round(end_sec * fps))
    if end_f <= start_f:
        print(f"  Skipping {tag}: no valid range.")
        return

    clip_name = clips_dir / f"clip_{tag}_{start_sec:08.3f}s_to_{end_sec:08.3f}s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vout = cv2.VideoWriter(str(clip_name), fourcc, fps, (w, h))
    if not vout.isOpened():
        print(f"  Warning: cannot open writer for {clip_name}")
        return

    seek_frame(cap, start_f)

    total_range = int(after_sec + before_sec)
    save_idxs = {int(round((start_sec + i) * fps))
                 for i in range(0, total_range + 1, FRAME_SAMPLING_EVERY_SEC)}

    wrote = 0
    idx = start_f

    while idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break

        vout.write(frame)

        if idx in save_idxs:
            t_abs = idx / fps
            m = int(t_abs // 60)
            s = t_abs - 60 * m
            label = f"{m:02d}-{s:06.3f}"
            out_png = frames_dir / f"frame_{tag}_t{label}.png"
            cv2.imwrite(str(out_png), frame)

        wrote += 1
        idx += 1

    vout.release()
    print(f"  {tag}: wrote {wrote} frames to {clip_name.name}")

# ----------------------------
# Process a single match
# ----------------------------
def process_match(match_folder: pathlib.Path, output_base: pathlib.Path):
    print(f"\n{'='*60}")
    print(f"Processing match: {match_folder.name}")
    print(f"{'='*60}")
    
    # Find JSON file
    json_files = list(match_folder.glob("*.json"))
    if len(json_files) == 0:
        print(f"ERROR: No JSON file found in {match_folder.name}")
        return
    if len(json_files) > 1:
        print(f"WARNING: Multiple JSON files found in {match_folder.name}, using first one")
    json_path = json_files[0]
    
    # Find video files
    video_half1 = list(match_folder.glob("*_1.mkv"))
    video_half2 = list(match_folder.glob("*_2.mkv"))
    
    if len(video_half1) == 0:
        print(f"ERROR: No first half video (*_1.mkv) found in {match_folder.name}")
        return
    if len(video_half2) == 0:
        print(f"ERROR: No second half video (*_2.mkv) found in {match_folder.name}")
        return
    
    video_half1_path = video_half1[0]
    video_half2_path = video_half2[0]
    
    print(f"JSON: {json_path.name}")
    print(f"First Half Video: {video_half1_path.name}")
    print(f"Second Half Video: {video_half2_path.name}")
    
    # Load JSON timestamps
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Separate timestamps by half
    timestamps_half1 = []
    timestamps_half2 = []
    
    for comment in data.get("comments", []):
        ts = comment.get("time_stamp", "")
        half = comment.get("half", 0)
        
        if ts:  # Only process if timestamp exists
            if half == 1:
                timestamps_half1.append(ts)
            elif half == 2:
                timestamps_half2.append(ts)
    
    # Remove duplicates while preserving order
    seen = set()
    ts_half1_unique = []
    for ts in timestamps_half1:
        if ts not in seen:
            seen.add(ts)
            ts_half1_unique.append(ts)
    
    seen = set()
    ts_half2_unique = []
    for ts in timestamps_half2:
        if ts not in seen:
            seen.add(ts)
            ts_half2_unique.append(ts)
    
    print(f"Found {len(ts_half1_unique)} unique timestamps for first half")
    print(f"Found {len(ts_half2_unique)} unique timestamps for second half")
    
    # Create output directories
    match_out = output_base / match_folder.name
    half1_out = match_out / "first_half"
    half2_out = match_out / "second_half"
    
    half1_clips = half1_out / "clips"
    half1_frames = half1_out / "frames"
    half2_clips = half2_out / "clips"
    half2_frames = half2_out / "frames"
    
    half1_clips.mkdir(parents=True, exist_ok=True)
    half1_frames.mkdir(parents=True, exist_ok=True)
    half2_clips.mkdir(parents=True, exist_ok=True)
    half2_frames.mkdir(parents=True, exist_ok=True)
    
    # Process first half
    if ts_half1_unique:
        print(f"\n--- Processing First Half ---")
        process_half(video_half1_path, ts_half1_unique, half1_clips, half1_frames)
    
    # Process second half
    if ts_half2_unique:
        print(f"\n--- Processing Second Half ---")
        process_half(video_half2_path, ts_half2_unique, half2_clips, half2_frames)
    
    print(f"\nCompleted processing: {match_folder.name}")

# ----------------------------
# Process a single half
# ----------------------------
def process_half(video_path: pathlib.Path, timestamps: list, clips_dir: pathlib.Path, frames_dir: pathlib.Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"FPS: {fps:.3f}, Frames: {total_frames}, Duration: {duration_sec:.2f}s")
    
    # Sort timestamps
    ts_sorted = sorted(timestamps, key=parse_time_stamp)
    
    for ts in ts_sorted:
        try:
            center_sec = parse_time_stamp(ts)
        except ValueError as e:
            print(f"  Skipping bad timestamp {ts}: {e}")
            continue
        
        tag = norm_ts_for_filename(ts)
        write_clip_and_frames(cap, fps, duration_sec, w, h, center_sec, 
                            CLIP_BEFORE_SEC, CLIP_AFTER_SEC, tag, clips_dir, frames_dir)
    
    cap.release()
    print(f"Extracted clips → {clips_dir.name}, frames → {frames_dir.name}")

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    if not project_folder.exists():
        print(f"ERROR: Project folder does not exist: {project_folder}")
        exit(1)
    
    # Get all subdirectories (each should be a match)
    match_folders = [f for f in project_folder.iterdir() if f.is_dir()]
    
    if not match_folders:
        print(f"ERROR: No match folders found in {project_folder}")
        exit(1)
    
    print(f"Found {len(match_folders)} match folders to process")
    
    # Process each match
    for match_folder in sorted(match_folders):
        try:
            process_match(match_folder, base_out)
        except Exception as e:
            print(f"\nERROR processing {match_folder.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"All matches processed!")
    print(f"Output saved to: {base_out}")
    print(f"{'='*60}")