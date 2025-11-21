import json, re, pathlib, cv2

video_path = "/content/watford-fc-liverpool-fc_1.mp4"
json_path  = "/content/2017-08-12_Watford_3-3_Liverpool_AaZvBO5T (2).json"

# create clips/ and frames/
base_out   = pathlib.Path("extracts_first_half")
clips_dir  = base_out / "clips"
frames_dir = base_out / "frames"
clips_dir.mkdir(parents=True, exist_ok=True)
frames_dir.mkdir(parents=True, exist_ok=True)

CLIP_BEFORE_SEC = 15
CLIP_AFTER_SEC  = 15
FRAME_SAMPLING_EVERY_SEC = 1  

# Parse timestamps: convert “mm:ss” or “mm+X:ss” to seconds
def parse_time_stamp(mm_ss: str) -> float:
    """Parse timestamps like '45:12' or '45+2:10' into seconds."""
    if not mm_ss:
        raise ValueError("Empty timestamp")
    m = re.fullmatch(r'(\d+)(?:\+(\d+))?:(\d{2})', mm_ss.strip())
    if not m:
        raise ValueError(f"Unrecognized time format: {mm_ss!r}")
    mm = int(m.group(1))
    added = int(m.group(2)) if m.group(2) else 0
    ss = int(m.group(3))
    return (mm + added) * 60 + ss

# Produce filenames

def norm_ts_for_filename(ts: str) -> str:
    """Convert 'mm:ss' into a filename-safe label like 12-30.000."""
    sec = parse_time_stamp(ts)
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}-{s:06.3f}"


# Create output folders with comment_type name

def get_output_dirs(comment_type: str):
    """
    Return (clip_subdir, frame_subdir) based on comment_type.
    Automatically creates the folders.
    """
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', comment_type.strip()) or "Unknown"

    clip_subdir  = clips_dir  / safe
    frame_subdir = frames_dir / safe

    clip_subdir.mkdir(parents=True, exist_ok=True)
    frame_subdir.mkdir(parents=True, exist_ok=True)

    return clip_subdir, frame_subdir


# Seeking helper for OpenCV
def seek_frame(cap, frame_idx: int) -> bool:
    """Seek to an absolute frame number in the video."""
    return cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ts_list = []  # list of (timestamp_str, comment_type)
seen = set()

for c in data.get("comments", []):
    ts = c.get("time_stamp", "")
    ctype = c.get("comments_type", "Unknown")

    if ts and ts not in seen:
        seen.add(ts)
        ts_list.append((ts, ctype))

if not ts_list:
    print("No timestamps found.")
    raise SystemExit

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps if fps > 0 else 0.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video FPS: {fps:.3f}, frames: {total_frames}, duration: {duration_sec:.2f}s")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# extract a clip and frames around time "center_sec"

def write_clip_and_frames(center_sec: float,
                          before_sec: float,
                          after_sec: float,
                          tag: str,
                          clip_out_dir,
                          frame_out_dir):

    # Compute time boundaries for the clip
    start_sec = max(0.0, center_sec - before_sec)
    end_sec   = min(duration_sec, center_sec + after_sec)

    start_f = int(round(start_sec * fps))
    end_f   = int(round(end_sec   * fps))

    if end_f <= start_f:
        print(f"Skipping {tag}: no valid frame range.")
        return

    # Output clip filename
    clip_path = clip_out_dir / f"clip_{tag}_{start_sec:08.3f}s_to_{end_sec:08.3f}s.mp4"

    # Video writer
    vout = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))
    if not vout.isOpened():
        print(f"Warning: cannot create clip file {clip_path}")
        return

    # Seek to the starting frame
    seek_frame(cap, start_f)

    # Determine frames
    total_clip_len = int(after_sec + before_sec)
    save_idxs = {
        int(round((start_sec + i) * fps))
        for i in range(0, total_clip_len + 1, FRAME_SAMPLING_EVERY_SEC)
    }

    wrote = 0
    idx = start_f

    # Read frames until we reach end_f
    while idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break

        # Write each frame into the MP4 clip
        vout.write(frame)

        # Save selected keyframes as PNG images
        if idx in save_idxs:
            t_abs = idx / fps
            m = int(t_abs // 60)
            s = t_abs - 60 * m
            label = f"{m:02d}-{s:06.3f}"
            png_path = frame_out_dir / f"frame_{tag}_t{label}.png"
            cv2.imwrite(str(png_path), frame)

        wrote += 1
        idx += 1

    vout.release()
    print(f"{tag}: saved {wrote} frames → {clip_path.name}")


# Sort timestamps by chronological order)

ts_list_sorted = sorted(ts_list, key=lambda x: parse_time_stamp(x[0]))

# extract clip + frames

for ts, ctype in ts_list_sorted:
    try:
        center_sec = parse_time_stamp(ts)
    except ValueError as e:
        print(f"Skipping bad timestamp {ts}: {e}")
        continue

    tag = norm_ts_for_filename(ts)

    # Create/locate the correct output subfolders for this comment_type
    clip_out_dir, frame_out_dir = get_output_dirs(ctype)

    # Extract video clip + frames
    write_clip_and_frames(center_sec,
                          CLIP_BEFORE_SEC,
                          CLIP_AFTER_SEC,
                          tag,
                          clip_out_dir,
                          frame_out_dir)

cap.release()

print(f"Done! Clips → {clips_dir}, Frames → {frames_dir}")
