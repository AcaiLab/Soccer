import json, re, pathlib, cv2

video_path = "/Users/yuntingyin/Documents/Research/Soccer/watford-fc-liverpool-fc_1.mkv"
json_path  = "/Users/yuntingyin/Documents/Research/Soccer/2017-08-12_Watford_3-3_Liverpool_AaZvBO5T.json"

base_out   = pathlib.Path("extracts_first_half")
clips_dir  = base_out / "clips"
frames_dir = base_out / "frames"
clips_dir.mkdir(parents=True, exist_ok=True)
frames_dir.mkdir(parents=True, exist_ok=True)

CLIP_BEFORE_SEC = 15     # seconds before event
CLIP_AFTER_SEC  = 15     # seconds after event
FRAME_SAMPLING_EVERY_SEC = 1  # save one frame every 1 second

# file name
def norm_ts_for_filename(ts: str) -> str:
    sec = parse_time_stamp(ts)
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}-{s:06.3f}"

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

def seek_frame(cap, frame_idx: int) -> bool:
    return cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

raw_ts = [c.get("time_stamp", "") for c in data.get("comments", []) if c.get("time_stamp", "")]
seen, ts_list = set(), []
for ts in raw_ts:
    if ts not in seen:
        seen.add(ts)
        ts_list.append(ts)

if not ts_list:
    print("No timestamps found.")
    raise SystemExit

# read video
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

def write_clip_and_frames(center_sec: float, before_sec: float, after_sec: float, tag: str):
    start_sec = max(0.0, center_sec - before_sec)    
    end_sec   = min(duration_sec, center_sec + after_sec)

    start_f = int(round(start_sec * fps))
    end_f   = int(round(end_sec   * fps))
    if end_f <= start_f:
        print(f"Skipping {tag}: no valid range.")
        return

    clip_name = clips_dir / f"clip_{tag}_{start_sec:08.3f}s_to_{end_sec:08.3f}s.mp4"
    vout = cv2.VideoWriter(str(clip_name), fourcc, fps, (w, h))
    if not vout.isOpened():
        print(f"Warning: cannot open writer for {clip_name}")
        return

    seek_frame(cap, start_f)
    save_idxs = {int(round(t * fps)) for t in 
                 [start_sec + i for i in range(0, int(after_sec + before_sec)+1, FRAME_SAMPLING_EVERY_SEC)]}

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
    print(f"{tag}: wrote {wrote} frames to {clip_name.name}")

ts_list_sorted = sorted(ts_list, key=parse_time_stamp)
for ts in ts_list_sorted:
    try:
        center_sec = parse_time_stamp(ts)
    except ValueError as e:
        print(f"Skipping bad timestamp {ts}: {e}")
        continue
    tag = norm_ts_for_filename(ts)
    write_clip_and_frames(center_sec, CLIP_BEFORE_SEC, CLIP_AFTER_SEC, tag)

cap.release()
print(f"Extracted Clips → {clips_dir}, Frames → {frames_dir}")
