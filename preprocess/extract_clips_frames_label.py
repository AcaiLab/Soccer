import json, re, pathlib, cv2

match_folder = pathlib.Path("/Users/nathn/OneDrive/Desktop/AcaiLab/Soccer/10_matches/match_05")

video_half1 = next(match_folder.glob("*_1.mkv"))
video_half2 = next(match_folder.glob("*_2.mkv"))
json_path = next(match_folder.glob("*.json"))

base_out = pathlib.Path("extracts_frames_only_match_05")
# commented out
# clips_root = base_out / "clips"
frames_root = base_out / "frames"
# clips_root.mkdir(parents=True, exist_ok=True)
frames_root.mkdir(parents=True, exist_ok=True)


CLIP_BEFORE_SEC = 30
CLIP_AFTER_SEC = 30
FRAME_SAMPLING_EVERY_SEC = 1

REJECTED_GOAL_PATTERNS = [
    r"\bvar\b",
    r"\breview\b",
    r"\bgoal stands\b",
    r"\breferee\b",
    r"\bcentre spot\b",
    r"\bno infringement\b",
    r"\bverdict\b",
    r"\boffside\b",
]

CONFIRMED_GOAL_PATTERNS = [
    r"\bgo+a+l(?:\s|-|!|:|\.)",
    r"\bwhat a phenomenal goal\b",
    r"\ba superb goal\b",
    r"\bfantastic team goal\b",
    r"\bit'?s a goal!",
    r"\bit'?s in the back of the net!",
    r"\bthe score is(?: now)? \d+:\d+\b",
    r"\bit'?s(?: now)? \d+:\d+\b",
    r"\bmakes? it \d+:\d+\b",
    r"\bto make (?:it|the score) \d+:\d+\b",
    r"\bequali[sz]es\b",
    r"\blevels? the score\b",
    r"\bputs [^.]* ahead\b",
    r"\bnets\b",
    r"\bscores?\b",
    r"\bfire(?:s)? home\b",
    r"\bslams? home\b",
    r"\b(?:slots?|drills?|powers?|steers?|plants?|sends?|lashes?|pokes?|rolls?)\b[^.]{0,120}\b(?:into|inside)\b",
    r"\bslotting the ball past the goalkeeper\b",
    r"\binto the back of the net\b",
    r"\binto the open net\b",
    r"\binto the net\b",
    r"\binto the roof of the net\b",
    r"\binto the middle of the target\b",
    r"\binto the (?:top|bottom|left|right|middle) (?:corner|side of the net|side of the goal)\b",
    r"\binside the (?:top|bottom|left|right|middle) (?:corner|side of the net|side of the goal|left post|right post)\b",
    r"\bgoes into the (?:middle|left|right) of the net\b",
    r"\bwent into the (?:bottom right corner|bottom left corner|top right corner|top left corner)\b",
    r"\bof his own goal\b",
    r"\bbeats? the (?:goalkeeper|keeper)\b",
]


def parse_time_stamp(mm_ss: str) -> float:
    m = re.fullmatch(r'(\d+)(?:\+(\d+))?:(\d{2})', mm_ss.strip())
    mm = int(m.group(1))
    added = int(m.group(2)) if m.group(2) else 0
    ss = int(m.group(3))
    return (mm + added) * 60 + ss


def norm_ts_for_filename(ts: str) -> str:
    sec = parse_time_stamp(ts)
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}-{s:06.3f}"


def sanitize_event_name(s: str) -> str:
    s = (s or "unknown").strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "unknown"


def is_confirmed_goal_comment(event_name: str, comment_text: str) -> bool:
    if event_name != "goal":
        return True

    text = (comment_text or "").strip().lower().replace("’", "'")
    if not text:
        return False

    if any(re.search(pattern, text) for pattern in REJECTED_GOAL_PATTERNS):
        return False

    return any(re.search(pattern, text) for pattern in CONFIRMED_GOAL_PATTERNS)


with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

pairs = []
seen = set()

for c in data.get("comments", []):
    ts = c.get("time_stamp", "")
    half = c.get("half", None)
    event = c.get("comments_type", "unknown")
    comment_text = c.get("comments_text", "")

    if not ts or half not in [1, 2]:
        continue

    if not is_confirmed_goal_comment(event, comment_text):
        continue

    key = (half, ts, event)
    if key in seen:
        continue
    seen.add(key)

    pairs.append((ts, event, half))

pairs.sort(key=lambda x: parse_time_stamp(x[0]))

# Open both videos
cap1 = cv2.VideoCapture(str(video_half1))
cap2 = cv2.VideoCapture(str(video_half2))

if not cap1.isOpened() or not cap2.isOpened():
    raise RuntimeError("Could not open one of the half videos.")

fps1 = cap1.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)

duration1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT) / fps1
duration2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT) / fps2

w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")


def write_clip_and_frames(cap, fps, duration_sec, center_sec, tag_ts, event_name):
    event_dir = sanitize_event_name(event_name)
    #commented out
    # clips_dir = clips_root / event_dir
    frames_dir = frames_root / event_dir
    # clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    start_sec = max(0.0, center_sec - CLIP_BEFORE_SEC)
    end_sec = min(duration_sec, center_sec + CLIP_AFTER_SEC)

    start_f = int(round(start_sec * fps))
    end_f = int(round(end_sec * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    # commented out 
    # clip_name = clips_dir / f"{event_dir}_clip_{tag_ts}.mp4"
    # vout = cv2.VideoWriter(str(clip_name), fourcc, fps, (w, h))

    idx = start_f
    save_idxs = {
        int(round((start_sec + t) * fps))
        for t in range(0, int(end_sec - start_sec) + 1, FRAME_SAMPLING_EVERY_SEC)
    }

    while idx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        # commented out 
        # vout.write(frame)

        if idx in save_idxs:
            frames_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frames_dir / f"{event_dir}_frame_{tag_ts}_{idx}.png"), frame)

        idx += 1
    # commented out 
    # vout.release()
    print(f"{event_dir}@{tag_ts} extracted")


for ts, event, half in pairs:
    center_sec = parse_time_stamp(ts)
    tag_ts = norm_ts_for_filename(ts)

    if half == 1:
        write_clip_and_frames(cap1, fps1, duration1, center_sec, tag_ts, event)
    else:
        write_clip_and_frames(cap2, fps2, duration2, center_sec, tag_ts, event)

cap1.release()
cap2.release()

print("Extraction complete for both halves.")
