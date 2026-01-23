from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path("extracts_first_half")
CLIPS_ROOT = BASE / "clips"
FRAMES_ROOT = BASE / "frames"

def count_files_in_subdirs(root: Path, exts=()):
    counts = {}
    if not root.exists():
        return counts

    for d in root.iterdir():
        if not d.is_dir():
            continue
        if exts:
            n = sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)
        else:
            n = sum(1 for p in d.iterdir() if p.is_file())
        counts[d.name] = n
    return counts

clip_counts = count_files_in_subdirs(CLIPS_ROOT, exts=(".mp4",))
frame_counts = count_files_in_subdirs(FRAMES_ROOT, exts=(".png", ".jpg", ".jpeg"))

labels = sorted(set(clip_counts) | set(frame_counts))
clips = [clip_counts.get(l, 0) for l in labels]
frames = [frame_counts.get(l, 0) for l in labels]

print("Label\tClips\tFrames")
for l, c, f in zip(labels, clips, frames):
    print(f"{l}\t{c}\t{f}")

# clips
plt.figure()
plt.bar(labels, clips)
plt.xticks(rotation=75, ha="right")
plt.ylabel("# Clips")
plt.title("Clips per label")
plt.tight_layout()
plt.show()

# frames
plt.figure()
plt.bar(labels, frames)
plt.xticks(rotation=75, ha="right")
plt.ylabel("# Frames")
plt.title("Frames per label")
plt.tight_layout()
plt.show()
