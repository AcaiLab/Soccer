import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON_BASE = Path("data/videos")
total = {}

for folder in sorted(JSON_BASE.iterdir()):
    if not folder.is_dir():
        continue
    jsons = list(folder.glob("*.json"))
    if not jsons:
        continue
    with open(jsons[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    for c in data.get("comments", []):
        etype = c.get("comments_type", "unknown")
        if not etype:
            etype = "unknown"
        total[etype] = total.get(etype, 0) + 1

labels = sorted(total.keys(), key=lambda x: total[x], reverse=True)
counts = [total[l] for l in labels]

print(f"{'Event Type':<30s}  {'Count':>5s}")
print("-" * 40)
for l, c in zip(labels, counts):
    print(f"{l:<30s}  {c:>5d}")
print("-" * 40)
print(f"{'TOTAL':<30s}  {sum(counts):>5d}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(len(labels)), counts, color="#3498db")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
ax.set_ylabel("# Events")
ax.set_title("Event Distribution by Label - 10 EPL Games (2021-2022)", fontweight="bold")
for i, v in enumerate(counts):
    ax.text(i, v + 1, str(v), ha="center", fontsize=8, fontweight="bold")
fig.tight_layout()
fig.savefig("preprocess/event_distribution.png", dpi=150)
print("\nSaved preprocess/event_distribution.png")
