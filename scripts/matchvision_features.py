"""
Encode events with MatchVision (UniSoccer, Rao et al., CVPR 2025) instead of DINOv2.

MatchVision is SigLIP-base with temporal attention, trained on soccer clips of
30 frames at 1 fps (15 s either side of an annotated moment), 224x224. This
script feeds it the same 30 seconds from the frames batch_pipeline.py already
extracted (offsets -15 ... +14 s, nearest extracted frame where one is missing)
and saves one 768-d vector per frame in the clip_temporal_padded.npz format, so
cluster_analysis.py, cluster_animation.py and the classifiers read it unchanged.

MatchVision was trained on the SoccerReplay-1988 train split (validation was
used to pick checkpoints). Evaluate it on official test matches only; on train
matches its features have already seen the labels.

Needs the UniSoccer code (github.com/jyrao/UniSoccer), its checkpoint
pretrained_classification.pth (huggingface.co/Homie0609/UniSoccer), the
google/siglip-base-patch16-224 weights, and timm + einops.

Usage:
    python scripts/matchvision_features.py --unisoccer ~/ext/UniSoccer \
        --checkpoint ~/ext/weights/pretrained_classification.pth \
        --siglip <path to siglip-base-patch16-224> --splits test valid --out features/matchvision
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from soccer.annotations import tag_to_seconds
from soccer.config import data_path, load_config, repo_path
from soccer.features import TEMPORAL_FILE
from soccer.frames import scan_events
from soccer.manifest import load_manifest

OFFSETS = np.arange(-15, 15)  # 30 one-second frames, as MatchVision was trained
# the 23 classes of UniSoccer's config/pretrain_classification.py, in its order
KEYWORDS = ["var", "end of half game", "clearance", "second yellow card", "injury", "ball possession", "throw in",
            "show added time", "shot off target", "start of half game", "substitution", "saved by goal-keeper",
            "red card", "lead to corner", "ball out of play", "off side", "goal", "penalty", "yellow card",
            "foul lead to penalty", "corner", "free kick", "foul with no card"]


def pick_frames(frames: list, t0: float) -> tuple[list, int]:
    """For each offset in OFFSETS, the extracted frame nearest t0 + offset; and how many were within 0.5 s."""
    times = np.array([t for t, _ in frames])
    paths, exact = [], 0
    for off in OFFSETS:
        j = int(np.abs(times - (t0 + off)).argmin())
        exact += abs(times[j] - (t0 + off)) <= 0.5
        paths.append(frames[j][1])
    return paths, exact


class Clips(Dataset):
    def __init__(self, items, processor):
        self.items, self.processor = items, processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        images = [Image.open(p).convert("RGB") for p in self.items[i]["paths"]]
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]  # (T, C, H, W)
        return pixels.permute(1, 0, 2, 3), i  # (C, T, H, W), as MatchVision expects


def load_model(unisoccer: str, checkpoint: str, siglip: str, device):
    sys.path.insert(0, str(pathlib.Path(unisoccer).expanduser()))
    from model.MatchVision_classifier import MatchVision_Classifier

    model = MatchVision_Classifier(keywords=KEYWORDS, model_name=siglip)
    state = torch.load(pathlib.Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
    state = state.get("state_dict", state.get("model", state)) if isinstance(state, dict) else state
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    encoder_missing = [k for k in missing if k.startswith("siglip_model.")]
    if encoder_missing:
        raise RuntimeError(f"checkpoint lacks {len(encoder_missing)} MatchVision encoder weights, e.g. {encoder_missing[:3]}")
    print(f"loaded {checkpoint}: {len(missing)} missing (none in the encoder), {len(unexpected)} unexpected keys")
    return model.to(device).eval()


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Per-frame MatchVision features for each event.")
    parser.add_argument("--unisoccer", required=True, help="clone of github.com/jyrao/UniSoccer")
    parser.add_argument("--checkpoint", required=True, help="pretrained_classification.pth")
    parser.add_argument("--siglip", required=True, help="google/siglip-base-patch16-224 (name or local path)")
    parser.add_argument("--splits", nargs="+", default=["test"], help="official splits to encode (default: test)")
    parser.add_argument("--out", required=True, help="directory to write clip_temporal_padded.npz")
    parser.add_argument("--batch-size", type=int, default=8, help="clips per batch")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    from transformers import SiglipImageProcessor

    games = {g.game_num: g for g in load_manifest(repo_path(cfg, "manifest"))
             if g.official_split in args.splits}
    events = {k: v for k, v in scan_events(data_path(cfg, "frames")).items() if k[0] in games}
    keys = sorted(events)
    items, short = [], 0
    for k in keys:
        paths, exact = pick_frames(sorted(events[k]), tag_to_seconds(k[3]))
        short += exact < len(OFFSETS)
        items.append({"key": k, "paths": paths})
    print(f"{len(games)} matches ({'/'.join(args.splits)}), {len(items)} events; "
          f"{short} events lack some of the 30 seconds (nearest frame repeated)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.unisoccer, args.checkpoint, args.siglip, device)
    processor = SiglipImageProcessor.from_pretrained(args.siglip)  # images only; no tokenizer needed
    loader = DataLoader(Clips(items, processor), batch_size=args.batch_size, num_workers=args.workers)

    feats = np.zeros((len(items), len(OFFSETS), 768), dtype=np.float32)
    done = 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for pixels, idx in loader:
            out = model.siglip_model(pixels.to(device))  # (B, 30, 768)
            feats[idx.numpy()] = out.float().cpu().numpy()
            done += len(idx)
            if done % (args.batch_size * 50) < args.batch_size:
                print(f"  {done}/{len(items)} events", flush=True)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    g, h, lab, tag = (np.array(c) for c in zip(*keys, strict=True))
    np.savez_compressed(out_dir / TEMPORAL_FILE, features=feats, masks=np.ones(feats.shape[:2], bool),
                        labels=lab, games=g, halves=h, event_tags=tag)
    print(f"saved {out_dir / TEMPORAL_FILE}  features {feats.shape}")


if __name__ == "__main__":
    main()
