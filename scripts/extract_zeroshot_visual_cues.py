import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline_utils import choose_device, ensure_dir, load_config, read_jsonl, repo_root, resolve_path, write_jsonl


LABEL_GROUPS = {
    "shot_type": [
        "wide broadcast view of a soccer field",
        "close up of a soccer player",
        "soccer replay shot",
        "statistics graphic on a soccer broadcast",
        "crowd or bench reaction shot",
    ],
    "objects": [
        "referee showing a card",
        "substitution board on the sideline",
        "goalkeeper making a save",
        "soccer players gathered around the referee",
        "players gathered in the penalty area",
        "corner kick setup",
        "free kick setup",
    ],
}


def load_open_clip(model_name: str, pretrained: str, device: torch.device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def score_labels(model, tokenizer, image_features: torch.Tensor, labels: list[str], device: torch.device) -> list[dict[str, float]]:
    prompts = [f"a photo of {label}" for label in labels]
    with torch.no_grad():
        text = tokenizer(prompts).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
    return [{"label": label, "score": float(score)} for label, score in sorted(zip(labels, sims, strict=True), key=lambda x: -x[1])]


def extract_for_window(row: dict, model, preprocess, tokenizer, device: torch.device, frames_per_window: int) -> dict[str, object]:
    paths = row["sampled_image_paths"]
    if frames_per_window > 0:
        paths = paths[:frames_per_window]
    group_scores: dict[str, list[list[dict[str, float]]]] = {name: [] for name in LABEL_GROUPS}
    with torch.no_grad():
        for path in paths:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for group, labels in LABEL_GROUPS.items():
                group_scores[group].append(score_labels(model, tokenizer, image_features, labels, device))

    aggregated = {}
    for group, frame_scores in group_scores.items():
        score_by_label = {label: [] for label in LABEL_GROUPS[group]}
        for scores in frame_scores:
            for item in scores:
                score_by_label[item["label"]].append(item["score"])
        ranked = sorted(
            [{"label": label, "score": float(np.mean(vals))} for label, vals in score_by_label.items()],
            key=lambda item: -item["score"],
        )
        aggregated[group] = ranked
    return {
        "window_id": row["window_id"],
        "game_label": row["game_label"],
        "event_label": row["event_label"],
        "zeroshot_visual_cues": aggregated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--frames-per-window", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    base = repo_root()
    output_root = resolve_path(cfg["project"]["output_root"], base) or base
    manifest = args.manifest or output_root / "manifests" / "local_windows.jsonl"
    out = args.out or output_root / "features" / "zeroshot_visual_cues.jsonl"
    device = choose_device(args.device)
    windows = read_jsonl(manifest)
    if args.limit:
        windows = windows[: args.limit]
    model, preprocess, tokenizer = load_open_clip(args.model_name, args.pretrained, device)
    rows = [
        extract_for_window(row, model, preprocess, tokenizer, device, args.frames_per_window)
        for row in windows
    ]
    ensure_dir(out.parent)
    write_jsonl(out, rows)
    print(json.dumps({"output": str(out), "windows": len(rows), "device": str(device)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
