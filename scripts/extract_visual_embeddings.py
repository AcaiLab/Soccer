import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from pipeline_utils import choose_device, ensure_dir, load_config, read_jsonl, repo_root, resolve_path, seed_everything, write_json


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as image:
            return self.transform(image.convert("RGB")), idx


def make_model(model_name: str) -> tuple[nn.Module, str, int]:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = nn.Identity()
        return model, "imagenet", 576
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        return model, "imagenet", 512
    raise ValueError(f"Unsupported visual model: {model_name}")


def unique_sampled_paths(windows: list[dict]) -> list[str]:
    paths = []
    seen = set()
    for row in windows:
        for path in row["sampled_image_paths"]:
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def extract_embeddings(
    image_paths: list[str],
    model_name: str,
    image_size: int,
    batch_size: int,
    device: torch.device,
    dtype: str,
) -> tuple[np.ndarray, dict[str, object]]:
    model, weight_status, dim = make_model(model_name)
    model = model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    loader = DataLoader(ImagePathDataset(image_paths, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = []
    with torch.no_grad():
        for step, (images, _) in enumerate(loader, start=1):
            images = images.to(device)
            features = model(images)
            if features.ndim > 2:
                features = torch.flatten(features, 1)
            outputs.append(features.cpu().numpy())
            if step % 20 == 0:
                print(f"embedded {min(step * batch_size, len(image_paths))}/{len(image_paths)} frames")
    arr = np.vstack(outputs)
    if dtype == "float16":
        arr = arr.astype(np.float16)
    else:
        arr = arr.astype(np.float32)
    return arr, {"model": model_name, "weights": weight_status, "dim": dim, "dtype": str(arr.dtype)}


def build_window_tensor(windows: list[dict], image_paths: list[str], frame_embeddings: np.ndarray) -> np.ndarray:
    row_by_path = {path: i for i, path in enumerate(image_paths)}
    sequences = []
    for row in windows:
        idx = [row_by_path[path] for path in row["sampled_image_paths"]]
        sequences.append(frame_embeddings[idx])
    return np.stack(sequences, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    base = repo_root()
    output_root = resolve_path(cfg["project"]["output_root"], base) or base
    features_dir = ensure_dir(output_root / "features")
    manifest = args.manifest or output_root / "manifests" / "local_windows.jsonl"
    out = args.out or features_dir / f"visual_{cfg['embeddings']['visual_model']}_local.npz"
    device = choose_device(args.device)

    windows = read_jsonl(manifest)
    image_paths = unique_sampled_paths(windows)
    missing = [path for path in image_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} sampled frame paths; first: {missing[0]}")

    print(f"device: {device}")
    print(f"windows: {len(windows)}; unique sampled frames: {len(image_paths)}")
    frame_embeddings, meta = extract_embeddings(
        image_paths=image_paths,
        model_name=cfg["embeddings"]["visual_model"],
        image_size=int(cfg["embeddings"].get("image_size", 160)),
        batch_size=int(cfg["embeddings"].get("batch_size", 128)),
        device=device,
        dtype=cfg["embeddings"].get("cache_dtype", "float16"),
    )
    window_embeddings = build_window_tensor(windows, image_paths, frame_embeddings)
    np.savez_compressed(
        out,
        window_embeddings=window_embeddings,
        frame_embeddings=frame_embeddings,
        window_ids=np.array([row["window_id"] for row in windows]),
        event_labels=np.array([row["event_label"] for row in windows]),
        game_labels=np.array([row["game_label"] for row in windows]),
        sampled_image_paths=np.array(image_paths),
    )
    meta.update(
        {
            "manifest": str(manifest),
            "output": str(out),
            "windows": len(windows),
            "unique_sampled_frames": len(image_paths),
            "window_embedding_shape": list(window_embeddings.shape),
        }
    )
    write_json(out.with_suffix(".metadata.json"), meta)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
