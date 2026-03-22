from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from pathlib import Path
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

BASE = Path("extract_event_frames_only/extracts_event_frames_only_match_20/frames")  
OUT = Path("event_clip_features/event_clip_features_match20")
OUT.mkdir(exist_ok=True)

for label_dir in BASE.iterdir():
    if not label_dir.is_dir():
        continue

    out_label_dir = OUT / label_dir.name
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for event_dir in label_dir.iterdir():
        if not event_dir.is_dir():
            continue

        features = []
        frame_files = sorted(event_dir.glob("*.png"))

        for img_path in frame_files:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
                pooled_output = vision_outputs.pooler_output
                image_features = model.visual_projection(pooled_output)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            features.append(image_features.squeeze().cpu().numpy())

        if features:
            features = np.array(features)   # shape: (num_frames, 512)
            np.save(out_label_dir / f"{event_dir.name}.npy", features)
            print(f"Saved {label_dir.name}/{event_dir.name} with shape {features.shape}")

print("Event-level CLIP feature extraction complete.")