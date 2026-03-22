from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

BASE = Path("extracts_frames_only_match_20/frames")
OUT = Path("clip_features_match20")

OUT.mkdir(exist_ok=True)

for event_folder in BASE.iterdir():
    if not event_folder.is_dir():
        continue

    print(f"Processing {event_folder.name}")

    features = []

    for img_path in tqdm(list(event_folder.glob("*.png"))):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_output = vision_outputs.pooler_output
            image_features = model.visual_projection(pooled_output)

        features.append(image_features.squeeze().numpy())

    np.save(OUT / f"{event_folder.name}.npy", np.array(features))

print("Feature extraction complete.")