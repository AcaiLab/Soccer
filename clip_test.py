from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("test.png").convert("RGB")

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
    pooled_output = vision_outputs.pooler_output
    image_features = model.visual_projection(pooled_output)

print(image_features.shape)