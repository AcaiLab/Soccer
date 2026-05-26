"""
Unified encoder interface for vision feature extraction.

Supports:
  - CLIP   (openai/clip-vit-*)   -> 512-dim projected embeddings
  - DINOv2 (facebook/dinov2-*)   -> 768-dim CLS token embeddings
  - Any HuggingFace AutoModel    -> last_hidden_state CLS token

Factory
-------
    from model_encoders import build_encoder
    encoder = build_encoder("openai/clip-vit-base-patch32", device="cpu")
    feats   = encoder.encode_batch(list_of_pil_images)  # (N, 512) float32
    dim     = encoder.embed_dim                          # 512

    encoder = build_encoder("facebook/dinov2-base", device="cpu")
    feats   = encoder.encode_batch(list_of_pil_images)  # (N, 768) float32
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEncoder(abc.ABC):
    """Interface every encoder must satisfy."""

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the output embedding vectors."""

    @abc.abstractmethod
    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode a list of RGB PIL Images.

        Returns
        -------
        np.ndarray
            Shape (N, embed_dim), dtype float32, each row L2-normalized.
            Returns shape (0, embed_dim) for an empty input list.
        """


# ---------------------------------------------------------------------------
# CLIP encoder
# ---------------------------------------------------------------------------

class CLIPEncoder(BaseEncoder):
    """
    Wraps HuggingFace CLIPModel + CLIPProcessor.

    Inference:  vision_model -> pooler_output -> visual_projection -> L2-norm
    """

    def __init__(self, model_name: str, device: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(device)
        self._model.eval()

        # Probe embed_dim with a single dummy forward pass (no real image needed)
        dummy = Image.new("RGB", (224, 224))
        inp = self._processor(images=[dummy], return_tensors="pt")
        with torch.no_grad():
            vo = self._model.vision_model(
                pixel_values=inp["pixel_values"].to(device)
            )
            feat = self._model.visual_projection(vo.pooler_output)
        self._embed_dim: int = int(feat.shape[-1])

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self._embed_dim), dtype=np.float32)

        inputs = self._processor(images=images, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            vo    = self._model.vision_model(pixel_values=pv)
            feats = self._model.visual_projection(vo.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# DINOv2 (generic AutoModel) encoder
# ---------------------------------------------------------------------------

class DINOv2Encoder(BaseEncoder):
    """
    Wraps HuggingFace AutoModel + AutoImageProcessor.

    Uses the CLS token (index 0 of last_hidden_state) as the image embedding,
    which is the standard DINOv2 inference pattern.

    Works for any ViT-style model loadable via AutoModel, e.g.:
        facebook/dinov2-small   -> 384-dim
        facebook/dinov2-base    -> 768-dim
        facebook/dinov2-large   -> 1024-dim
    """

    def __init__(self, model_name: str, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self._model.eval()

        # Probe embed_dim
        dummy = Image.new("RGB", (224, 224))
        inputs = self._processor(images=[dummy], return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(
                **{k: v.to(device) for k, v in inputs.items()}
            )
            feat = outputs.last_hidden_state[:, 0, :]
        self._embed_dim: int = int(feat.shape[-1])

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self._embed_dim), dtype=np.float32)

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            feats   = outputs.last_hidden_state[:, 0, :]   # CLS token
            feats   = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_encoder(model_name: str, device: str) -> BaseEncoder:
    """
    Return the correct encoder based on the model name string.

        "clip" in model_name.lower()  ->  CLIPEncoder
        anything else                 ->  DINOv2Encoder

    Examples
    --------
        build_encoder("openai/clip-vit-base-patch32", "cpu")  -> CLIPEncoder (512-dim)
        build_encoder("facebook/dinov2-base", "cpu")          -> DINOv2Encoder (768-dim)
        build_encoder("facebook/dinov2-large", "cuda")        -> DINOv2Encoder (1024-dim)
    """
    if "clip" in model_name.lower():
        print(f"[build_encoder] CLIP model    -> CLIPEncoder  ({model_name})")
        return CLIPEncoder(model_name, device)
    else:
        print(f"[build_encoder] Non-CLIP model -> DINOv2Encoder ({model_name})")
        return DINOv2Encoder(model_name, device)
