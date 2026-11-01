from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image, ImageOps

CLASSES = ("laptop", "book", "other")


def l2_normalize(values: np.ndarray) -> np.ndarray:
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def majority_vote(seed_predictions: np.ndarray) -> np.ndarray:
    """Return the 3-seed majority; an all-different tie follows seed 0."""
    output: list[int] = []
    for column in seed_predictions.T:
        counts = np.bincount(column, minlength=3)
        winners = np.flatnonzero(counts == counts.max())
        output.append(int(winners[0]) if len(winners) == 1 else int(column[0]))
    return np.asarray(output, dtype=np.int64)


class DINOv2PrototypePredictor:
    def __init__(self, artifact_path: str | Path, device: str = "cpu"):
        from transformers import AutoImageProcessor, AutoModel

        artifact = np.load(Path(artifact_path), allow_pickle=False)
        self.prototypes = artifact["prototypes"].astype(np.float32)
        self.thresholds = artifact["thresholds"].astype(np.float32)
        if self.prototypes.shape[:2] != (3, 2) or self.thresholds.shape != (3, 2):
            raise ValueError("Invalid prototype artifact")

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.lock = Lock()

    @staticmethod
    def _prepare(image: Image.Image) -> Image.Image:
        return ImageOps.exif_transpose(image).convert("RGB")

    @torch.inference_mode()
    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        prepared = [self._prepare(image) for image in images]
        with self.lock:
            pixels = self.processor(images=prepared, return_tensors="pt")["pixel_values"].to(self.device)
            output = self.model(pixel_values=pixels)
            embeddings = torch.cat(
                [output.pooler_output, output.last_hidden_state[:, 1:].mean(dim=1)], dim=1
            ).float().cpu().numpy()
        embeddings = l2_normalize(embeddings)

        per_seed = []
        for seed in range(3):
            scores = embeddings @ self.prototypes[seed].T
            target = scores.argmax(axis=1)
            accept = scores[np.arange(len(scores)), target]
            threshold = self.thresholds[seed, target]
            per_seed.append(np.where(accept >= threshold, target, 2))
        voted = majority_vote(np.stack(per_seed))
        return [CLASSES[index] for index in voted]

    def predict(self, image: Image.Image) -> str:
        return self.predict_batch([image])[0]

