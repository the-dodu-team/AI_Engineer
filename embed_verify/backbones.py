"""Vision backbones (DINOv2, CLIP, SigLIP) that map an image to one embedding vector."""
from __future__ import annotations

import hashlib
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

MODELS = {
    "dinov2-small": ("facebook/dinov2-small", "dinov2"),
    "dinov2-base": ("facebook/dinov2-base", "dinov2"),
    "clip-vit-b16": ("openai/clip-vit-base-patch16", "clip"),
    "siglip-base": ("google/siglip-base-patch16-224", "siglip"),
}


def load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class Backbone:
    """Wraps one pretrained vision tower. `embed()` returns an un-normalised [B, D] tensor."""

    def __init__(self, name: str, device: str = "cpu", dinov2_pool: str = "cls_mean"):
        from transformers import AutoImageProcessor

        if name not in MODELS:
            raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
        self.name, self.device, self.dinov2_pool = name, device, dinov2_pool
        self.hf_id, self.kind = MODELS[name]
        self.processor = AutoImageProcessor.from_pretrained(self.hf_id)
        if self.kind == "dinov2":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.hf_id)
        elif self.kind == "clip":
            from transformers import CLIPVisionModelWithProjection
            self.model = CLIPVisionModelWithProjection.from_pretrained(self.hf_id)
        else:
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(self.hf_id)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ---- model facts -------------------------------------------------
    @property
    def params_m(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    @property
    def revision(self) -> str:
        return str(getattr(self.model.config, "_commit_hash", None) or "unknown")

    @property
    def image_size(self) -> int:
        size = self.processor.size
        if "shortest_edge" in size:
            crop = getattr(self.processor, "crop_size", None)
            return int(crop["height"]) if crop else int(size["shortest_edge"])
        return int(size["height"])

    @property
    def mean_std(self) -> tuple[list[float], list[float]]:
        return list(self.processor.image_mean), list(self.processor.image_std)

    # ---- forward -----------------------------------------------------
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)
        if self.kind == "dinov2":
            out = self.model(pixel_values=pixel_values)
            cls = out.pooler_output
            if self.dinov2_pool == "cls":
                return cls
            return torch.cat([cls, out.last_hidden_state[:, 1:].mean(dim=1)], dim=1)
        if self.kind == "clip":
            return self.model(pixel_values=pixel_values).image_embeds
        return self.model(pixel_values=pixel_values).pooler_output

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    # ---- fine-tuning helpers ----------------------------------------
    def last_blocks(self, n: int) -> list[torch.nn.Module]:
        layers = self.model.encoder.layer if self.kind == "dinov2" else self.model.vision_model.encoder.layers
        return list(layers)[-n:] if n > 0 else []

    def final_modules(self) -> list[torch.nn.Module]:
        if self.kind == "dinov2":
            return [self.model.layernorm]
        if self.kind == "clip":
            return [self.model.vision_model.post_layernorm, self.model.visual_projection]
        return [self.model.vision_model.post_layernorm, self.model.vision_model.head]


@torch.inference_mode()
def extract(backbone: Backbone, paths: list[str], batch_size: int = 16, desc: str = "") -> np.ndarray:
    """Embed images in order. Returns float32 [N, D] (not normalised)."""
    from tqdm.auto import tqdm

    backbone.model.eval()
    chunks = []
    for start in tqdm(range(0, len(paths), batch_size), desc=desc or backbone.name, leave=False):
        images = [load_image(p) for p in paths[start:start + batch_size]]
        chunks.append(backbone.embed(backbone.preprocess(images)).float().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def cached_extract(backbone: Backbone, paths: list[str], sha256s: list[str], cache_dir: Path,
                   batch_size: int = 16) -> tuple[np.ndarray, bool]:
    """`extract` with an on-disk cache keyed by model, pooling and image hashes."""
    key = hashlib.sha1("\n".join(sorted(sha256s)).encode()).hexdigest()[:16]
    pool = backbone.dinov2_pool if backbone.kind == "dinov2" else "-"
    cache_dir.mkdir(parents=True, exist_ok=True)
    file = cache_dir / f"{backbone.name}_{pool}_{key}.npz"
    if file.is_file():
        data = np.load(file, allow_pickle=False)
        lookup = dict(zip(data["sha256"].tolist(), range(len(data["sha256"]))))
        return data["X"][[lookup[s] for s in sha256s]], True
    X = extract(backbone, paths, batch_size)
    np.savez(file, X=X, sha256=np.array(sha256s))
    return X, False


def measure_latency(backbone: Backbone, paths: list[str], n: int = 20, warmup: int = 3) -> dict:
    """Batch-size-1 latency: image decode + preprocess, and model forward, in milliseconds."""
    sample = paths[:max(n, 1)]
    order = [sample[i % len(sample)] for i in range(warmup + len(sample))]
    prep, fwd = [], []
    with torch.inference_mode():
        for i, path in enumerate(order):
            t0 = time.perf_counter()
            pixels = backbone.preprocess([load_image(path)])
            t1 = time.perf_counter()
            out = backbone.embed(pixels)
            t2 = time.perf_counter()
            if i >= warmup:
                prep.append((t1 - t0) * 1000)
                fwd.append((t2 - t1) * 1000)
    return {"model": backbone.name, "hf_id": backbone.hf_id, "revision": backbone.revision,
            "params_m": round(backbone.params_m, 2), "embedding_dim": int(out.shape[1]),
            "device": backbone.device, "threads": torch.get_num_threads(),
            "preprocess_ms_median": round(statistics.median(prep), 2),
            "forward_ms_median": round(statistics.median(fwd), 2),
            "total_ms_median": round(statistics.median(prep) + statistics.median(fwd), 2),
            "timed_images": len(fwd)}
