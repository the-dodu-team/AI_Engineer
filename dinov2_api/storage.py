"""Optional Supabase persistence for received images and predictions.

Enabled only when SUPABASE_URL and SUPABASE_SECRET_KEY are set. Failures are logged and never
affect the API response.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass

import httpx

log = logging.getLogger("dinov2_api.storage")

MODEL_NAME = "dinov2-base-prototype"
_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp"}
_MIME = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp"}


@dataclass
class Upload:
    data: bytes
    filename: str
    fmt: str | None
    width: int
    height: int
    prediction: str
    processing_ms: float


class SupabaseStore:
    def __init__(self, url: str, key: str, bucket: str, source: str) -> None:
        self.bucket = bucket
        self.source = source
        self.base = url.rstrip("/")
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    @classmethod
    def from_env(cls) -> "SupabaseStore | None":
        url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SECRET_KEY")
        if not url or not key:
            return None
        return cls(url, key, os.getenv("SUPABASE_BUCKET", "test-images"), os.getenv("SUPABASE_SOURCE", "ut_phone"))

    async def save(self, items: list[Upload]) -> None:
        try:
            async with httpx.AsyncClient(timeout=30, headers=self.headers) as client:
                for item in items:
                    await self._save_one(client, item)
        except Exception:  # never break the API because of storage
            log.exception("Supabase save failed")

    async def _save_one(self, client: httpx.AsyncClient, item: Upload) -> None:
        sha = hashlib.sha256(item.data).hexdigest()
        ext = _EXT.get((item.fmt or "").upper(), "jpg")
        path = f"{sha[:2]}/{sha}.{ext}"
        up = await client.post(
            f"{self.base}/storage/v1/object/{self.bucket}/{path}",
            content=item.data,
            headers={"Content-Type": _MIME[ext], "x-upsert": "true"},
        )
        up.raise_for_status()

        found = await client.get(f"{self.base}/rest/v1/images", params={"sha256": f"eq.{sha}", "select": "id"})
        found.raise_for_status()
        rows = found.json()
        if rows:
            image_id = rows[0]["id"]
        else:
            ins = await client.post(
                f"{self.base}/rest/v1/images",
                json={
                    "sha256": sha, "bucket": self.bucket, "storage_path": path, "source": self.source,
                    "width": item.width, "height": item.height, "size_bytes": len(item.data), "note": item.filename,
                },
                headers={"Prefer": "return=representation"},
            )
            ins.raise_for_status()
            image_id = ins.json()[0]["id"]

        pred = await client.post(
            f"{self.base}/rest/v1/predictions",
            json={"image_id": image_id, "model": MODEL_NAME, "predicted": item.prediction,
                  "processing_ms": item.processing_ms},
        )
        pred.raise_for_status()
