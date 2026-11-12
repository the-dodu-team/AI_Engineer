"""Optional Supabase persistence of prediction records (no original images are stored).

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
    def __init__(self, url: str, key: str, source: str) -> None:
        self.source = source
        self.base = url.rstrip("/")
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}"}

    @classmethod
    def from_env(cls) -> "SupabaseStore | None":
        url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SECRET_KEY")
        if not url or not key:
            return None
        return cls(url, key, os.getenv("SUPABASE_SOURCE", "ut_phone"))

    async def save(self, items: list[Upload]) -> None:
        try:
            async with httpx.AsyncClient(timeout=30, headers=self.headers) as client:
                for item in items:
                    await self._save_one(client, item)
        except Exception:  # never break the API because of storage
            log.exception("Supabase save failed")

    async def _save_one(self, client: httpx.AsyncClient, item: Upload) -> None:
        sha = hashlib.sha256(item.data).hexdigest()
        found = await client.get(f"{self.base}/rest/v1/images", params={"sha256": f"eq.{sha}", "select": "id"})
        found.raise_for_status()
        rows = found.json()
        if rows:
            image_id = rows[0]["id"]
        else:
            ins = await client.post(
                f"{self.base}/rest/v1/images",
                json={
                    "sha256": sha, "source": self.source,
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
