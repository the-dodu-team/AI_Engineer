"""Smoke test for the DINOv2 API. Prints only statuses, never the API key.

    DINOV2_API_KEY=... python scripts/smoke_test.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import httpx
from PIL import Image

CLASSES = {"book", "laptop", "other"}


def png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buffer, "PNG")
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    key = os.environ.get("DINOV2_API_KEY", "")
    headers = {"ngrok-skip-browser-warning": "1"}
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}".rstrip())

    with httpx.Client(timeout=120, headers=headers) as client:
        r = client.get(f"{base}/health")
        check("GET /health", r.status_code == 200 and r.json().get("status") == "ok", f"HTTP {r.status_code}")

        files = {"file": ("a.png", png((120, 80, 200)), "image/png")}
        r = client.post(f"{base}/predict", files=files)
        check("POST /predict without key is rejected", r.status_code == 401 if key else True, f"HTTP {r.status_code}")
        r = client.post(f"{base}/predict", files=files, headers={"X-API-Key": "wrong-key"})
        check("POST /predict with a wrong key is rejected", r.status_code == 401 if key else True, f"HTTP {r.status_code}")

        auth = {"X-API-Key": key} if key else {}
        r = client.post(f"{base}/predict", files={"file": ("a.png", png((120, 80, 200)), "image/png")}, headers=auth)
        ok = r.status_code == 200 and r.json()["result"]["prediction"] in CLASSES
        check("POST /predict returns a class", ok, f"HTTP {r.status_code}" + (f" -> {r.json()['result']['prediction']}" if ok else ""))

        batch = [("files", (f"{i}.png", png((40 * i, 90, 160)), "image/png")) for i in range(1, 4)]
        r = client.post(f"{base}/predict/batch", files=batch, headers=auth)
        ok = r.status_code == 200 and r.json()["image_count"] == 3 and all(x["prediction"] in CLASSES for x in r.json()["results"])
        check("POST /predict/batch (3 images)", ok, f"HTTP {r.status_code}")

        r = client.post(f"{base}/predict", files={"file": ("bad.png", b"not an image", "image/png")}, headers=auth)
        check("invalid image is rejected", r.status_code == 400, f"HTTP {r.status_code}")

    print("RESULT:", "all passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
