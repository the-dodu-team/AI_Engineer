from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from secrets import compare_digest
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .predictor import DINOv2PrototypePredictor

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BATCH_IMAGES = 20


class PredictionItem(BaseModel):
    filename: str
    prediction: Literal["book", "laptop", "other"]


class PredictionResponse(BaseModel):
    model: str = "dinov2-base-prototype"
    processing_ms: float
    result: PredictionItem


class BatchPredictionResponse(BaseModel):
    model: str = "dinov2-base-prototype"
    image_count: int
    processing_ms: float
    results: list[PredictionItem]


def _check_api_key(value: str | None) -> None:
    expected = os.getenv("DINOV2_API_KEY")
    if expected and not compare_digest(value or "", expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty image: {file.filename}")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds 10 MiB: {file.filename}")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid image: {file.filename}") from None


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = Path(os.getenv("DINOV2_ARTIFACT", "artifacts/dinov2_base_prototypes.npz"))
    device = os.getenv("DINOV2_DEVICE", "cpu")
    app.state.predictor = await run_in_threadpool(DINOv2PrototypePredictor, artifact, device)
    try:
        yield
    finally:
        del app.state.predictor


app = FastAPI(
    title="DINOv2-base UT classifier",
    version="0.0.0",
    description="fastapi_inference_v2: book / laptop / other image classification",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "dinov2-base-prototype", "api": "fastapi_inference_v2"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="JPEG, PNG, WebP or another Pillow-readable image")],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> PredictionResponse:
    _check_api_key(x_api_key)
    image = await _read_image(file)
    started = perf_counter()
    prediction = await run_in_threadpool(request.app.state.predictor.predict, image)
    elapsed = (perf_counter() - started) * 1000
    return PredictionResponse(
        processing_ms=round(elapsed, 2),
        result=PredictionItem(filename=file.filename or "image", prediction=prediction),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    files: Annotated[list[UploadFile], File(description="1-20 images")],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> BatchPredictionResponse:
    _check_api_key(x_api_key)
    if not 1 <= len(files) <= MAX_BATCH_IMAGES:
        raise HTTPException(status_code=400, detail="Batch must contain 1-20 images")
    images = [await _read_image(file) for file in files]
    started = perf_counter()
    predictions = await run_in_threadpool(request.app.state.predictor.predict_batch, images)
    elapsed = (perf_counter() - started) * 1000
    return BatchPredictionResponse(
        image_count=len(files),
        processing_ms=round(elapsed, 2),
        results=[
            PredictionItem(filename=file.filename or f"image-{index + 1}", prediction=prediction)
            for index, (file, prediction) in enumerate(zip(files, predictions))
        ],
    )

