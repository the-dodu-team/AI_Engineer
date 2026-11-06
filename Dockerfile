# DINOv2-base laptop/book/other API, run as a container on Vercel (Container Runtime).
# No application code is changed: the same dinov2_api package that runs locally.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    PYTHONPATH=/app \
    DINOV2_ARTIFACT=/app/artifacts/dinov2_base_prototypes.npz \
    DINOV2_DEVICE=cpu \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first: the default PyPI wheel pulls about 2 GB of CUDA libraries.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the DINOv2-base weights (about 330 MB) into the image so a cold start needs no download.
RUN python -c "from transformers import AutoImageProcessor, AutoModel; AutoImageProcessor.from_pretrained('facebook/dinov2-base'); AutoModel.from_pretrained('facebook/dinov2-base')"

COPY dinov2_api ./dinov2_api
COPY artifacts ./artifacts

ENV HF_HUB_OFFLINE=1

# Vercel sends traffic to $PORT (default 80). DINOV2_API_KEY must be set as a project environment variable.
CMD ["sh", "-c", "exec python -m uvicorn dinov2_api.app:app --host 0.0.0.0 --port ${PORT:-80} --workers 1"]
