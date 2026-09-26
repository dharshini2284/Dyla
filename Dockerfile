# Deployable image for the matcher UI.
#
# Two choices worth explaining:
#
#  * CPU-only torch, from PyTorch's cpu index. The default wheel drags in the
#    whole CUDA stack -- ~2.5 GB of it -- which is dead weight on every free
#    tier, all of which are CPU.
#
#  * The backbone is downloaded at BUILD time, not first request. It is ~400 MB,
#    and a scale-to-zero platform would otherwise pay that on every cold start.
#    Baking it in trades image size for a predictable first response.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    OMP_NUM_THREADS=2 \
    PORT=8765

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, so the later install cannot pull the CUDA build.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Bake the backbone into the image (see note above).
RUN python -c "import timm; timm.create_model('vit_base_patch16_siglip_256.v2_webli', \
        pretrained=True, num_classes=0)" && \
    du -sh $HF_HOME

# The bundle from scripts/package_deploy.py: index, whitener, distractors,
# calibrator, slim metadata and 200px thumbnails (~28 MB for 3k items).
COPY deploy/bundle/data/ ./data/

EXPOSE 8765

# Cloud Run and friends inject $PORT; default to 8765 locally.
CMD ["sh", "-c", "vpm ui --index data/index.npz --items data/items.jsonl \
     --images data/images --calibrator data/calibrator \
     --distractors data/distractors.npz --host 0.0.0.0 --port ${PORT}"]
