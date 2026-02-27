# =============================================================================
# VetVision-LM Dockerfile
#
# Build:  docker build -t vetvision-lm .
# Run smoke test:
#   docker run --rm vetvision-lm python scripts/pretrain.py --smoke-test
# Run demo:
#   docker run --rm -p 7860:7860 vetvision-lm python scripts/demo.py
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# ── system dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── working directory ────────────────────────────────────────────────────────
WORKDIR /workspace/vetvision-lm

# ── Python dependencies (layer-cached) ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── copy source ──────────────────────────────────────────────────────────────
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# ── environment ──────────────────────────────────────────────────────────────
ENV PYTHONPATH="/workspace/vetvision-lm/src:${PYTHONPATH}"
ENV TOKENIZERS_PARALLELISM="false"

# ── expose Gradio port ───────────────────────────────────────────────────────
EXPOSE 7860

# ── default: run smoke test ──────────────────────────────────────────────────
CMD ["python", "scripts/pretrain.py", "--smoke-test"]
