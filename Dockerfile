# WhisperX ASR API Service Dockerfile
#
# Build args (override per image variant):
#   TORCH_VERSION   - PyTorch version to install (default 2.7.1, broadly
#                     compatible from Pascal through Hopper).
#   TORCH_INDEX_URL - PyTorch wheel index URL (default cu126). For Blackwell
#                     (RTX 50xx) use TORCH_VERSION=2.8.0 with cu128.
#
# Image structure notes:
# - ubuntu:22.04 base instead of nvidia/cuda devel: the torch and
#   ctranslate2 wheels bundle the CUDA runtime they need, and the NVIDIA
#   container runtime injects the driver libraries. The devel toolchain
#   (~8 GB) was never used at runtime.
# - All pip installs in a single layer so the WhisperX-induced torch
#   upgrade and subsequent re-pin do not persist as dead layers
#   (~10 GB of stale torch copies in the previous layering).
ARG TORCH_VERSION=2.7.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

FROM ubuntu:22.04

ARG TORCH_VERSION
ARG TORCH_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Ubuntu security updates newer than the ubuntu:22.04 base image (e.g.
# openssl) are applied with apt-get upgrade.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Single layer: torch pin -> WhisperX (silently upgrades torch) -> pyannote
# -> torch re-pin -> transformers -> API deps -> security floors for
# transitive dependencies. Intermediate states never materialize as image
# layers. The floors are installed with the default only-if-needed upgrade
# strategy, so torch and the rest of the ML stack keep their pinned versions.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torchaudio==${TORCH_VERSION} \
        --index-url ${TORCH_INDEX_URL} && \
    pip3 install --no-cache-dir git+https://github.com/sealambda/whisperX.git@feat/pyannote-audio-4 && \
    sed -i 's/use_token=/token=/g' \
        /usr/local/lib/python3.10/dist-packages/whisperx/diarize.py && \
    pip3 install --no-cache-dir --upgrade pyannote.audio && \
    pip3 install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torchaudio==${TORCH_VERSION} \
        --index-url ${TORCH_INDEX_URL} && \
    pip3 install --no-cache-dir "transformers>=5.13,<6" && \
    pip3 install --no-cache-dir \
        fastapi==0.162.0 \
        "starlette>=1.3.1" \
        "uvicorn[standard]==0.54.0" \
        python-multipart==0.0.32 \
        pydantic==2.14.1 \
        prometheus-client==0.20.0 \
        "ray[serve]>=2.56.0" \
        "protobuf<7" && \
    pip3 install --no-cache-dir \
        "anyio>=4.14.2" \
        "urllib3>=2.8.0" \
        "aiohttp>=3.14.4" \
        "nltk>=3.10.3" \
        "msgpack>=1.2.1" \
        "fsspec>=2026.6.0" \
        "lightning>=2.6.6" \
        "certifi>=2026.10.19" \
        "virtualenv>=21.14.9"

# Prefer torch's bundled cuDNN over any system cuDNN
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Pre-download NLTK data for timestamp alignment (enables offline use)
RUN python3 -c "import nltk; nltk.download('punkt_tab', download_dir='/.cache/nltk_data')"
ENV NLTK_DATA=/.cache/nltk_data

RUN mkdir -p /.cache && chmod 777 /.cache
ENV HF_HOME=/.cache

COPY app /workspace/app
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

EXPOSE 9000 8265

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys, requests; r = requests.get('http://localhost:9000/health', timeout=8); sys.exit(r.status_code != 200)"

ENV SERVE_MODE=simple

CMD ["/workspace/entrypoint.sh"]
