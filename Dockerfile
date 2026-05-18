# WhisperX ASR API Service Dockerfile
# Based on NVIDIA CUDA for GPU support
#
# Build args (override per image variant):
#   TORCH_VERSION   - PyTorch version to install (default 2.7.1, broadly
#                     compatible from Pascal through Hopper).
#   TORCH_INDEX_URL - PyTorch wheel index URL (default cu126; still supports
#                     Pascal/Volta/Turing/Ampere/Hopper). For Blackwell
#                     (RTX 50xx) use TORCH_VERSION=2.8.0 with cu128. CUDA
#                     12.8 dropped Pascal/Maxwell support per upstream.
ARG TORCH_VERSION=2.12.0+cu130
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

FROM --platform=linux/arm64 nvidia/cuda:13.1.2-cudnn-devel-ubuntu22.04

# Re-declare ARGs after FROM so they're visible inside the build stage.
ARG TORCH_VERSION
ARG TORCH_INDEX_URL

# DGX Spark (GB10)
ENV TORCH_CUDA_ARCH_LIST="12.1"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support (includes bundled cuDNN). The WhisperX
# install below will silently upgrade torch to satisfy its own requirements;
# we re-pin to ${TORCH_VERSION} after that step so the requested version sticks.
RUN pip3 install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchaudio==2.11.0+cu130 \
    --index-url ${TORCH_INDEX_URL}

# Set library path to prefer PyTorch's bundled cuDNN over system cuDNN
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Install WhisperX from sealambda's pyannote-audio-4 compatible branch
# Credit: https://github.com/sealambda/whisperX/tree/feat/pyannote-audio-4
RUN pip3 install --no-cache-dir git+https://github.com/sealambda/whisperX.git@feat/pyannote-audio-4

# Patch WhisperX diarize.py to use 'token=' instead of 'use_token=' for pyannote.audio 4.0
# This handles both single-line and multi-line formatting
RUN sed -i 's/use_token=/token=/g' \
    /usr/local/lib/python3.10/dist-packages/whisperx/diarize.py

# Compiling CTranslate2 from source, bypassing Blackwell architecture parser error in CMake
RUN pip3 uninstall -y ctranslate2 && \
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2 && \
    cd /tmp/ctranslate2 && \
    # Hotfix: Hardcoding sm_121 right after the flag generation function
    sed -i '/cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})/a \    list(APPEND ARCH_FLAGS "-gencode;arch=compute_121,code=sm_121")' CMakeLists.txt && \
    # Remove old architectures incompatible with CUDA 13 from the default fallback list
    sed -i 's/"3.5;5.0;5.3;6.0;6.1;7.0;7.5;8.0;8.6;8.6+PTX"/"9.0;9.0+PTX"/' CMakeLists.txt 2>/dev/null || true && \
    mkdir build && cd build && \
    cmake -DWITH_CUDA=ON \
          -DCUDA_ARCH_LIST="9.0" \
          -DWITH_MKL=OFF \
          -DOPENMP_RUNTIME=COMP .. && \
    make -j$(nproc) install && \
    cd ../python && \
    pip3 install --no-cache-dir . && \
    rm -rf /tmp/ctranslate2

# Install latest pyannote.audio for community-1 model support
RUN pip3 install --no-cache-dir --upgrade pyannote.audio

# Re-pin torch/torchaudio to the requested version. WhisperX (sealambda fork)
# requires torch>=2.8 and silently upgrades the install above to 2.8, which
# breaks Pascal cards. Reinstalling here ensures TORCH_VERSION sticks for the
# image variant being built (cu126/2.7.1 default; cu128/2.8.0 for Blackwell).
RUN pip3 install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchaudio==2.11.0+cu130 \
    --index-url ${TORCH_INDEX_URL}

# Install API dependencies
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    prometheus-client==0.20.0 \
    "ray[serve]>=2.9"

# Pre-download NLTK data for timestamp alignment (enables offline use)
RUN python3 -c "import nltk; nltk.download('punkt_tab', download_dir='/.cache/nltk_data')"
ENV NLTK_DATA=/.cache/nltk_data

# Create cache directory
RUN mkdir -p /.cache && chmod 777 /.cache

# Copy application code
COPY app /workspace/app

# Copy entrypoint script
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Expose API port (9000) and Ray dashboard (8265)
EXPOSE 9000 8265

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:9000/health')" || exit 1

# Default: simple mode (uvicorn). Set SERVE_MODE=ray for Ray Serve.
ENV SERVE_MODE=simple

CMD ["/workspace/entrypoint.sh"]
