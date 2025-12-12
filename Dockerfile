# WhisperX ASR API Service Dockerfile
# Based on NVIDIA CUDA for GPU support

FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support (includes bundled cuDNN 9.8)
RUN pip3 install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Set library path to prefer PyTorch's bundled cuDNN over system cuDNN
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Install WhisperX from sealambda's pyannote-audio-4 compatible branch
# Credit: https://github.com/sealambda/whisperX/tree/feat/pyannote-audio-4
RUN pip3 install --no-cache-dir git+https://github.com/sealambda/whisperX.git@feat/pyannote-audio-4

# Patch WhisperX diarize.py to use 'token=' instead of 'use_token=' for pyannote.audio 4.0
# This handles both single-line and multi-line formatting
RUN sed -i 's/use_token=/token=/g' \
    /usr/local/lib/python3.10/dist-packages/whisperx/diarize.py

# Install latest pyannote.audio for community-1 model support
RUN pip3 install --no-cache-dir --upgrade pyannote.audio

# Install API dependencies
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0

# Pre-download NLTK data for timestamp alignment (enables offline use)
RUN python3 -c "import nltk; nltk.download('punkt_tab', download_dir='/.cache/nltk_data')"
ENV NLTK_DATA=/.cache/nltk_data

# Create cache directory
RUN mkdir -p /.cache && chmod 777 /.cache

# Copy application code
COPY app /workspace/app

# Expose API port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:9000/health')" || exit 1

# Run the FastAPI application
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
