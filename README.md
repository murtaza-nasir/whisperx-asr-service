# WhisperX ASR API Service

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Docker Hub](https://img.shields.io/badge/docker%20hub-whisperx--asr--service-blue.svg)](https://hub.docker.com/r/learnedmachine/whisperx-asr-service)
[![Docker Pulls](https://img.shields.io/docker/pulls/learnedmachine/whisperx-asr-service)](https://hub.docker.com/r/learnedmachine/whisperx-asr-service)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com/)
[![GPU Required](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/murtaza-nasir/whisperx-asr-service)

**⚠️ Alpha Version - For Self-Hosting Enthusiasts**

A simple ASR API service powered by WhisperX for transcription with speaker diarization. Built for self-hosters running Speakr or similar applications.

## What This Does

- Transcribes audio files using OpenAI Whisper models
- Identifies speakers ("Who spoke when") using Pyannote.audio
- Returns word-level timestamps
- Supports 90+ languages
- Outputs JSON, SRT, VTT, TSV formats
- Runs on your own GPU hardware via Docker

## Limitations

- **Not production-grade**: Basic error handling, no authentication
- **Single instance**: No built-in scaling or load balancing
- **GPU required**: Needs NVIDIA GPU with 14GB+ VRAM for large models
- **File size limits**: Large audio files (>1GB) can cause out-of-memory errors
- **VRAM usage**: Memory consumption increases with file size and diarization
- **Alpha software**: Expect bugs and breaking changes

## How It Works

Audio → Whisper (transcription) → Wav2Vec2 (alignment) → Pyannote (speaker ID) → Output

## Prerequisites

### Hardware Requirements

GPU memory requirements vary by model size:

| Whisper Model | VRAM Required (with diarization) | Suitable GPUs |
|---------------|----------------------------------|---------------|
| tiny, base | ~4-5GB | RTX 3060 8GB, RTX 2060, GTX 1660 Ti |
| small | ~6GB | RTX 3060, RTX 2070, RTX 2080 |
| medium | ~10GB | RTX 3080, RTX 3060 12GB, RTX 2080 Ti |
| large-v2, large-v3 | ~14GB | **RTX 3090**, RTX 4090, A6000, A100 |

*Note: Measured with preloaded model + alignment + pyannote community-1 diarization on RTX 3090*

**Minimum Configuration (small/medium models):**
- GPU: NVIDIA RTX 3060 (12GB VRAM) or better
- CPU: 8+ cores
- RAM: 16GB
- Storage: 50GB SSD

**Recommended (large-v3 with diarization):**
- GPU: NVIDIA RTX 3090 (24GB VRAM) or RTX 4090
- CPU: 12+ cores
- RAM: 32GB
- Storage: 100GB SSD

### Software Requirements

- **Docker** and **Docker Compose**
- **NVIDIA Docker Runtime** (for GPU support)
- **Hugging Face Account** (for speaker diarization models)

## Quick Start

**Two Docker Compose Files Available:**

- **`docker-compose.yml`** - Uses prebuilt image from Docker Hub (recommended for most users)
- **`docker-compose.dev.yml`** - Builds from source with live code mounting (for development)

### Option A: Use Prebuilt Docker Image (Recommended)

The easiest way to get started is using our prebuilt Docker images from Docker Hub:

```bash
# Pull the latest image
docker pull learnedmachine/whisperx-asr-service:latest

# Or pull a specific version
docker pull learnedmachine/whisperx-asr-service:v0.1.1alpha
```

Then skip to [Step 2](#2-get-hugging-face-token-and-model-access) to configure your Hugging Face token.

### Option B: Build from Source

### 1. Clone or Create Repository

```bash
# Clone the repository
git clone https://github.com/murtaza-nasir/whisperx-asr-service.git
cd whisperx-asr-service

# Or create from scratch
mkdir whisperx-asr-service && cd whisperx-asr-service
git init
```

### 2. Get Hugging Face Token and Model Access

Speaker diarization requires a Hugging Face token and model access. **You must complete ALL steps below:**

#### Step 2.1: Create Hugging Face Account
- Visit: [https://huggingface.co/join](https://huggingface.co/join)
- Sign up with your email

#### Step 2.2: Accept Model User Agreements (ALL REQUIRED)
You need to accept agreements for **all three models** used by the diarization pipeline:

1. **Main diarization model:**
   - Visit: [https://huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - Click **"Agree and access repository"**

2. **Segmentation model:**
   - Visit: [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Click **"Agree and access repository"**

3. **Speaker diarization 3.1 (dependency):**
   - Visit: [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Click **"Agree and access repository"**

Fill out the form (Company/university and use case) - approval is instant.

#### Step 2.3: Generate Access Token
- Visit: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Click **"New token"**
- Name it (e.g., "whisperx-diarization")
- Select **"Read"** permission
- Click **"Generate token"**
- Copy the token (starts with `hf_...`)

**⚠️ Important:** Without accepting the model agreement in Step 2.2, you will get a "403 Access Denied" error even with a valid token.

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Hugging Face token
nano .env
```

Update `HF_TOKEN` in `.env`:
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Run the Service

**Using Prebuilt Image (Recommended):**

```bash
# Start the service (pulls image automatically)
docker compose up -d

# Check logs
docker compose logs -f
```

**For Development (Build from Source):**

```bash
# Build and start the service
docker compose -f docker-compose.dev.yml up -d --build

# Check logs
docker compose -f docker-compose.dev.yml logs -f
```

The service will be available at `http://localhost:9000`

**Note:** The dev compose file mounts your local `./app` directory, so code changes are reflected without rebuilding.

### 5. Test the API

```bash
# Health check
curl http://localhost:9000/health

# Transcribe an audio file
curl -X POST -F "audio_file=@test.mp3" \
  -F "language=en" \
  -F "output_format=json" \
  http://localhost:9000/asr
```

## API Documentation

Once running, visit http://localhost:9000/docs for interactive API documentation.

### Main Endpoint: POST /asr

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_file` | File | Required | Audio file to transcribe |
| `task` | String | `transcribe` | Task type: `transcribe` or `translate` |
| `language` | String | Auto-detect | Language code (e.g., `en`, `es`, `fr`) |
| `model` | String | `large-v3` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `output_format` | String | `json` | Output format: `json`, `text`, `srt`, `vtt`, `tsv` |
| `word_timestamps` | Boolean | `true` | Return word-level timestamps |
| `enable_diarization` | Boolean | `true` | Enable speaker diarization |
| `num_speakers` | Integer | Auto | Exact number of speakers (if known, overrides min/max) |
| `min_speakers` | Integer | Auto | Minimum number of speakers |
| `max_speakers` | Integer | Auto | Maximum number of speakers |

**Example Request (JSON output):**

```bash
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@meeting.mp3" \
  -F "language=en" \
  -F "model=large-v3" \
  -F "output_format=json" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=5"
```

**Example Request (SRT subtitles):**

```bash
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@video.mp4" \
  -F "language=en" \
  -F "output_format=srt" \
  -F "enable_diarization=false"
```

**Example Response (JSON):**

```json
{
  "text": [
    {
      "start": 0.5,
      "end": 2.3,
      "text": " Hello, welcome to the meeting.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello", "start": 0.5, "end": 0.8, "score": 0.95},
        {"word": "welcome", "start": 0.9, "end": 1.2, "score": 0.93}
      ]
    }
  ],
  "language": "en",
  "segments": [...],
  "word_segments": [...]
}
```

### Advanced Speaker Diarization Features

#### Exact Speaker Count

When you know the exact number of speakers, use `num_speakers` for more accurate diarization:

```bash
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@interview.mp3" \
  -F "num_speakers=2" \
  -F "enable_diarization=true"
```

This overrides `min_speakers` and `max_speakers` and typically provides better accuracy than range-based detection.

#### Exclusive Speaker Diarization

This service automatically uses **exclusive speaker diarization** when available from the pyannote community-1 model. This feature simplifies reconciliation between fine-grained speaker diarization timestamps and transcription timestamps, making it ideal for applications like Speakr where you need to align transcripts with speaker segments.

**Benefits:**
- More accurate timestamp alignment between speakers and words
- Better handling of speaker transitions
- Simplified post-processing for multi-speaker transcripts

## Integration with Speakr

To use this service with Speakr instead of the default ASR endpoint:

### If Running on the Same Machine

Update Speakr's `.env` file:

```bash
# Enable ASR endpoint
USE_ASR_ENDPOINT=true

# Point to WhisperX service
ASR_BASE_URL=http://whisperx-asr-api:9000
```

If Speakr and WhisperX are in the same Docker Compose stack, use the container name. Otherwise, use `http://localhost:9000`.

### If Running on a Different GPU Machine

1. **On GPU Machine:** Deploy this service

```bash
# Make service accessible from network
# Edit docker compose.yml ports:
ports:
  - "0.0.0.0:9000:9000"  # Expose to network
```

2. **On Speakr Machine:** Update configuration

```bash
# In Speakr's .env file
USE_ASR_ENDPOINT=true
ASR_BASE_URL=http://<GPU_MACHINE_IP>:9000
```

**Note:** Replace `<GPU_MACHINE_IP>` with your GPU server's IP address. Use firewall rules to restrict access to trusted machines only.

## Configuration

### Environment Variables

Edit `.env` to customize:

```bash
# GPU or CPU processing
DEVICE=cuda              # cuda for GPU, cpu for CPU-only

# Computation precision
COMPUTE_TYPE=float16     # float16 (GPU), float32 (CPU), int8 (faster, lower quality)

# Batch size (higher = faster but more memory)
BATCH_SIZE=16           # 16 for 8GB VRAM, 32+ for high-end GPUs

# Hugging Face token for diarization
HF_TOKEN=hf_xxx...

# Model preloading (optional, reduces first-request latency)
PRELOAD_MODEL=large-v3   # Leave empty to disable, or set to: tiny, base, small, medium, large-v2, large-v3

# Maximum file size in MB (prevents out-of-memory errors)
MAX_FILE_SIZE_MB=1000    # Default 1GB, adjust lower for GPUs with <16GB VRAM
```

### Model Selection

Available Whisper models (speed vs accuracy tradeoff):

| Model | Parameters | VRAM (model only) | VRAM (full pipeline*) | Speed | Quality |
|-------|------------|-------------------|----------------------|-------|---------|
| `tiny` | 39M | ~1GB | ~4GB | Fastest | Lowest |
| `base` | 74M | ~1GB | ~5GB | Very Fast | Low |
| `small` | 244M | ~2GB | ~6GB | Fast | Medium |
| `medium` | 769M | ~5GB | ~10GB | Moderate | Good |
| `large-v2` | 1550M | ~10GB | ~14GB | Slow | Excellent |
| `large-v3` | 1550M | ~10GB | ~14GB | Slow | Best |

*Full pipeline = Whisper model + alignment model + pyannote speaker diarization (measured on RTX 3090)

**Recommendation:**
- Use `large-v3` for best quality (requires 16GB+ VRAM)
- Use `small` or `medium` for speed/resource constraints (8-12GB VRAM)

## Running the Service

```bash
# Start in foreground (see logs)
docker compose up

# Or run in background
docker compose up -d

# View logs
docker compose logs -f
```

## Monitoring and Logs

### View Logs

```bash
# Real-time logs
docker compose logs -f

# Last 100 lines
docker compose logs --tail=100

# Specific container logs
docker logs whisperx-asr-api
```

### Health Check

```bash
# Check service health
curl http://localhost:9000/health

# Response:
{
  "status": "healthy",
  "device": "cuda",
  "loaded_models": ["large-v3"]
}
```

### Performance Monitoring

Monitor GPU usage:

```bash
# NVIDIA GPU stats
nvidia-smi -l 1

# Docker container stats
docker stats whisperx-asr-api
```

## Troubleshooting

### GPU Not Detected

**Symptom:** Service runs on CPU despite having GPU

**Solution:**
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of Memory Errors

**Symptom:** `CUDA out of memory` errors or VRAM exhaustion with large files

**Solutions:**
1. **Reduce file size limit** in `.env`: `MAX_FILE_SIZE_MB=500` (default is 1000MB)
2. **Use smaller model**: `small` or `medium` instead of `large-v3`
3. **Reduce batch size** in `.env`: `BATCH_SIZE=8` or `BATCH_SIZE=4`
4. **Use int8 precision**: `COMPUTE_TYPE=int8` (lower quality but less memory)
5. **Split large files**: Process audio in smaller chunks before uploading
6. **Disable diarization**: For very large files, skip speaker diarization

**Note:** The service automatically clears GPU cache between operations to minimize VRAM buildup, but very large files (>500MB) can still cause issues.

### Speaker Diarization Not Working

**Symptom:** No speaker labels in output

**Solutions:**
1. Verify HF_TOKEN is set correctly
2. Accept model user agreements on Hugging Face
3. Check logs for diarization errors: `docker compose logs`
4. Ensure `enable_diarization=true` in request

### Slow Processing

**Symptom:** Transcription takes too long

**Solutions:**
1. Use GPU instead of CPU (`DEVICE=cuda`)
2. Use smaller model for faster processing
3. Increase `BATCH_SIZE` (if you have VRAM)
4. Disable diarization if not needed: `enable_diarization=false`

### API Returns 500 Errors

**Check logs:**
```bash
docker compose logs whisperx-asr
```

Common causes:
- Invalid audio format (use ffmpeg to convert)
- Model not loaded (check VRAM, logs)
- Incorrect parameters (check API docs)

## Supported Audio Formats

The service supports formats that WhisperX can process (via FFmpeg):

- **Audio:** MP3, WAV, M4A, FLAC, AAC, OGG, WMA
- **Video:** MP4, AVI, MOV, MKV, WebM (audio track extracted)
- **Other:** AMR, 3GP, 3GPP

**Note:** Large files (>1GB) may cause out-of-memory errors as files are loaded entirely into memory.

## Security Notes

**This service has NO built-in authentication or security features.**

If exposing to a network:
- Use firewall rules to restrict access
- Consider putting behind a reverse proxy
- Store HF_TOKEN securely (use `.env` file, not hardcoded)

## Maintenance

### Updating WhisperX

```bash
# Pull latest changes
git pull

# Rebuild image
docker compose build --no-cache

# Restart service
docker compose up -d
```

### Clearing Cache

```bash
# Remove model cache
docker compose down -v
docker volume rm whisperx-asr-service_whisperx-cache

# Rebuild
docker compose up -d
```

### Backup

Backup the cache volume to preserve downloaded models:

```bash
docker run --rm -v whisperx-asr-service_whisperx-cache:/cache \
  -v $(pwd):/backup ubuntu tar czf /backup/whisperx-cache-backup.tar.gz /cache
```

## License

This project is MIT licensed. See [LICENSE](LICENSE) for details.

WhisperX is licensed under BSD-4-Clause. See [WhisperX repository](https://github.com/m-bain/whisperX) for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:

- **GitHub Issues:** [Create an issue](https://github.com/murtaza-nasir/whisperx-asr-service/issues)
- **WhisperX Issues:** [WhisperX repository](https://github.com/m-bain/whisperX/issues)

## Credits

- **WhisperX:** [m-bain/whisperX](https://github.com/m-bain/whisperX)
- **WhisperX Pyannote.audio 4 Support:** [sealambda/whisperX@feat/pyannote-audio-4](https://github.com/sealambda/whisperX/tree/feat/pyannote-audio-4) - This service uses sealambda's fork for pyannote.audio 4.0 compatibility
- **OpenAI Whisper:** [openai/whisper](https://github.com/openai/whisper)
- **Pyannote.audio:** [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **Docker WhisperX:** [jim60105/docker-whisperX](https://github.com/jim60105/docker-whisperX)

## Changelog

### v0.1.1alpha (2025-11-23)
- Initial release
- WhisperX integration with API wrapper
- Speaker diarization support
- Docker deployment
- Compatible API with openai-whisper-asr-webservice
