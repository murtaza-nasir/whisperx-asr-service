# WhisperX ASR API Service

A production-ready Automatic Speech Recognition (ASR) API service powered by **WhisperX**, providing high-quality transcription with **speaker diarization** and **word-level timestamps**.

This service provides an API compatible with `openai-whisper-asr-webservice`, making it a drop-in replacement for applications using that service, such as Speakr.

## Features

- ✅ **High-Quality Transcription** using OpenAI Whisper models via WhisperX
- ✅ **Speaker Diarization** with Pyannote.audio 3.1 ("Who spoke when")
- ✅ **Word-Level Timestamps** with phoneme-based alignment
- ✅ **Multi-Language Support** - 90+ languages
- ✅ **Multiple Output Formats** - JSON, SRT, VTT, TSV, plain text
- ✅ **GPU Acceleration** for fast processing
- ✅ **RESTful API** compatible with existing ASR services
- ✅ **Docker Deployment** for easy scaling and portability
- ✅ **Model Caching** for improved performance

## Architecture

```
Audio Input → WhisperX Model → Timestamp Alignment → Speaker Diarization → Formatted Output
```

**Pipeline Steps:**
1. **Transcription**: WhisperX transcribes audio with coarse timestamps
2. **Alignment**: Wav2Vec2 provides precise word-level timestamps
3. **Diarization**: Pyannote.audio identifies speakers and assigns to words
4. **Formatting**: Results formatted as JSON, SRT, VTT, or other formats

## Prerequisites

### Hardware Requirements

**Minimum (CPU-only):**
- CPU: 6+ cores
- RAM: 8GB
- Storage: 20GB

**Recommended (GPU):**
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 3080, etc.)
- CPU: 8+ cores
- RAM: 16GB
- Storage: 50GB SSD

**Optimal (Production):**
- GPU: NVIDIA A100, RTX 4090, or equivalent
- CPU: 16+ cores
- RAM: 32GB
- Storage: 100GB+ NVMe SSD

### Software Requirements

- **Docker** and **Docker Compose**
- **NVIDIA Docker Runtime** (for GPU support)
- **Hugging Face Account** (for speaker diarization models)

## Quick Start

### 1. Clone or Create Repository

```bash
# If using Git
git init
git remote add origin <your-repo-url>

# Or just create the directory structure as shown in this repository
```

### 2. Get Hugging Face Token

Speaker diarization requires a Hugging Face token and model access:

1. Create account at [huggingface.co](https://huggingface.co/join)
2. Generate token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept user agreements for:
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) (latest model - pyannote.audio 4.0)

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Hugging Face token
nano .env
```

Update `HF_TOKEN` in `.env`:
```bash
HF_TOKEN=your_actual_huggingface_token_here
```

### 4. Build and Run

```bash
# Build the Docker image
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f
```

The service will be available at `http://localhost:9000`

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
# Edit docker-compose.yml ports:
ports:
  - "0.0.0.0:9000:9000"  # Expose to network
```

2. **On Speakr Machine:** Update configuration

```bash
# In Speakr's .env file
USE_ASR_ENDPOINT=true
ASR_BASE_URL=http://<GPU_MACHINE_IP>:9000
```

**Security Note:** For production, use HTTPS and authentication. Consider using a reverse proxy (nginx, Traefik) with SSL certificates.

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
```

### Model Selection

Available Whisper models (speed vs accuracy tradeoff):

| Model | Size | VRAM | Speed | Quality |
|-------|------|------|-------|---------|
| `tiny` | 39M | ~1GB | Fastest | Lowest |
| `base` | 74M | ~1GB | Very Fast | Low |
| `small` | 244M | ~2GB | Fast | Medium |
| `medium` | 769M | ~5GB | Moderate | Good |
| `large-v2` | 1550M | ~10GB | Slow | Excellent |
| `large-v3` | 1550M | ~10GB | Slow | Best |

**Recommendation:** Use `large-v3` for best quality, `small` for speed/resource constraints.

## Deployment Options

### Development (Local Machine)

```bash
docker-compose up
```

### Production (Dedicated GPU Server)

```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Scaling (Multiple Workers)

For high-volume deployments, run multiple instances behind a load balancer:

```bash
# Scale to 3 instances
docker-compose up -d --scale whisperx-asr=3
```

Use nginx or Traefik for load balancing.

### Cloud Deployment

**AWS EC2 with GPU:**
- Instance type: `g4dn.xlarge` or better
- AMI: Deep Learning AMI (Ubuntu)
- Open port 9000 in security group

**Google Cloud Platform:**
- Machine type: `n1-standard-4` with Tesla T4 GPU
- Image: Deep Learning VM Image
- Firewall: Allow port 9000

## Monitoring and Logs

### View Logs

```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

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

**Symptom:** `CUDA out of memory` errors

**Solutions:**
1. Use smaller model (`small` instead of `large-v3`)
2. Reduce batch size in `.env`: `BATCH_SIZE=8`
3. Use `COMPUTE_TYPE=int8` for lower memory usage
4. Process shorter audio segments

### Speaker Diarization Not Working

**Symptom:** No speaker labels in output

**Solutions:**
1. Verify HF_TOKEN is set correctly
2. Accept model user agreements on Hugging Face
3. Check logs for diarization errors: `docker-compose logs`
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
docker-compose logs whisperx-asr
```

Common causes:
- Invalid audio format (use ffmpeg to convert)
- Model not loaded (check VRAM, logs)
- Incorrect parameters (check API docs)

## Supported Audio Formats

The service supports all formats supported by FFmpeg:

- **Audio:** MP3, WAV, M4A, FLAC, AAC, OGG, WMA
- **Video:** MP4, AVI, MOV, MKV, WebM (audio track extracted)
- **Other:** AMR, 3GP, 3GPP

Large files are automatically handled in chunks.

## Performance Benchmarks

Tested on RTX 3080 (10GB VRAM):

| Model | Audio Length | Processing Time | Realtime Factor |
|-------|--------------|-----------------|-----------------|
| tiny | 10 minutes | 20 seconds | 30x |
| small | 10 minutes | 45 seconds | 13x |
| medium | 10 minutes | 90 seconds | 6.7x |
| large-v3 | 10 minutes | 180 seconds | 3.3x |

*With diarization enabled, add ~30% processing time*

## Security Considerations

### For Production Deployment:

1. **Use HTTPS:** Deploy behind reverse proxy with SSL
2. **Authentication:** Add API key authentication
3. **Rate Limiting:** Implement request rate limiting
4. **Input Validation:** Validate file sizes and formats
5. **Network Security:** Use firewall rules, VPN, or private networks
6. **Secrets Management:** Use Docker secrets or vault for HF_TOKEN

### Example nginx configuration:

```nginx
server {
    listen 443 ssl;
    server_name asr.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Optional: Add API key authentication
        if ($http_x_api_key != "your-secret-key") {
            return 401;
        }
    }
}
```

## Maintenance

### Updating WhisperX

```bash
# Pull latest changes
git pull

# Rebuild image
docker-compose build --no-cache

# Restart service
docker-compose up -d
```

### Clearing Cache

```bash
# Remove model cache
docker-compose down -v
docker volume rm whisperx-asr-service_whisperx-cache

# Rebuild
docker-compose up -d
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

- **GitHub Issues:** [Create an issue](https://github.com/yourusername/whisperx-asr-service/issues)
- **WhisperX Issues:** [WhisperX repository](https://github.com/m-bain/whisperX/issues)

## Credits

- **WhisperX:** [m-bain/whisperX](https://github.com/m-bain/whisperX)
- **OpenAI Whisper:** [openai/whisper](https://github.com/openai/whisper)
- **Pyannote.audio:** [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **Docker WhisperX:** [jim60105/docker-whisperX](https://github.com/jim60105/docker-whisperX)

## Changelog

### v1.0.0 (2025-11-22)
- Initial release
- WhisperX integration with API wrapper
- Speaker diarization support
- Docker deployment
- Compatible API with openai-whisper-asr-webservice
