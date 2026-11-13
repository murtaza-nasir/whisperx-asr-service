# WhisperX ASR Service - Setup Guide for Speakr Integration

This guide walks you through setting up the WhisperX ASR service for use with Speakr.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup on the Same Machine](#setup-on-the-same-machine)
3. [Setup on a Separate GPU Machine](#setup-on-a-separate-gpu-machine)
4. [Configuration](#configuration)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware

- **GPU Machine Requirements:**
  - NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 3080, A100, etc.)
  - 16GB+ RAM
  - 50GB+ free disk space
  - Ubuntu 20.04+ or similar Linux distribution

### Software

1. **Docker and Docker Compose**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **NVIDIA Docker Runtime**
   ```bash
   # Add NVIDIA Docker repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # Install nvidia-container-toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit

   # Restart Docker
   sudo systemctl restart docker

   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```

3. **Hugging Face Account and Token** - **YOU MUST COMPLETE ALL STEPS:**

   a. **Create Account:**
      - Visit: https://huggingface.co/join
      - Sign up with your email

   b. **Accept Model Agreements (REQUIRED - ALL THREE):**
      Accept agreements for all models used by the pipeline:
      - https://huggingface.co/pyannote/speaker-diarization-community-1
      - https://huggingface.co/pyannote/segmentation-3.0
      - https://huggingface.co/pyannote/speaker-diarization-3.1

      For each: Click **"Agree and access repository"**
      Fill out form (Company/use case) - approval is instant

   c. **Generate Token:**
      - Visit: https://huggingface.co/settings/tokens
      - Click "New token" → Name it → Select "Read" permission
      - Copy token (starts with `hf_...`)

   ⚠️ **Without step (b), you will get "403 Access Denied" errors!**

---

## Setup on the Same Machine

If you're running both Speakr and WhisperX ASR on the same machine:

### Step 1: Clone/Create Repository

```bash
# Navigate to where you want the service
cd /path/to/your/projects

# Create the directory
mkdir whisperx-asr-service
cd whisperx-asr-service

# If using Git, initialize
git init
```

### Step 2: Add Files

Copy all the files from this repository into the directory:
- `app/main.py`
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `.env.example`
- `.gitignore`

### Step 3: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit and add your Hugging Face token
nano .env
```

Update `.env`:
```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEVICE=cuda
COMPUTE_TYPE=float16
BATCH_SIZE=16
# Model loaded at startup. Leave empty to load models on first use; requests
# that name no model then use DEFAULT_MODEL, or large-v3.
PRELOAD_MODEL=large-v3
```

### Step 3.5: Pick the Right Image Variant

Two prebuilt Docker images are published per release. They use the same code
but ship different PyTorch wheels:

| Tag | PyTorch | Supported GPUs |
|-----|---------|----------------|
| `:latest` (`learnedmachine/whisperx-asr-service:latest`) | 2.7.1 / cu126 | Pascal (10xx) through Hopper |
| `:blackwell` (`learnedmachine/whisperx-asr-service:blackwell`) | 2.8.0 / cu128 | Blackwell (RTX 50xx) |

If you have an RTX 50xx, switch the `image:` line in `docker-compose.yml` to
the `-blackwell` tag. Everyone else can stick with `:latest`. Pascal users
specifically need `:latest`; PyTorch 2.8 dropped Pascal support and the
Blackwell image will fail with `CUDA error: no kernel image is available
for execution on the device`.

If you build from source, override the build args:

```bash
docker build \
  --build-arg TORCH_VERSION=2.7.1 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
  -t whisperx-asr-service:custom .
```

### Step 4: Build and Start Service

```bash
# Build the Docker image (this will take 10-15 minutes)
docker compose build

# Start the service
docker compose up -d

# Check logs to ensure it's running
docker compose logs -f
```

Look for:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000
```

### Step 5: Test the Service

```bash
# Health check
curl http://localhost:9000/health

# Expected response (shortened):
{
  "status": "healthy",
  "device": "cuda",
  "loaded_models": [],
  "reasons": []
}
```

A 503 response with `"status": "degraded"` lists its causes in `reasons` (low shared memory or disk, or a pipeline stage that does not answer).

### Step 6: Configure Speakr

Update Speakr's `.env` file:

```bash
# Enable ASR endpoint
USE_ASR_ENDPOINT=true

# Point to WhisperX service (using container name if in same Docker network)
ASR_BASE_URL=http://whisperx-asr-api:9000

# Or if not in same network:
# ASR_BASE_URL=http://localhost:9000
```

### Step 7: Restart Speakr

```bash
cd /path/to/speakr
docker compose restart
```

---

## Setup on a Separate GPU Machine

If you have a dedicated GPU machine for ASR processing:

### On the GPU Machine

#### Step 1: Initial Setup

```bash
# SSH into your GPU machine
ssh user@gpu-machine-ip

# Create project directory
mkdir -p ~/whisperx-asr-service
cd ~/whisperx-asr-service
```

#### Step 2: Add Files and Configure

Follow Steps 2-4 from "Setup on the Same Machine" section above.

#### Step 3: Expose Service to Network

Edit `docker-compose.yml` to expose the service:

```yaml
services:
  whisperx-asr:
    # ... existing configuration ...
    ports:
      - "0.0.0.0:9000:9000"  # Changed from "9000:9000" to expose to network
```

#### Step 4: Configure Firewall

```bash
# Allow port 9000 (Ubuntu/Debian with ufw)
sudo ufw allow 9000/tcp

# Or for RHEL/CentOS with firewalld
sudo firewall-cmd --permanent --add-port=9000/tcp
sudo firewall-cmd --reload
```

#### Step 5: Start Service

```bash
docker compose up -d
docker compose logs -f
```

#### Step 6: Test from GPU Machine

```bash
# Get your machine's IP
ip addr show | grep 'inet '

# Test locally
curl http://localhost:9000/health
```

### On the Speakr Machine

#### Step 1: Test Connectivity

```bash
# Replace GPU_MACHINE_IP with actual IP address
curl http://GPU_MACHINE_IP:9000/health
```

If this fails:
- Check firewall settings on GPU machine
- Verify both machines are on the same network or have routing configured
- Check if port 9000 is open

#### Step 2: Update Speakr Configuration

Edit Speakr's `.env` file:

```bash
USE_ASR_ENDPOINT=true
ASR_BASE_URL=http://GPU_MACHINE_IP:9000
```

Replace `GPU_MACHINE_IP` with the actual IP address of your GPU machine.

#### Step 3: Restart Speakr

```bash
cd /path/to/speakr
docker compose restart
```

---

## Configuration

### Asynchronous Jobs

| Variable | Default | Meaning |
|---|---|---|
| `ASYNC_JOBS_ENABLED` | `true` | Offer `POST /jobs` and the rest of the job API |
| `MAX_ASYNC_JOBS` | `16` | Jobs queued or running at once (`0` for no limit); further submits answer 503 |
| `JOB_RESULT_TTL_SECONDS` | `86400` | How long a finished job's result stays fetchable |

A job waits for room in the pipeline without a deadline, because `MAX_ASYNC_JOBS` bounds the backlog. Synchronous requests keep the admission queue bounds. Finished results are held under `JOB_DIR` and removed with the job directory when the TTL passes, so `JOB_DIR` needs room for the transcripts that are kept.

### Performance Tuning

Edit `.env` on the WhisperX service:

#### For High-End GPU (RTX 3080+, A100)
```bash
BATCH_SIZE=32
COMPUTE_TYPE=float16
```

#### For Mid-Range GPU (RTX 3060, RTX 2080)
```bash
BATCH_SIZE=16
COMPUTE_TYPE=float16
```

#### For Low-End GPU (GTX 1660, RTX 2060)
```bash
BATCH_SIZE=8
COMPUTE_TYPE=int8
```

#### For CPU-Only (Not Recommended)

The `BATCH_SIZE` default is now device-aware: 16 on cuda, 2 on cpu. For CPU
runs longer than ~15 minutes drop it to 1 to avoid OOM kills (exit 137).

```bash
DEVICE=cpu
COMPUTE_TYPE=int8
# default on cpu; use 1 for >30 min audio, smaller models
BATCH_SIZE=2
# tiny/base/small are realistic on cpu; large-v3 is not
PRELOAD_MODEL=small
```

### Idle Model Eviction

If you serve multiple Whisper models in rotation and want to reclaim VRAM
between bursts, set `MODEL_KEEP_ALIVE_SECONDS` to the idle window after which
a model should be unloaded:

```bash
# unload models idle for 1 hour
MODEL_KEEP_ALIVE_SECONDS=3600
# sweep cadence (floor 30 seconds)
MODEL_EVICTION_INTERVAL_SECONDS=60
```

Default is `0`, which keeps the previous behaviour (models stay resident
until the process exits). The next request that needs an evicted model
reloads it transparently.

### Model Selection in Speakr

Speakr will use the model specified in its requests. A request that names no model uses `DEFAULT_MODEL`, then `PRELOAD_MODEL`, then `large-v3`, so leaving `PRELOAD_MODEL` empty only disables preloading. The WhisperX service supports:

- `tiny` - Fastest, lowest quality
- `base` - Fast, low quality
- `small` - Good balance of speed and quality
- `medium` - Good quality, slower
- `large-v2` - Excellent quality, slow
- `large-v3` - Best quality, slow

Models are downloaded on first use and cached.

---

## Testing

### Step 1: Basic Health Check

```bash
curl http://localhost:9000/health
# Or remote: curl http://GPU_MACHINE_IP:9000/health
```

Expected response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "loaded_models": []
}
```

### Step 1c: Prometheus Metrics

```bash
curl http://localhost:9000/metrics | head -20
```

Expected: OpenMetrics text starting with `# HELP whisperx_...` lines, not
JSON. Add a Prometheus scrape job for this endpoint to monitor request
throughput, durations, audio sizes, and VRAM usage. See the README's
"Prometheus Metrics" section for the full metric list.

### Step 2: Test Transcription

Create a test audio file or use an existing one:

```bash
# Test with a simple MP3 file
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@test.mp3" \
  -F "language=en" \
  -F "model=small" \
  -F "output_format=json" \
  -F "enable_diarization=false"
```

### Step 3: Test with Diarization

```bash
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@meeting.mp3" \
  -F "language=en" \
  -F "model=small" \
  -F "output_format=json" \
  -F "enable_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

### Step 3.5: Test an Asynchronous Job

A submit returns a job id and the transcription continues on the server, which suits recordings longer than a client's timeout.

```bash
JOB=$(curl -s -X POST "http://localhost:9000/jobs?diarize=false" \
  -F "audio_file=@test.mp3" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")

curl -s "http://localhost:9000/jobs/$JOB" | python3 -m json.tool
curl -s "http://localhost:9000/jobs/$JOB/result" | python3 -m json.tool | head -20
```

The status answer carries `status`, `progress` and `error`. The result answers 202 while the job is unfinished. Set `ASYNC_JOBS_ENABLED=false` to turn the API off.

### Step 4: Test from Speakr

1. Log into Speakr
2. Upload or record an audio file
3. Check Speakr logs for ASR requests:
   ```bash
   docker compose logs -f app | grep ASR
   ```
4. Check WhisperX service logs:
   ```bash
   docker compose logs -f whisperx-asr
   ```

---

## Troubleshooting

### Issue: Service Won't Start

**Check logs:**
```bash
docker compose logs whisperx-asr
```

**Common causes:**
- GPU not accessible: Verify `nvidia-smi` works inside Docker
- Port 9000 already in use: Change port in `docker-compose.yml`
- Invalid HF_TOKEN: Check token and model access agreements

### Issue: Speakr Can't Connect to WhisperX

**From Speakr machine:**
```bash
# Test connectivity
curl -v http://ASR_BASE_URL/health

# Check if port is open (from Speakr machine)
telnet GPU_MACHINE_IP 9000
```

**Solutions:**
- Verify firewall allows port 9000
- Check both machines are on same network
- Try IP address instead of hostname
- Check Docker network configuration

### Issue: Slow Processing

**Check GPU usage:**
```bash
nvidia-smi -l 1
```

**Solutions:**
- Increase BATCH_SIZE if GPU has available memory
- Use smaller model (e.g., `small` instead of `large-v3`)
- Check if GPU is being used (should show in nvidia-smi)
- Disable diarization for faster processing

### Issue: Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce BATCH_SIZE: `BATCH_SIZE=8`
2. Use smaller model in Speakr
3. Use `COMPUTE_TYPE=int8`
4. Close other GPU applications

### Issue: Long Files Fail or the Service Stops Answering

**Symptoms:** several long recordings (hours each) sent at once end in errors such as "object store is full", or the service stops answering until it is restarted.

**Solutions:**
1. Limit the audio in progress and the queue in `.env` (all limits are off by default):
   ```bash
   # Refuse files longer than 8 hours (413)
   MAX_AUDIO_SECONDS=28800
   # At most 6 hours of audio in the pipeline at once; further requests are queued
   MAX_INFLIGHT_AUDIO_SECONDS=21600
   # At most 4 waiting requests and 12 hours of waiting audio, each for at
   # most 30 minutes; beyond that, 503 with Retry-After
   MAX_QUEUED_REQUESTS=4
   MAX_QUEUED_AUDIO_SECONDS=43200
   QUEUE_WAIT_TIMEOUT_SECONDS=1800
   ```
2. Point `JOB_DIR` at a volume with free disk space (about 115 MB per hour of audio, plus the upload).
3. Keep the `restart: unless-stopped` policy. After about three minutes of failed health checks, the container exits and is started again with clean shared memory (`WATCHDOG_*` settings in the README).
4. In Speakr, keep the number of concurrent ASR jobs low, and let a timed-out long job finish on the server before sending it again.

See "Long Files, Overload and Self-Healing" in the README for every setting.

### Issue: Speaker Diarization Fails

**Check:**
1. HF_TOKEN is set correctly
2. Accepted model user agreements on Hugging Face
3. Check logs: `docker compose logs | grep -i diarization`

**Solutions:**
- Look for `"diarization": {"status": "failed", "error": ...}` in the `/asr` JSON response or an `X-Diarization-Status: failed` header; the cause is in the error text
- Verify token: Visit https://huggingface.co/settings/tokens
- Accept agreements again
- Check if pyannote models can download (internet access required on first run)

### Issue: API Returns 500 Error

**Check logs:**
```bash
docker compose logs whisperx-asr | tail -100
```

**Common causes:**
- Invalid audio format
- Model loading failed
- Out of memory
- Missing dependencies

**Solutions:**
- Convert audio to MP3 or WAV
- Restart service: `docker compose restart`
- Check available disk space
- Rebuild image: `docker compose build --no-cache`

---

## Monitoring

### Check Service Status

```bash
# Container status
docker compose ps

# Resource usage
docker stats whisperx-asr-api

# GPU usage
nvidia-smi -l 1
```

### View Logs

```bash
# Real-time logs
docker compose logs -f

# Last 100 lines
docker compose logs --tail=100

# Errors only
docker compose logs | grep -i error
```

### Performance Monitoring

Create a simple monitoring script:

```bash
#!/bin/bash
# monitor.sh

echo "=== WhisperX ASR Service Status ==="
echo "Container Status:"
docker compose ps

echo -e "\n=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

echo -e "\n=== Service Health ==="
curl -s http://localhost:9000/health | json_pp

echo -e "\n=== Recent Logs ==="
docker compose logs --tail=10
```

---

## Security Notes

**This service has NO authentication or security features.**

Basic protection:
```bash
# Restrict access to specific IP (e.g., your Speakr machine)
sudo ufw allow from SPEAKR_IP to any port 9000
sudo ufw deny 9000/tcp

# Store HF_TOKEN in .env file, not hardcoded
```

For internet exposure, you'll need to add your own reverse proxy and authentication.

---

## Next Steps

1. **Monitor Performance:** Watch logs during first few transcriptions
2. **Tune Configuration:** Adjust BATCH_SIZE and model based on your needs
3. **Set Up Backups:** Backup model cache to avoid re-downloading
4. **Configure Alerts:** Set up monitoring for service health
5. **Plan for Scaling:** Consider multiple instances for high volume

---

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review service logs: `docker compose logs`
3. Test with simple audio files first
4. Verify GPU access: `nvidia-smi`
5. Create an issue with logs and configuration details
