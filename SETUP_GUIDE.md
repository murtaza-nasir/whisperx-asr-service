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

3. **Hugging Face Account and Token**
   - Create account: https://huggingface.co/join
   - Generate token: https://huggingface.co/settings/tokens
   - Accept agreements for:
     - https://huggingface.co/pyannote/speaker-diarization-community-1 (latest model - pyannote.audio 4.0)

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

# Expected response:
{
  "status": "healthy",
  "device": "cuda",
  "loaded_models": []
}
```

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
```bash
DEVICE=cpu
COMPUTE_TYPE=int8
BATCH_SIZE=4
```

### Model Selection in Speakr

Speakr will use the model specified in its requests. The WhisperX service supports:

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

### Issue: Speaker Diarization Fails

**Check:**
1. HF_TOKEN is set correctly
2. Accepted model user agreements on Hugging Face
3. Check logs: `docker compose logs | grep -i diarization`

**Solutions:**
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

## Security Recommendations

### For Production Deployment

1. **Use HTTPS with Reverse Proxy**
   ```bash
   # Install nginx
   sudo apt-get install nginx

   # Configure SSL with Let's Encrypt
   sudo certbot --nginx -d asr.yourdomain.com
   ```

2. **Add API Key Authentication**

   Update nginx configuration to check API keys.

3. **Restrict Network Access**
   ```bash
   # Only allow from Speakr machine IP
   sudo ufw allow from SPEAKR_IP to any port 9000
   ```

4. **Use Docker Secrets for HF_TOKEN**

   Instead of .env file, use Docker secrets for production.

5. **Regular Updates**
   ```bash
   # Update WhisperX service monthly
   git pull
   docker compose build --no-cache
   docker compose up -d
   ```

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
