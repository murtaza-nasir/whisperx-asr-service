# WhisperX ASR API Service

ASR API powered by WhisperX with speaker diarization, word-level timestamps, and OpenAI-compatible endpoints. Built for self-hosters running [Speakr](https://github.com/murtaza-nasir/speakr) or similar applications.

Source, configuration reference, full API documentation, and changelog: https://github.com/murtaza-nasir/whisperx-asr-service

## Image variants

Two variants are published per release. They share identical application code and differ only in the bundled PyTorch wheel.

| Tag | PyTorch | CUDA wheels | Supported GPUs |
|-----|---------|-------------|----------------|
| `:latest`, `:0.3.2` | 2.7.1 | cu126 | Pascal (10xx) through Hopper |
| `:blackwell`, `:0.3.2-blackwell` | 2.8.0 | cu128 | Blackwell (RTX 50xx) |

If your GPU is an RTX 50xx, pull `:blackwell`. Every other NVIDIA card from 10xx onward should pull `:latest`.

## Quick start

```bash
# Pick the tag that matches your GPU.
# 10xx, 20xx, 30xx, 40xx, A-series, H-series
IMAGE=learnedmachine/whisperx-asr-service:latest
# RTX 50xx
# IMAGE=learnedmachine/whisperx-asr-service:blackwell

docker run -d \
  --name whisperx-asr-api \
  --gpus all \
  -p 9000:9000 \
  -e DEVICE=cuda \
  -e COMPUTE_TYPE=float16 \
  -e BATCH_SIZE=16 \
  -e HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  -e PRELOAD_MODEL=large-v3 \
  -v whisperx-cache:/.cache \
  --restart unless-stopped \
  "$IMAGE"
```

The service listens on port 9000.

```bash
curl http://localhost:9000/health
curl -X POST http://localhost:9000/asr -F "audio_file=@your_audio.mp3"
```

## Hugging Face token (required for diarization)

Speaker diarization needs a Hugging Face token and acceptance of three model agreements:

1. https://huggingface.co/pyannote/speaker-diarization-community-1
2. https://huggingface.co/pyannote/segmentation-3.0
3. https://huggingface.co/pyannote/speaker-diarization-3.1

Generate a read token at https://huggingface.co/settings/tokens and pass it as `HF_TOKEN`.

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /asr` | Primary endpoint compatible with whisper-asr-webservice |
| `POST /v1/audio/transcriptions` | OpenAI-compatible transcription |
| `POST /v1/audio/translations` | OpenAI-compatible translation |
| `GET /v1/models` | List available models |
| `POST /jobs` | Submit a transcription and receive a job id |
| `GET /jobs/{id}` | Job status and progress |
| `GET /jobs` | List jobs (off unless `JOBS_LIST_ENABLED`) |
| `GET /jobs/{id}/result` | Finished transcription, in any output format |
| `DELETE /jobs/{id}` | Cancel a job or drop a finished transcript |
| `GET /health` | Health probe |
| `GET /metrics` | Prometheus OpenMetrics text |
| `GET /queue-metrics` | Legacy JSON queue snapshot |

## Configuration

Common environment variables. Full reference is in the GitHub README.

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `float16` (cuda), `int8` (cpu) | Computation precision |
| `BATCH_SIZE` | `16` (cuda), `2` (cpu) | Larger is faster, uses more memory |
| `HF_TOKEN` | unset | Hugging Face token for diarization |
| `PRELOAD_MODEL` | unset (no preload) | Model to load on startup. The examples set `large-v3`. |
| `DEFAULT_MODEL` | `PRELOAD_MODEL`, else `large-v3` | Model for requests that name none |
| `ALLOWED_MODELS` | unset (all) | Comma-separated models clients may request |
| `MAX_LOADED_MODELS` | `0` (no limit) | Whisper models kept in memory at once |
| `MAX_FILE_SIZE_MB` | `1000` | Reject larger uploads |
| `SERVE_MODE` | `simple` | `simple` (uvicorn) or `ray` (Ray Serve with batching) |
| `MODEL_KEEP_ALIVE_SECONDS` | `0` (disabled) | Unload idle Whisper models after this many seconds |
| `MODEL_EVICTION_INTERVAL_SECONDS` | `60` (floor 30) | Sweep cadence for the eviction daemon |
| `ASYNC_JOBS_ENABLED` | `true` | Offer the `/jobs` API |
| `MAX_ASYNC_JOBS` | `16` | Jobs queued or running at once (`0` for no limit) |
| `JOB_RESULT_TTL_SECONDS` | `86400` | How long a finished job's result stays fetchable |
| `JOBS_LIST_ENABLED` | `false` | Whether `GET /jobs` lists job ids |

A submit to `/jobs` answers with a job id while the transcription continues on the server, which suits recordings that take longer than a client is willing to wait. Identical submits attach to the job already in progress. The service has no authentication, so a job id is the only gate on its transcript.

For multi-GPU, Ray Serve, pipeline strategies, hotwords, OpenAI-compatible parameters, offline use, and the full Prometheus metrics catalogue, see the [GitHub README](https://github.com/murtaza-nasir/whisperx-asr-service#readme).

## Source and license

- Source: https://github.com/murtaza-nasir/whisperx-asr-service
- Issues: https://github.com/murtaza-nasir/whisperx-asr-service/issues
- License: MIT
- WhisperX (BSD-4-Clause): https://github.com/m-bain/whisperX
