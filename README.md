# WhisperX ASR API Service

[![Version](https://img.shields.io/badge/version-0.3.2-blue.svg)](https://github.com/murtaza-nasir/whisperx-asr-service/releases/tag/v0.3.2)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Docker Build](https://github.com/murtaza-nasir/whisperx-asr-service/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/murtaza-nasir/whisperx-asr-service/actions/workflows/docker-publish.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/learnedmachine/whisperx-asr-service)](https://hub.docker.com/r/learnedmachine/whisperx-asr-service)
[![GPU Required](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/murtaza-nasir/whisperx-asr-service)

**⚠️ Alpha Version - For Self-Hosting Enthusiasts**

A simple ASR API service powered by WhisperX for transcription with speaker diarization. Built for self-hosters running [Speakr](https://github.com/murtaza-nasir/speakr) or similar applications.

> **Two Docker image variants are published per release.**
>
> | Device | Pull this tag |
> |----------|---------------|
> | **RTX 50xx GPU (Blackwell)** | `learnedmachine/whisperx-asr-service:blackwell` (PyTorch 2.8.0 / cu128) |
> | CPU or any other NVIDIA GPU (10xx, 20xx, 30xx, 40xx, A-series, H-series) | `learnedmachine/whisperx-asr-service:latest` (PyTorch 2.7.1 / cu126) |
>
> See [Image Variants](#image-variants) for details.

## What This Does

- Transcribes audio files using OpenAI Whisper models
- Identifies speakers ("Who spoke when") using Pyannote.audio
- Returns word-level timestamps
- Supports 90+ languages
- Outputs JSON, SRT, VTT, TSV formats
- Runs on your own GPU hardware via Docker

## Limitations

- **Not production-grade**: Basic error handling, no authentication
- **GPU required**: Needs an NVIDIA GPU; ~5GB VRAM covers the large models
- **File size limits**: Large audio files (>1GB) can cause out-of-memory errors
- **VRAM usage**: Memory consumption increases with file size and diarization
- **Alpha software**: Expect bugs and breaking changes

## How It Works

```
Audio --> Whisper (transcription) --> Wav2Vec2 (alignment) --> Pyannote (speaker ID) --> Output
```

The service supports two serving modes:

- **Simple mode** (default): Single-process uvicorn with an async GPU queue. Requests are serialized through a semaphore so only one pipeline runs on the GPU at a time. Good for single-GPU, low-traffic, or development use.
- **Ray Serve mode**: Runs on Ray Serve with cross-request batching (`@serve.batch`). Scales from 1 GPU to multi-GPU. Two pipeline strategies are available:
  - **Replicate** (default): Each GPU runs the complete pipeline (Whisper + Align + Diarize). 4 GPUs = 4 independent pipeline replicas, no cross-GPU data transfer.
  - **Split**: Each stage runs as a separate deployment with fractional GPU allocation. Useful when you want independent scaling per stage.

## Prerequisites

### Hardware Requirements

GPU memory requirements vary by model size:

| Whisper Model | VRAM Required (with diarization) | Suitable GPUs |
|---------------|----------------------------------|---------------|
| tiny, base | ~1GB | Any CUDA GPU with 2GB+ |
| small | ~1.5GB | GTX 1660, RTX 2060, or better |
| medium, distil-large-v3.5, large-v3-turbo | ~3GB | RTX 2070, RTX 3060, or better |
| large-v2, large-v3 | ~5GB | RTX 3060 8GB, RTX 2080, **RTX 3090**, or better |

*Note: Measured on an RTX 3090 (float16, batch 16) running transcription +
alignment + pyannote community-1 diarization. Leave headroom above these
figures for concurrent requests and larger batches.*

**Minimum Configuration (small/medium models):**
- GPU: NVIDIA GPU with 6GB VRAM (e.g. RTX 2060) or better
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

## Image Variants

Three prebuilt Docker images are published per release. They differ in the
PyTorch wheel and platform they ship; the application code is identical.

| Tag | PyTorch | CUDA wheels | Platform | Supported GPUs |
|-----|---------|-------------|----------|----------------|
| `:latest`, `:X.Y.Z` | 2.7.1 | cu126 | amd64 | Pascal (10xx) through Hopper. Compatible with the broadest GPU range. |
| `:blackwell`, `:X.Y.Z-blackwell` | 2.8.0 | cu128 | amd64 | Blackwell (RTX 50xx). Drops Pascal/Maxwell support per the PyTorch 2.8 cuDNN/CUDA 12.8 build. |
| `:spark`, `:X.Y.Z-spark` | 2.12.0 | cu130 | arm64 | NVIDIA DGX Spark (GB10, sm_121). CTranslate2 compiled from source. |

If you have an RTX 50xx, use the `-blackwell` tag. On a DGX Spark, use the
`-spark` tag. Everyone else: use `:latest`.

**Tested hardware disclaimer.** I develop and test on RTX 3090 (Ampere)
hardware only. The default `:latest` images are validated end to end on that
hardware before release. The `-blackwell` and `-spark` variants are built
from the same source in CI but I cannot run them; they rely on community
validation. The Spark recipe was contributed and hardware-tested by the
community (PR #23). Please report regressions on these variants in the
issues.

To build a custom variant locally, override the build args:

```bash
docker build \
  --build-arg TORCH_VERSION=2.7.1 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
  -t whisperx-asr-service:custom .
```

## Quick Start (Prebuilt Image)

Get up and running in 3 steps using the prebuilt Docker image:

### 1. Get Hugging Face Token and Model Access

Speaker diarization requires a Hugging Face token and model access:

**a) Create Hugging Face Account:**
- Visit: [https://huggingface.co/join](https://huggingface.co/join) and sign up

**b) Accept Model User Agreements (ALL REQUIRED):**

You need to accept agreements for all three models:
1. [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) - Click "Agree and access repository"
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) - Click "Agree and access repository"
3. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) - Click "Agree and access repository"

**c) Generate Access Token:**
- Visit: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Click "New token", name it (e.g., "whisperx-diarization")
- Select "Read" permission and generate
- Copy the token (starts with `hf_...`)

⚠️ **Important:** Without accepting all model agreements, you'll get "403 Access Denied" errors.

### 2. Create Configuration File

Create a `.env` file with your Hugging Face token:

```bash
# Create .env file
cat > .env << 'EOF'
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEVICE=cuda
COMPUTE_TYPE=float16
BATCH_SIZE=16
PRELOAD_MODEL=large-v3
MAX_FILE_SIZE_MB=1000
# Serve mode: simple (default) or ray (Ray Serve with batching)
SERVE_MODE=simple
EOF
```

Replace `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` with your actual token.

### 3. Run with Docker Compose (Recommended)

Download the docker-compose.yml file and start the service:

```bash
# Download docker-compose.yml
curl -O https://raw.githubusercontent.com/murtaza-nasir/whisperx-asr-service/main/docker-compose.yml

# RTX 50xx (Blackwell) only: switch to the blackwell image
# sed -i 's|whisperx-asr-service:latest|whisperx-asr-service:blackwell|' docker-compose.yml

# Start the service (pulls prebuilt image automatically)
docker compose up -d

# Check logs
docker compose logs -f
```

The bundled `docker-compose.yml` uses `:latest` (PyTorch 2.7.1 / cu126),
which works on every NVIDIA card from Pascal through Hopper. RTX 50xx
users should swap to `:blackwell` (PyTorch 2.8.0 / cu128). See
[Image Variants](#image-variants) for the full matrix.

**Or run with Docker command:**

```bash
# Pick the right tag for your GPU:
# 10xx-40xx, A/H-series
IMAGE=learnedmachine/whisperx-asr-service:latest
# RTX 50xx only
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

The service will be available at `http://localhost:9000`

### 4. Test the Service

```bash
# Health check
curl http://localhost:9000/health

# Test transcription
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@your_audio.mp3" \
  -F "language=en"
```

---

## Build from Source (Advanced)

For development or if you want to build from source:

### 1. Clone Repository

```bash
git clone https://github.com/murtaza-nasir/whisperx-asr-service.git
cd whisperx-asr-service
```

### 2. Set Up Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Hugging Face token
nano .env
```

### 3. Build and Run

**Using docker-compose.dev.yml (with live code mounting):**

```bash
# Build and start
docker compose -f docker-compose.dev.yml up -d --build

# Check logs
docker compose -f docker-compose.dev.yml logs -f
```

**Or build manually:**

```bash
# Build image
docker build -t whisperx-asr-service .

# Run container
docker run -d \
  --name whisperx-asr-api \
  --gpus all \
  -p 9000:9000 \
  --env-file .env \
  -v whisperx-cache:/.cache \
  whisperx-asr-service
```

**Note:** The `docker-compose.dev.yml` file mounts `./app` directory for live code changes without rebuilding.

---

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
| `initial_prompt` | String | None | Context or spelling guide to steer the model |
| `hotwords` | String | None | Comma-separated words to bias transcription toward |
| `output_format` | String | `json` | Output format: `json`, `text`, `srt`, `vtt`, `tsv` |
| `word_timestamps` | Boolean | `true` | Return word-level timestamps |
| `diarize` | Boolean | `true` | Enable speaker diarization |
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
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=5"
```

**Example Request (SRT subtitles):**

```bash
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@video.mp4" \
  -F "language=en" \
  -F "output_format=srt" \
  -F "diarize=false"
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

### Asynchronous Jobs

A submit returns a job id at once and the transcription continues on the server, so a client timeout or a dropped connection does not lose the work. The synchronous endpoints are unchanged; this surface is an addition, and `ASYNC_JOBS_ENABLED=false` removes it.

| Endpoint | Purpose |
|---|---|
| `POST /jobs` | Accept an upload and return a job id. |
| `GET /jobs/{job_id}` | Status and progress. |
| `GET /jobs/{job_id}/result` | The finished transcription, in the requested format. |
| `DELETE /jobs/{job_id}` | Cancel a job, or drop a finished transcript. |
| `GET /jobs` | The jobs this process knows about, newest first. Off unless `JOBS_LIST_ENABLED` is set. |

`POST /jobs` takes the same file field and the same parameters as `/asr`, apart from `output_format`. The raw result is stored and formatted when it is fetched, so one job serves every format.

The upload is received and decoded before the submit answers. A file longer than `MAX_AUDIO_SECONDS` is therefore refused at submit with **413**, audio that cannot be decoded with **400**, and no job is created in either case. Only the wait for the pipeline happens in the background.

**Submit**

```bash
curl -X POST "http://localhost:9000/jobs?diarize=true&language=en" \
  -F "audio_file=@meeting.mp3"
```

```json
{
  "job_id": "9f8c1d0e4b7a42f38e5c6d7a8b9c0d1e",
  "status": "queued",
  "audio_seconds": 7412.4,
  "filename": "meeting.mp3",
  "progress": {"stage": null, "fraction": 0.0, "fraction_estimated": true,
               "elapsed_seconds": 0.0, "eta_seconds": null},
  "error": null,
  "deduplicated": false,
  "submissions": 1,
  "poll_after_seconds": 10,
  "status_url": "/jobs/9f8c1d0e4b7a42f38e5c6d7a8b9c0d1e",
  "result_url": "/jobs/9f8c1d0e4b7a42f38e5c6d7a8b9c0d1e/result"
}
```

The answer is **202** for a new job and **200** for one that already exists (see Deduplication). A `Location` header carries the status URL. When more than `MAX_ASYNC_JOBS` jobs are queued or running, the answer is **503** with `Retry-After`.

**Poll**

```bash
curl "http://localhost:9000/jobs/9f8c1d0e4b7a42f38e5c6d7a8b9c0d1e"
```

Status is one of:

| Status | Meaning |
|---|---|
| `queued` | Accepted, waiting for room in the pipeline. |
| `running` | In a stage. |
| `done` | The result is ready. |
| `error` | The run failed; `error` says why. |
| `cancelled` | Cancelled through `DELETE`. |
| `interrupted` | The service restarted while the job was in flight. Submit it again. |

`progress.stage` is the stage now running (`whisper`, `align`, `diarize`, or `pipeline` with the replicate strategy) and is exact. `progress.fraction` is an estimate from the stage weights and the measured throughput, which is why `fraction_estimated` is reported beside it. It reads `1.0` with `fraction_estimated` false once the job is done.

`poll_after_seconds` is a suggested wait, and the same number is sent as `Retry-After` while a job is unfinished.

**Fetch the result**

```bash
curl "http://localhost:9000/jobs/9f8c1d0e4b7a42f38e5c6d7a8b9c0d1e/result?output_format=srt"
```

The body and the `X-Diarization-Status` header are those of the synchronous endpoints. While the job is unfinished the answer is **202** with the status body and `Retry-After`; for a job that ended without a result it is **409**; for a job that is unknown or has expired it is **404**.

A finished result stays fetchable for `JOB_RESULT_TTL_SECONDS` and is then removed with its directory.

**Errors**

A failed job carries an object, so a client can decide whether to try again:

```json
{"code": "out_of_memory", "message": "CUDA out of memory", "retryable": true}
```

| Code | Retryable |
|---|---|
| `audio_too_long`, `file_too_large`, `decode_failed`, `invalid_request` | No |
| `out_of_memory`, `unavailable`, `stage_failed`, `interrupted`, `internal` | Yes |

**Deduplication**

A submit that matches a job which is queued, running or finished returns that job id with `"deduplicated": true` instead of starting a second run. Two submits match when the transcription parameters are equal and either the uploaded bytes hash to the same value or the same `Idempotency-Key` header was sent:

```bash
curl -X POST "http://localhost:9000/jobs" \
  -H "Idempotency-Key: recording-5182" \
  -F "audio_file=@meeting.mp3"
```

With a key, the key stands in for the content hash, so re-encoding the same recording still attaches to the first job. A key that stays the same while the audio changes also attaches to the first job, so the key has to change when the audio does. Changed parameters always start a new job, so reprocessing with other settings never returns the earlier result. A job that failed, was cancelled or was interrupted is not reused.

**Cancel**

```bash
curl -X DELETE "http://localhost:9000/jobs/9f8c1d0e4b7a42f38e5c6d7a8b9c0d1e"
```

A queued job is dropped, and for a running job the stage call is cancelled, which releases the GPU once the stage returns. A job that finished between the poll and the cancel has its stored transcript removed. The answer carries the job's final state.

**Settings**

| Variable | Default | Meaning |
|---|---|---|
| `ASYNC_JOBS_ENABLED` | `true` | When false, every `/jobs` path answers 404 and the capability is advertised as absent. |
| `MAX_ASYNC_JOBS` | `16` | Jobs queued or running at once (`0` for no limit). Further submits are refused with 503. |
| `JOB_RESULT_TTL_SECONDS` | `JOB_TTL_SECONDS` (86400) | How long a finished result stays fetchable. |
| `JOBS_LIST_ENABLED` | `false` | Whether `GET /jobs` lists job ids. A listing makes the ids enumerable, and an id is the only gate on its transcript. The counts in `/health` carry no ids and are unaffected. |

Admission control still governs the pipeline: a job waits for `MAX_INFLIGHT_AUDIO_SECONDS` without a deadline, because `MAX_ASYNC_JOBS` bounds the backlog instead. Synchronous requests keep the queue bounds and `QUEUE_WAIT_TIMEOUT_SECONDS`.

`MAX_ASYNC_JOBS` also sets the disk the backlog can hold. Decoding happens at submit, so a queued job keeps its decoded audio, about 115 MB per hour of it, until the job ends. At the default of 16 jobs of 8 hours each, that is about 15 GB under `JOB_DIR`. A finished job keeps only its transcript and its state, which are small, until `JOB_RESULT_TTL_SECONDS` passes. `HEALTH_MIN_DISK_FREE_MB` makes `/health` report 503 before the directory fills.

**Detecting the surface**

`/health` and `/` report what this build offers:

```json
{"capabilities": {"async_jobs": true, "jobs_api_version": 1, "dedupe": true,
                  "idempotency_key": true, "progress": true, "cancel": true,
                  "list_jobs": false, "job_result_ttl_seconds": 86400,
                  "max_async_jobs": 16}}
```

`/health` also reports `jobs` with the counts of jobs known, active, queued and running. Queue state for deciding whether to submit is in the existing `admission` block.

### Advanced Speaker Diarization Features

#### Exact Speaker Count

When you know the exact number of speakers, use `num_speakers` for more accurate diarization:

```bash
curl -X POST http://localhost:9000/asr \
  -F "audio_file=@interview.mp3" \
  -F "num_speakers=2" \
  -F "diarize=true"
```

This overrides `min_speakers` and `max_speakers` and typically provides better accuracy than range-based detection.

#### Exclusive Speaker Diarization

This service automatically uses **exclusive speaker diarization** when available from the pyannote community-1 model. This feature simplifies reconciliation between fine-grained speaker diarization timestamps and transcription timestamps, making it ideal for applications like [Speakr](https://github.com/murtaza-nasir/speakr) where you need to align transcripts with speaker segments.

**Benefits:**
- More accurate timestamp alignment between speakers and words
- Better handling of speaker transitions
- Simplified post-processing for multi-speaker transcripts

#### Tuning Diarization Hyperparameters

When speakers are merged into a single label, or short back-and-forth turns are missed, the pyannote community-1 pipeline can be tuned through environment variables. These are unset by default, so the service runs with the model's published defaults unless you opt in. The pipeline's full default parameter schema is logged at startup; for community-1 it is `{'segmentation': {'min_duration_off': 0.0}, 'clustering': {'threshold': 0.6, 'Fa': 0.07, 'Fb': 0.8}}`.

| Variable | Effect | Typical values |
|----------|--------|----------------|
| `DIARIZE_CLUSTERING_THRESHOLD` | The main lever for merged speakers. Lower it to split similar or merged voices more aggressively; raise it for fewer speakers. | `0.4`-`0.8` (default `0.6`; lower = more speakers) |
| `DIARIZE_MIN_DURATION_OFF` | Non-speech gaps shorter than this (seconds) are filled, merging the turns on either side. Raise it to suppress over-segmentation. It does not recover rapid turns, since the default is already `0.0`. | `0.0`-`0.5` (default `0.0`) |
| `DIARIZE_PARAM_OVERRIDES` | Escape hatch: a JSON object deep-merged into the pipeline's instantiated parameters, for any key the variables above do not cover (for example `clustering.Fa`, `clustering.Fb`). | `{"clustering": {"Fb": 1.0}}` |
| `DIARIZE_FILL_NEAREST` | Assign the nearest speaker to words/segments that fall outside every diarization turn, instead of leaving them untagged. Fixes "orphan" segments such as a closing line with no speaker label. | `false` (default), `true` |

```bash
# Split merged speakers (the most common fix); tag any orphan segments
DIARIZE_CLUSTERING_THRESHOLD=0.5
DIARIZE_FILL_NEAREST=true
```

These are global settings applied when the pipeline loads. Invalid or unrecognised keys are logged and ignored rather than failing diarization, so the pipeline always falls back to defaults if an override cannot be applied. To find the best value for your audio without restarting the service, use the sweep script described below.

#### Reducing Text Invented in Silence and Noise

Text such as "Thank you." sometimes appears in quiet stretches of a transcript. It is invented by Whisper from background noise that is passed to the model as speech. Which parts of the audio are passed to Whisper is decided by voice activity detection (VAD), which can be adjusted with the settings below. They are unset by default, so WhisperX's own defaults apply unless you set them. Whisper's `no_speech_threshold`, `log_prob_threshold` and `compression_ratio_threshold` are not applied in the batched WhisperX pipeline, so they cannot be used for this.

| Variable | Effect | Typical values |
|----------|--------|----------------|
| `VAD_METHOD` | The VAD model. | `pyannote` (default), `silero` |
| `VAD_ONSET` | Speech probability above which a stretch is treated as speech. With a higher value, more noise is kept away from Whisper; with too high a value, quiet or distant speech is also lost. | `0.5` (default) to `0.8` |
| `VAD_OFFSET` | Probability below which speech is treated as ended. | `0.363` (default) |

```bash
# Keep a noisy floor in quiet stretches away from Whisper
VAD_ONSET=0.8
# or
VAD_METHOD=silero
```

I tested these settings on speech with 45-second gaps of silence, low room tone, hum, clicks, distant voices, and a noise floor raised by a recorder's automatic gain control. With the defaults, "Thank you." appeared only in the gaps with the raised noise floor. That text was removed with `VAD_ONSET=0.8` and with `VAD_METHOD=silero`, but not with `0.6` or `0.7`. With either setting, about a tenth of the wording elsewhere differs from the default, because the speech is divided into chunks at different points. At `0.8`, a few percent of quiet real speech was also lost, and with `silero` more of the distant voices were transcribed. In silence, low room tone, hum and clicks, no text appeared with any setting.

The settings are read when the model is loaded, so the service must be restarted after a change. Invalid values are logged and ignored. They apply to the Whisper backend only.

#### Sweeping Parameters on Your Own Audio

`tests/diarize_sweep.py` runs diarization on one file across several settings and reports, per config, the number of speakers, turns, and a turn timeline, so you can see which value recovers your missing turns. It isolates the diarizer (no transcription), so it is fast. Run it as a one-off container so it does not disturb a running service:

```bash
docker run --rm --gpus '"device=0"' \
  -v whisperx-cache:/.cache \
  -v "$PWD/tests:/work/tests:ro" -v "$PWD/testfiles:/work/testfiles:ro" \
  -e HF_TOKEN="$HF_TOKEN" -e HF_HOME=/.cache -e DEVICE=cuda \
  learnedmachine/whisperx-asr-service:latest \
  python3 /work/tests/diarize_sweep.py /work/testfiles/your_audio.mp3 --min-speakers 2
```

Pass `--grid "none:none,0.5:0.0,0.45:0.0"` for a custom set of `threshold:min_duration_off` pairs, and `--num-speakers` / `--min-speakers` / `--max-speakers` when the count is known.

For audio that mixes languages within a single file (for example English and Mandarin in one meeting), also consider passing an explicit `language` per request: WhisperX loads one alignment model for the detected language, and poor word-level timestamps from a mismatched alignment model are a common cause of merged or missing speaker turns. Diarization quality is also inherently limited when voices are very similar (for example synthetic or dubbed dialogue).

### Custom Vocabulary (Hotwords)

Whisper often misspells brand names, acronyms, and domain-specific terms. You can improve accuracy using `hotwords` and `initial_prompt`:

- `hotwords` biases the model's beam search to favor specific words during decoding
- `initial_prompt` provides a sentence of context that primes the model to expect certain spellings

**Example:** transcribing audio that says *"We deployed Speakr on a Kubernetes cluster using CTranslate2 for inference. PyAnnote handles the diarization."*

```bash
# Baseline (no hints)
curl -X POST "http://localhost:9000/asr?language=en" \
  -F "audio_file=@meeting.mp3"
# Result: "We deployed Speaker...using ctranslate2...PnNote handles the diarization."

# With hotwords
curl -X POST "http://localhost:9000/asr?language=en&hotwords=Speakr,CTranslate2,PyAnnote" \
  -F "audio_file=@meeting.mp3"
# Result: "We deployed Speaker...using CTranslate2...PyAnnote handles the diarization."

# With hotwords + initial_prompt (best results)
curl -X POST "http://localhost:9000/asr?language=en&hotwords=Speakr,CTranslate2,PyAnnote&initial_prompt=Speakr is a transcription app." \
  -F "audio_file=@meeting.mp3"
# Result: "We deployed Speakr...using CTranslate2...PyAnnote handles the diarization."
```

| Word | No hints | Hotwords | Hotwords + initial_prompt |
|---|---|---|---|
| CTranslate2 | ctranslate2 | CTranslate2 | CTranslate2 |
| PyAnnote | PnNote | PyAnnote | PyAnnote |
| Speakr | Speaker | Speaker | Speakr |
`hotwords` alone fixes most spelling issues. Words that sound identical to common English words (like "Speakr" vs "Speaker") may need `initial_prompt` as well to provide enough context for the model to override its default.

The OpenAI-compatible endpoints (`/v1/audio/transcriptions`) also support a `hotwords` form field. If only `prompt` is provided, it is used as hotwords.

A test script is included to verify hotwords with your own audio:

```bash
./tests/test_hotwords.sh testfiles/your_audio.flac

# Custom hotwords
HOTWORDS="MyBrand,TechTerm" INITIAL_PROMPT="MyBrand is a product." \
  ./tests/test_hotwords.sh testfiles/your_audio.flac
```

## Integration with Speakr

To use this service with [Speakr](https://github.com/murtaza-nasir/speakr) instead of the default ASR endpoint:

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
# cuda for GPU, cpu for CPU-only
DEVICE=cuda

# Computation precision
# float16 (GPU), float32 (CPU), int8 (faster, lower quality)
COMPUTE_TYPE=float16

# Alignment stage device (defaults to DEVICE). cpu keeps the Wav2Vec2
# alignment model off the GPU to reduce VRAM on small cards, at the cost
# of slower word timestamps.
# ALIGN_DEVICE=cpu

# Batch size (higher = faster but more memory). Default is device-aware:
# 16 on cuda, 2 on cpu. Long audio on CPU benefits from BATCH_SIZE=1.
# 16 for 8GB VRAM, 32+ for high-end GPUs, 1-2 on CPU
BATCH_SIZE=16

# Hugging Face token for diarization
HF_TOKEN=hf_xxx...

# Model preloading (optional, reduces first-request latency). Leave empty
# to load models on first use; requests that name no model then use
# DEFAULT_MODEL, or large-v3. See "Model Defaults and Preloading" below.
PRELOAD_MODEL=large-v3
# Model for requests that name none (default: PRELOAD_MODEL, else large-v3)
# DEFAULT_MODEL=large-v3
# Optional limits: models clients may request, and models kept in memory
# ALLOWED_MODELS=large-v3,turbo,distil-large-v3.5
# MAX_LOADED_MODELS=2

# Maximum file size in MB (prevents out-of-memory errors)
# Default 1GB, adjust lower for GPUs with <16GB VRAM
MAX_FILE_SIZE_MB=1000

# Long files and overload (all 0 = off). See "Long Files, Overload and
# Self-Healing" below.
# MAX_AUDIO_SECONDS=0
# MAX_INFLIGHT_AUDIO_SECONDS=0
# MAX_QUEUED_AUDIO_SECONDS=0
# MAX_QUEUED_REQUESTS=0
# QUEUE_WAIT_TIMEOUT_SECONDS=0

# Asynchronous jobs (POST /jobs). See "Asynchronous Jobs" above.
# ASYNC_JOBS_ENABLED=true
# MAX_ASYNC_JOBS=16
# JOB_RESULT_TTL_SECONDS=86400

# Idle model eviction (default disabled). When > 0, Whisper models that have
# not served a request in this many seconds are unloaded from memory by a
# background sweep. The next request that needs the model will reload it.
# 0 disables eviction; e.g. 3600 = 1 hour
MODEL_KEEP_ALIVE_SECONDS=0
# Sweep frequency (floor of 30 seconds)
MODEL_EVICTION_INTERVAL_SECONDS=60

# Diarization hyperparameter tuning (optional, unset = model defaults).
# See "Tuning Diarization Hyperparameters" above for details.
# default 0.6; lower = split merged voices
# DIARIZE_CLUSTERING_THRESHOLD=0.5
# default 0.0; raise = reduce over-segmentation
# DIARIZE_MIN_DURATION_OFF=0.2
# DIARIZE_PARAM_OVERRIDES={"clustering": {"Fb": 1.0}}  # JSON escape hatch (Fa/Fb)
# true = tag orphan segments with nearest speaker
# DIARIZE_FILL_NEAREST=false

# Voice activity detection (optional, unset = WhisperX defaults).
# See "Reducing Text Invented in Silence and Noise" above.
# default 0.5; 0.8 keeps a noisy floor away from Whisper
# VAD_ONSET=0.8
# default pyannote; silero is the alternative
# VAD_METHOD=silero
# VAD_OFFSET=0.363

# Rebuild segments at speaker-change boundaries after diarization, so rapid
# turns are not merged into one speaker's segment. Always on for the qwen3
# backend; opt-in here for the whisper backend (changes segment shape
# compared to previous releases).
# RESEGMENT_BY_SPEAKER=true

# Experimental Qwen3-ASR backend (see "Qwen3-ASR Backend" below).
# ASR_BACKEND=qwen3
# QWEN3_ASR_MODEL=Qwen/Qwen3-ASR-1.7B-hf
# QWEN3_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B-hf
# QWEN3_CHUNK_SECONDS=90
# standing vocabulary/context biasing text
# QWEN3_DEFAULT_CONTEXT=
```

### Model Defaults and Preloading

Three settings decide which Whisper model handles a request:

- `PRELOAD_MODEL` loads a model at startup, so the first request does not wait for it. Leave it empty to load models only when a request needs them, for example on a GPU shared with other services. It is skipped on the `qwen3` and `external` backends, which do not transcribe with a Whisper model.
- `DEFAULT_MODEL` is the model used when a request does not name one. If it is unset, `PRELOAD_MODEL` is used, and if that is also empty, `large-v3`. An empty `PRELOAD_MODEL` therefore disables preloading without leaving such requests without a model. An unusable value falls back to `large-v3` with a warning in the log.
- `OPENAI_WHISPER1_MODEL` sets the model behind the OpenAI alias `whisper-1` (default: the default model).

A request may name any model that `/v1/models` lists, an OpenAI-style alias (`whisper-1`, `whisper-large-v3`, ...), a Hugging Face id (`org/repo`) or a local model directory. Any other name is rejected with HTTP 400 and the list of accepted names.

Two optional limits protect the GPU on shared servers:

- `ALLOWED_MODELS` (comma-separated) restricts the models clients may request. The default model is always allowed, and `/v1/models` lists only the allowed models.
- `MAX_LOADED_MODELS` caps the number of Whisper models kept in memory. Loading another model first unloads the least recently used model that no request is using. It combines with `MODEL_KEEP_ALIVE_SECONDS`, which unloads models after a period without use.

The startup log shows the effective settings, for example `Models: default model: large-v3, preload: none`.

### Serve Mode

Controlled by the `SERVE_MODE` environment variable in your `.env` file.

#### Simple Mode (default)

```bash
SERVE_MODE=simple
```

Runs uvicorn directly. Requests are serialized through an async GPU semaphore so the event loop stays responsive while GPU work runs in a thread pool. This is backward-compatible with previous versions.

You can tune `GPU_CONCURRENCY=1` (default) to control how many pipeline runs execute concurrently. Leave at 1 for single-GPU setups.

#### Ray Serve Mode

```bash
SERVE_MODE=ray
```

Runs on Ray Serve with cross-request batching (`@serve.batch`). Two pipeline strategies are available, controlled by `PIPELINE_STRATEGY`:

##### Replicate Strategy (default, recommended)

```bash
PIPELINE_STRATEGY=replicate
# one full pipeline per GPU
NUM_GPU_REPLICAS=4
```

Each GPU runs the complete 3-stage pipeline. Ray Serve routes incoming requests across replicas.

```
                    +-- GPU 0: [Whisper + Align + Diarize] --+
HTTP --> Proxy -->  +-- GPU 1: [Whisper + Align + Diarize] --+--> Response
                    +-- GPU 2: [Whisper + Align + Diarize] --+
                    +-- GPU 3: [Whisper + Align + Diarize] --+
```

##### Split Strategy

```bash
PIPELINE_STRATEGY=split
```

Each pipeline stage runs as a separate deployment with fractional GPU allocation and cross-request batching (`@serve.batch`). Stage-level pipelining means request B can start transcription while request A is still in diarization.

```
HTTP --> Proxy --> ASR Ingress
                       |
              +--------+--------+
              |        |        |
          Whisper   Align    Diarize
         (GPU 0.5) (GPU 0.3) (GPU 0.2)
```

##### Strategy Comparison

Both strategies achieve similar throughput (~6.3-6.5x speedup on 4x RTX 3090 with 8 concurrent workers).

| | Replicate | Split |
|---|-----------|-------|
| **Configuration** | Simple (just set `NUM_GPU_REPLICAS`) | Complex (GPU fractions, per-stage replicas, bin-packing tuning) |
| **Cross-GPU transfer** | None (audio stays on one GPU) | Yes (results move between stages) |
| **Tail latency** | Higher variance | Lower and more consistent |
| **Scaling** | Add a GPU, add a replica | Scale bottleneck stages independently (e.g. more Whisper replicas) |
| **VRAM per GPU** | Must fit all 3 models | Each stage uses a fraction |
| **Best for** | Most setups, multi-GPU throughput | Advanced tuning, heterogeneous stage scaling |

##### Ray Serve Configuration

All optional, shown with defaults:

```bash
# Pipeline strategy: replicate (full pipeline per GPU) or split (stage per GPU)
PIPELINE_STRATEGY=replicate

# Number of pipeline replicas (set to number of GPUs)
NUM_GPU_REPLICAS=1

# Cross-request batch sizes per stage (tune for GPU VRAM)
WHISPER_BATCH_SIZE=4
ALIGN_BATCH_SIZE=8
DIARIZE_BATCH_SIZE=2

# Seconds to wait collecting a batch before processing what's available
BATCH_WAIT_TIMEOUT=0.1

# Split strategy only: fractional GPU allocation per stage
WHISPER_GPU_FRACTION=0.5
ALIGN_GPU_FRACTION=0.3
DIARIZE_GPU_FRACTION=0.2

# Split strategy only: per-stage replica overrides (fall back to NUM_GPU_REPLICAS)
# WHISPER_NUM_REPLICAS=2
# ALIGN_NUM_REPLICAS=1
# DIARIZE_NUM_REPLICAS=1
```

##### Multi-GPU Examples

```bash
# 4 GPUs, full pipeline on each (recommended)
SERVE_MODE=ray
PIPELINE_STRATEGY=replicate
NUM_GPU_REPLICAS=4

# 3 GPUs, one stage per GPU
SERVE_MODE=ray
PIPELINE_STRATEGY=split
WHISPER_GPU_FRACTION=1.0
ALIGN_GPU_FRACTION=1.0
DIARIZE_GPU_FRACTION=1.0

# 4 GPUs, hybrid: 2 Whisper + 1 Align + 1 Diarize
SERVE_MODE=ray
PIPELINE_STRATEGY=split
WHISPER_NUM_REPLICAS=2  WHISPER_GPU_FRACTION=1.0
ALIGN_NUM_REPLICAS=1    ALIGN_GPU_FRACTION=1.0
DIARIZE_NUM_REPLICAS=1  DIARIZE_GPU_FRACTION=1.0
```

When running in Ray mode, the Ray Dashboard is available at `http://localhost:8265` on the Docker host for monitoring deployments, replicas, and request metrics.

##### Ray Dashboard Access

Jobs submitted through the Ray Dashboard are executed as code inside the container. For this reason, the dashboard is not reachable by default: it listens only inside the container. To use it, set `RAY_DASHBOARD_HOST=0.0.0.0` in `.env` and recreate the container. Port 8265 is published on `127.0.0.1` only in the Compose files, so the dashboard can then be opened at `http://localhost:8265` on the Docker host. To open it from another machine, use an SSH tunnel to the Docker host:

```bash
ssh -L 8265:localhost:8265 user@docker-host
# then open http://localhost:8265 in a local browser
```

Ray token authentication is enabled by default, and the token is requested when the dashboard is opened. A new token is generated each time the container starts; it can be read with:

```bash
docker exec whisperx-asr-api cat /root/.ray/auth_token
```

`RAY_DASHBOARD_PORT` changes the host port. Publishing the dashboard on all interfaces (for example `"8265:8265"` in an override file) is not recommended unless access is also restricted by a firewall or an authenticating reverse proxy.

##### GPU Pinning

To restrict which GPUs the service uses, create a `docker-compose.dev.local.yml` override (gitignored):

```yaml
services:
  whisperx-asr:
    environment:
      - CUDA_VISIBLE_DEVICES=0,1,2,3
      - NUM_GPU_REPLICAS=4
```

Use `CUDA_VISIBLE_DEVICES` (not `NVIDIA_VISIBLE_DEVICES`) since the Docker Compose `deploy` section exposes all GPUs to the container.

Then start with:

```bash
docker compose -f docker-compose.dev.yml -f docker-compose.dev.local.yml up -d
```

### Long Files, Overload and Self-Healing

Each request gets a working directory under `JOB_DIR`. The upload is copied there in 1 MB chunks, and `MAX_FILE_SIZE_MB` is checked during the copy (and from the `Content-Length` header, before the body is read). The file is then decoded once, in a background ffmpeg process, to 16 kHz mono 16-bit PCM. Each pipeline stage reads that file; the samples are identical to `whisperx.load_audio()` on the original upload. In Ray Serve mode, the audio therefore never enters the Ray object store, and the size of `/dev/shm` no longer limits how much audio can be in progress. The directory is removed when the request ends, also after errors and client disconnects. Directories and temp uploads left over from a crash are removed at container start.

#### Admission Control

The audio in progress can be limited by duration. All limits are off by default, so existing setups behave as before.

| Variable | Default | Meaning |
|---|---|---|
| `MAX_AUDIO_SECONDS` | `0` (off) | Longest file accepted. A longer file gets **413** with `"audio too long"` (OpenAI endpoints: code `audio_too_long`). |
| `MAX_INFLIGHT_AUDIO_SECONDS` | `0` (off) | Total audio seconds in the pipeline at once. Further requests are held in a first-in, first-out queue. A single file longer than this limit is still admitted when nothing else is in progress. |
| `MAX_QUEUED_REQUESTS` | `0` (no limit) | Requests allowed to wait. |
| `MAX_QUEUED_AUDIO_SECONDS` | `0` (no limit) | Audio seconds allowed to wait. |
| `QUEUE_WAIT_TIMEOUT_SECONDS` | `0` (no limit) | Longest wait for admission. |

When the queue is full or the wait times out, the response is **503** with a `Retry-After` header (an estimate from the measured throughput, 30 to 3600 seconds) and a JSON body:

```json
{"error": "busy", "reason": "queue_full", "retry_after": 240,
 "queue": {"inflight_requests": 2, "inflight_audio_seconds": 21600.0,
           "queued_requests": 4, "queued_audio_seconds": 30000.0,
           "estimated_wait_seconds": 240, "admission_rejections_total": 7}}
```

Wait for `Retry-After` before sending the file again. After giving up on a long request, do not resend it at once: work already started on the GPU is not interrupted.

The duration used for admission is read with `ffprobe` before decoding, so a rejected file is never decoded. The limits apply per server process (simple mode) or per ingress replica (Ray Serve mode, one replica).

#### Health and Restarts

`GET /health` is **200** (`"status": "healthy"`) when the service can take work and **503** (`"status": "degraded"`, with a `reasons` list) when:

- free space on `/dev/shm` or on `RAY_TMPDIR` is below `HEALTH_MIN_SHM_FREE_PCT` percent (Ray Serve mode),
- free disk space at `JOB_DIR` is below `HEALTH_MIN_DISK_FREE_MB`,
- a pipeline stage gives no answer to a ping within `HEALTH_STAGE_TIMEOUT_SECONDS` (Ray Serve mode). A stage whose request slots are all busy is not pinged and counts as working.

A full admission queue is reported under `admission` (`"saturated": true`), but it is not a reason for 503. `GET /` stays a plain liveness check.

The Docker healthcheck fails on any status other than 200. An `unhealthy` container is never restarted by Docker, so the service is run under a small watchdog (`app/watchdog.py`). `/health` is polled every `WATCHDOG_INTERVAL_SECONDS`. After `WATCHDOG_FAILURES` consecutive failed checks, or when `/dev/shm` stays above `WATCHDOG_SHM_MAX_USED_PCT` for `WATCHDOG_SHM_FAILURES` checks, the service is stopped and the container exits with status 1. With `restart: unless-stopped` (as in the Compose files) or `--restart unless-stopped`, a fresh container is then started, with an empty `/dev/shm`. Failed checks count only after the first healthy answer, so a long first start (model downloads) cannot cause a restart. `docker stop` works as before.

| Variable | Default | Meaning |
|---|---|---|
| `JOB_DIR` | `/tmp/whisperx-jobs` | Per-request working files. Point it at a volume with free space for long files (about 115 MB per hour of audio, plus the upload). |
| `JOB_TTL_SECONDS` | `86400` | Age after which a leftover job directory is removed at startup. |
| `HEALTH_MIN_SHM_FREE_PCT` | `5` | 0 turns the check off. |
| `HEALTH_MIN_DISK_FREE_MB` | `500` | 0 turns the check off. |
| `HEALTH_STAGE_TIMEOUT_SECONDS` | `5` | Ping timeout per stage. |
| `WATCHDOG_ENABLED` | `true` | `false` keeps only signal forwarding. |
| `WATCHDOG_INTERVAL_SECONDS` | `30` | Time between checks. |
| `WATCHDOG_FAILURES` | `6` | Consecutive failed checks before exit (about 3 minutes). |
| `WATCHDOG_TIMEOUT_SECONDS` | `10` | Timeout of one check. |
| `WATCHDOG_SHM_MAX_USED_PCT` | `98` | `/dev/shm` use that counts as full. |
| `WATCHDOG_SHM_FAILURES` | `4` | Consecutive checks with full `/dev/shm` before exit. |
| `DIARIZE_ON_ERROR` | `flag` | `flag`: after a failed diarization, the transcript is still returned, without speakers, with `"diarization": {"status": "failed", "error": ...}` in the JSON and an `X-Diarization-Status: failed` header. `fail`: the request ends with a 500 error. |

When diarization is requested, the JSON response of `/asr` has a `diarization` field (`"status"` is `ok`, `failed` or `unavailable` when `HF_TOKEN` is not set), and every output format has the `X-Diarization-Status` header. A CUDA out-of-memory error during diarization is retried once after the GPU cache is freed.

#### Ray Serve Memory Layout

In Ray Serve mode, only the Ray object store is in `/dev/shm`. Ray's session files and logs are in `RAY_TMPDIR` (a 2 GB tmpfs at `/ray-tmp` in the Compose files), with log rotation at `RAY_ROTATION_MAX_BYTES` (50 MB) and `RAY_ROTATION_BACKUP_COUNT` (1). Objects spill to `RAY_SPILL_DIR` on disk (`/.cache/ray-spill`, in the cache volume). A full log directory therefore cannot fill shared memory, and spilling frees shared memory.

| Variable | Default | Meaning |
|---|---|---|
| `RAY_OBJECT_STORE_MEMORY` | `4000000000` | Object store size in bytes; keep it well below `shm_size`. |
| `RAY_TMPDIR` | `/ray-tmp` | Ray session files and logs. Without the tmpfs mount, the directory is on the container's disk. |
| `RAY_SPILL_DIR` | `/.cache/ray-spill` | Object spilling target. |
| `RAY_ROTATION_MAX_BYTES` | `52428800` | Size of one Ray log file before rotation. |
| `RAY_ROTATION_BACKUP_COUNT` | `1` | Rotated copies kept per log file. |
| `STAGE_MAX_ONGOING_REQUESTS` | `2` | Requests each split-strategy stage replica accepts at once. |
| `PIPELINE_MAX_ONGOING_REQUESTS` | `5` | The same for replicate-strategy replicas. |

In each stage replica, one item at a time is processed in a worker thread. Replica health checks and pings are therefore answered during long calls, and a long alignment or diarization is not cut short by a replica restart.

If you run `docker run` without Compose, add `--shm-size=8g --tmpfs /ray-tmp:size=2g,exec --restart unless-stopped` for Ray Serve mode.

### Model Selection

Available Whisper models (speed vs accuracy tradeoff):

| Model | Parameters | VRAM (loaded) | VRAM (full pipeline*) | Speed | Quality |
|-------|------------|---------------|----------------------|-------|---------|
| `tiny` | 39M | 0.5GB | 1.0GB | Fastest | Lowest |
| `base` | 74M | 0.6GB | 1.1GB | Very Fast | Low |
| `small` | 244M | 0.9GB | 1.5GB | Fast | Medium |
| `medium` | 769M | 2.2GB | 2.8GB | Moderate | Good |
| `distil-large-v3.5` | 756M | 2.2GB | 2.8GB | Fast | Near-best |
| `large-v3-turbo` | 809M | 2.3GB | 2.9GB | Fast | Near-best |
| `large-v2` | 1550M | 3.9GB | 4.6GB | Slow | Excellent |
| `large-v3` | 1550M | 3.9GB | 4.6GB | Slow | Best |

*Measured on an RTX 3090 (float16, `BATCH_SIZE=16`, 12-minute file), running
transcription, alignment, and pyannote diarization. Peak VRAM occurs during
transcription; alignment and diarization add little on top. Actual peaks vary
somewhat with audio content and batch size.

**Recommendation:**
- Use `large-v3` for best quality (fits comfortably in 6GB+ VRAM)
- Use `distil-large-v3.5` or `large-v3-turbo` for near-large quality at
  about half the VRAM and higher speed
- Use `small` or `medium` for tight VRAM budgets (2-3GB)

### Qwen3-ASR Backend (Experimental)

Setting `ASR_BACKEND=qwen3` replaces the transcription and alignment stages
with Qwen3-ASR (`Qwen/Qwen3-ASR-1.7B-hf`) and the Qwen3 forced aligner
(`Qwen/Qwen3-ForcedAligner-0.6B-hf`), loaded through stock transformers.
Diarization is unchanged.

Why use it:

- The forced aligner is a single model covering zh, en, yue, fr, de, it,
  ja, ko, pt, ru, and es. Code-switched audio (for example Chinese and
  English in one recording) is transcribed and word-aligned in one pass,
  which the per-language Wav2Vec2 alignment stage cannot do.
- `hotwords` and `initial_prompt` become free-text context biasing, which
  corrects domain terms more reliably than token-level hotwords.

Why not:

- Roughly 10x slower than faster-whisper (about 12x realtime vs 116x on an
  RTX 3090). The full pipeline uses about 6 GB VRAM with the default 1.7B
  model (vs 4.6 GB for whisper large-v3), or about 3.6 GB with
  `QWEN3_ASR_MODEL=Qwen/Qwen3-ASR-0.6B-hf`.
- The requested Whisper model name is ignored and `task=translate` falls
  back to the whisper backend.
- Audio is processed in 90-second chunks (`QWEN3_CHUNK_SECONDS`); words
  spanning a chunk boundary may be split.

Weights (about 4.6 GB total) download to the cache volume on first use, the
same as every other model. Nothing downloads unless the backend is enabled.

```bash
ASR_BACKEND=qwen3
# Optional overrides (defaults shown):
# QWEN3_ASR_MODEL=Qwen/Qwen3-ASR-1.7B-hf
# QWEN3_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B-hf
# QWEN3_CHUNK_SECONDS=90
# standing vocabulary/context biasing text
# QWEN3_DEFAULT_CONTEXT=
```

**Code-switched audio requires a per-request `language`.** Passing any one
of the mixed languages makes the backend force verbatim transcription (the
other languages are still transcribed in their own script). Without a
language, the model picks the dominant language per chunk and translates
the rest into it; context hints do not prevent this. Speaker segments are
always rebuilt at speaker-change boundaries on this backend (see
`RESEGMENT_BY_SPEAKER` below for the whisper backend).

### External ASR Backend (Experimental)

Setting `ASR_BACKEND=external` outsources only the transcription stage to
an OpenAI-compatible API; alignment, diarization, speaker embeddings, and
voice profiles keep running locally. Be aware that your audio is uploaded
to the configured provider, which is why this is strictly opt-in.

```bash
ASR_BACKEND=external
EXTERNAL_ASR_BASE_URL=https://api.openai.com/v1
EXTERNAL_ASR_API_KEY=sk-...
EXTERNAL_ASR_MODEL=whisper-1
# default; or "chat" for audio-input
# EXTERNAL_ASR_MODE=transcriptions
                                     # chat models (OpenRouter, vLLM)
```

How word timestamps are produced depends on the provider response.
Providers that return timestamped segments (whisper-1 `verbose_json`,
Groq, self-hosted Whisper servers) feed the existing Wav2Vec2 alignment
stage directly. Text-only providers (gpt-4o-transcribe, Voxtral or other
audio models through OpenRouter's chat API, VibeVoice via vLLM) are
word-timestamped by the Qwen forced aligner instead, which downloads on
first use (~1.2 GB) and adds about 1 GB of VRAM. `task=translate` falls
back to the whisper backend, and the requested Whisper model name is
ignored.

Set `EXTERNAL_ASR_ALIGNER=qwen` to force the Qwen forced aligner even for
timestamped responses; it is language-agnostic across its supported set
and handles code-switched audio, unlike the per-language Wav2Vec2 models.
The reverse is not configurable: Wav2Vec2 requires segment timestamps, so
text-only responses always use the Qwen aligner.

## Running the Service

```bash
# Start in simple mode (default)
docker compose up -d

# Or start in Ray Serve mode
# Set SERVE_MODE=ray in your .env file, then:
docker compose up -d

# View logs
docker compose logs -f
```

**Note:** When using Ray Serve mode, Docker Compose is configured with `shm_size: 8g` for Ray's shared memory object store. The Ray Dashboard is off by default and, when enabled, published on `127.0.0.1:8265` only (see [Ray Dashboard Access](#ray-dashboard-access)).

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

# Response (shortened):
{
  "status": "healthy",
  "device": "cuda",
  "loaded_models": ["large-v3"],
  "serve_mode": "ray",
  "reasons": [],
  "stages": {"whisper": {"ok": true, "busy": false}, "align": {"ok": true}, "diarize": {"ok": true}},
  "resources": {"/dev/shm": {"free_pct": 99.9}, "/tmp/whisperx-jobs": {"free_mb": 51200.0}},
  "admission": {"inflight_requests": 0, "queued_requests": 0, "saturated": false}
}
```

The status code is 503 (`"status": "degraded"`) when shared memory or disk is low or a stage does not answer; `reasons` lists why. See [Health and Restarts](#health-and-restarts) for the thresholds and the watchdog that restarts an unhealthy container.

### Prometheus Metrics

`GET /metrics` returns OpenMetrics text format suitable for a Prometheus
scrape config:

```bash
curl http://localhost:9000/metrics
# # HELP whisperx_requests_total Total HTTP requests by endpoint and status
# # TYPE whisperx_requests_total counter
# whisperx_requests_total{endpoint="/asr",status="ok"} 12.0
# whisperx_request_duration_seconds_bucket{endpoint="/asr",le="60.0"} 11.0
# ...
```

| Metric | Type | Notes |
|--------|------|-------|
| `whisperx_requests_total{endpoint,status}` | Counter | `status` is `ok`, `http_<code>`, or `error` |
| `whisperx_request_duration_seconds{endpoint}` | Histogram | End-to-end handler time |
| `whisperx_active_transcriptions` | Gauge | In-flight `/asr` requests |
| `whisperx_loaded_models` | Gauge | Whisper models currently in cache |
| `whisperx_model_evictions_total{model}` | Counter | Models unloaded by the idle-eviction sweep |
| `whisperx_audio_duration_seconds` | Histogram | Submitted audio duration |
| `whisperx_audio_size_megabytes` | Histogram | Submitted file size |
| `whisperx_vram_allocated_bytes` | Gauge | CUDA `memory_allocated()` (0 on CPU) |
| `whisperx_service_info` | Info | Static labels: version, device, compute_type, serve_mode |

**Ray Serve caveat:** `/metrics` is served by the ingress process. Whisper
models load inside replica processes, so `whisperx_loaded_models`,
`whisperx_vram_allocated_bytes`, and `whisperx_model_evictions_total` will
read 0 (or be missing) in Ray Serve mode. These counters and gauges are
per-process; without `prometheus_client` multi-process mode the ingress'
registry never sees the replica's events. HTTP-level metrics (request
counts, durations, audio sizes) are accurate in both modes because they
are recorded inside the ingress handler. Use the Ray Dashboard (`http://localhost:8265`
on the Docker host) for per-replica state, and tail `serve/replica_*` log files for
eviction events. Multi-process metrics support is a planned follow-up.

In simple mode (`SERVE_MODE=simple`) all metrics including the eviction
counter and loaded-model gauge work as expected, since the model cache
and the `/metrics` handler share a process. Each Whisper model load
pre-registers a `whisperx_model_evictions_total{model="<name>"} 0` row
so dashboards can graph the metric from the first load onward, not only
after the first eviction.

The legacy JSON shape (queue/loaded model state) is still available at
`GET /queue-metrics` for callers that depended on it.

### Performance Monitoring

Monitor GPU usage:

```bash
# NVIDIA GPU stats
nvidia-smi -l 1

# Docker container stats
docker stats whisperx-asr-api

# Ray Dashboard (Ray mode only, localhost on the Docker host)
# Open http://localhost:8265 in a browser on the Docker host,
# or through an SSH tunnel: ssh -L 8265:localhost:8265 user@docker-host
```

## Offline Use

This service can run completely offline after an initial setup with internet access. This is useful for air-gapped environments or when you want to avoid network latency.

### Initial Setup (requires internet)

1. Start the container with internet access
2. Run at least one transcription request with diarization enabled to cache all models:
   ```bash
   curl -X POST http://localhost:9000/asr \
     -F "audio_file=@test.mp3" \
     -F "diarize=true"
   ```
3. This downloads and caches:
   - Whisper model (e.g., large-v3)
   - Alignment model (wav2vec2)
   - Pyannote speaker diarization models

### Enable Offline Mode

Add `HF_HUB_OFFLINE=1` to your `docker-compose.yml` environment section:

```yaml
environment:
  - HF_HUB_OFFLINE=1
  # ... other environment variables
```

**Important:** This must be set directly in `docker-compose.yml`, not in the `.env` file.

Then restart the container:
```bash
docker compose down && docker compose up -d
```

The service will now operate without any network requests to Hugging Face.

### Air-Gapped Transfer (no internet on the target machine)

If the machine that runs the service has no internet at all, populate the cache on an internet-connected machine and move it across as data. Do not rely on `docker commit`: the cache lives in a named Docker volume (`whisperx-cache:/.cache`), and `docker commit` captures only the container filesystem, not volume contents, so a committed-and-exported image arrives with an empty cache and fails to find the models at startup (the Swagger UI will also fail to load).

1. On the online machine, run one transcription with diarization (see above) so every model is cached.
2. Export the cache volume to a tarball:
   ```bash
   docker run --rm -v whisperx-cache:/cache -v "$PWD":/out alpine \
     tar czf /out/whisperx-cache.tgz -C /cache .
   ```
3. Transfer `whisperx-cache.tgz` and the image (`docker save IMAGE | gzip > image.tgz`) to the target machine.
4. On the target machine, load the image, extract the cache, and bind-mount it instead of a named volume:
   ```bash
   mkdir -p /srv/whisperx-cache
   tar xzf whisperx-cache.tgz -C /srv/whisperx-cache
   docker run -d --gpus all -p 9000:9000 \
     -e HF_HUB_OFFLINE=1 \
     -v /srv/whisperx-cache:/.cache \
     learnedmachine/whisperx-asr-service:latest
   ```
   On RTX 50xx / Blackwell cards use the `:0.3.2-blackwell` image and keep `COMPUTE_TYPE=float16` (the CTranslate2 INT8 path is not supported on Blackwell).

### What Gets Cached

| Component | Cache Location | Notes |
|-----------|---------------|-------|
| Whisper models | `/.cache/models--Systran--faster-whisper-*` | Downloaded on first use |
| Alignment model | `/.cache/wav2vec2_*.pth` | Downloaded on first alignment |
| Pyannote models | `/.cache/hub/models--pyannote--*` | Downloaded on first diarization |
| NLTK tokenizers | `/.cache/nltk_data/` | Pre-downloaded in Docker image |

### Troubleshooting Offline Mode

If you see errors like `Failed to resolve 'huggingface.co'`:
1. Ensure you ran a full transcription with diarization while online
2. Verify `HF_HUB_OFFLINE=1` is set in `docker-compose.yml` (not `.env`)
3. Check the cache volume contains the models: `docker exec whisperx-asr-api ls -la /.cache/` (faster-whisper models) and `docker exec whisperx-asr-api ls -la /.cache/hub/` (pyannote/HuggingFace models)

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
4. Ensure `diarize=true` in request

### Slow Processing

**Symptom:** Transcription takes too long

**Solutions:**
1. Use GPU instead of CPU (`DEVICE=cuda`)
2. Use smaller model for faster processing
3. Increase `BATCH_SIZE` (if you have VRAM)
4. Disable diarization if not needed: `diarize=false`

### PyTorch 2.6 Weights Loading Error

**Symptom:** Error message containing `Weights only load failed` or `GLOBAL omegaconf.listconfig.ListConfig was not an allowed global`

This occurs due to a security change in PyTorch 2.6 where `weights_only=True` became the default for `torch.load()`.

**Solution:**

Add this environment variable to your `docker-compose.yml`:

```yaml
environment:
  - TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
```

**Important:** Setting this in `.env` file alone may not work - it must be set directly in `docker-compose.yml` under the `environment` section.

See [WhisperX Issue #1304](https://github.com/m-bain/whisperX/issues/1304) for more details.

### API Returns 503 Errors

A 503 with a `Retry-After` header and `"error": "busy"` means the admission queue is full ([Admission Control](#admission-control)); retry after the given number of seconds. A 503 with `"error": "unavailable"` means the Ray object store, the disk or a stage replica failed for that request; the request can be retried, and `GET /health` shows whether the service is degraded. A container that stays degraded is restarted (see [Health and Restarts](#health-and-restarts)).

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

Because there is no authentication, a job id is the only thing standing between a request and the transcript it belongs to. Ids are 128 random bits. `GET /jobs` would make them enumerable, so that listing is off unless `JOBS_LIST_ENABLED` is set; the job counts in `/health` carry no ids. Set `ASYNC_JOBS_ENABLED=false` to remove the API altogether, and put the service behind a proxy that authenticates.

## Maintenance

### Updating WhisperX

If you run the prebuilt image, pull the new tag:

```bash
docker compose pull
docker compose up -d
```

If you build from source, the default build now installs PyTorch 2.7.1
from the cu126 wheel index (Pascal-Hopper). To target a different combo:

```bash
git pull
docker compose build --no-cache \
  --build-arg TORCH_VERSION=2.8.0 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
docker compose up -d
```

Build args also work via Compose's `build.args` if you check that into
your overlay file.

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

## Stress Testing

A stress test script is included to measure throughput and latency under concurrent load:

```bash
# Default: 4 concurrent workers, all files in testfiles/
python tests/stress_test.py

# 8 concurrent workers, 3 rounds
python tests/stress_test.py --workers 8 --rounds 3

# Test OpenAI-compat endpoint
python tests/stress_test.py --endpoint openai

# Without diarization
python tests/stress_test.py --no-diarize
```

Place `.mp3` files in the `testfiles/` directory (gitignored). The report shows per-request latency, throughput in requests/minute, and the speedup from concurrent execution. See `tests/README.md` for full details.

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

### v0.5.0 (2026-11-13)

**Upgrade notes**

- No configuration changes are required. The new limits are off by default, and responses for admitted requests are unchanged except for the new `diarization` field and `X-Diarization-Status` header when diarization is requested.
- The asynchronous job API is new and is on by default. The existing endpoints are unchanged, so a client that does not use `/jobs` is unaffected. Set `ASYNC_JOBS_ENABLED=false` to leave the paths answering 404.
- `/health` and `/` gained a `capabilities` object, and `/health` gained `jobs` counts. Both are additions; every field earlier versions answered with is still present.
- Because the service has no authentication, a job id is the only gate on the transcript it belongs to. `GET /jobs` would make the ids enumerable, so it answers 404 unless `JOBS_LIST_ENABLED=true`. See "Security Notes".
- Compose files from earlier releases: to use the new Ray Serve memory layout, add the `tmpfs` entry for `/ray-tmp` from the current `docker-compose.yml`. Without it, Ray's session files and logs are on the container's disk.
- The container now exits after sustained failed health checks so that the restart policy can start a fresh one. Set `WATCHDOG_ENABLED=false` to keep the previous behaviour.
- `/health` is 503 when the service is degraded. Monitors that only check for a response should accept that 503 means a problem.

**New features**

- **Asynchronous jobs:** `POST /jobs` returns a job id at once and the transcription continues on the server, so a client timeout or a dropped connection no longer loses the work. `GET /jobs/{id}` reports status and progress, `GET /jobs/{id}/result` returns the transcription in any of the output formats, and `DELETE /jobs/{id}` cancels a job or drops a finished transcript. Job state is kept beside the decoded audio, so jobs that were in flight when the service restarted are reported as `interrupted` instead of disappearing. See "Asynchronous Jobs".
- **Identical submits are not run twice:** a submit matching a job that is queued, running or finished returns that job id with `deduplicated` set. Matching is on the uploaded bytes and the transcription parameters, or on an `Idempotency-Key` header and those parameters.
- **A failed job reports a code:** `error` carries `code`, `message` and `retryable`, so a client can tell `audio_too_long` from `out_of_memory`.
- **Capability advertisement:** `/health` and `/` report a `capabilities` object, so a client can use the job API where it exists and the synchronous endpoints elsewhere.

**Fixes**

- **Long files no longer fill `/dev/shm` in Ray Serve mode:** the decoded audio was passed to each stage through the Ray object store (three copies per request with the split strategy). With a few concurrent multi-hour files, `/dev/shm` filled up, requests failed with "object store is full" or "local disk is full", and the service could stay unusable until a manual restart. The decoded audio is now read from a per-request file in each stage.
- **Uploads are streamed to disk:** the whole upload was read into memory before `MAX_FILE_SIZE_MB` was checked. The limit is now applied while the file is copied, and from `Content-Length` before the body is read.
- **Decoding no longer blocks the server:** ffmpeg was run synchronously in the request handler, so `/health` and every other request stalled for the duration of each decode (about 25 seconds for a 6-hour file). Decoding is now done in a background process.
- **Long stage calls no longer block replica health checks:** stage work is done in a worker thread, so a single alignment or diarization call longer than the 300-second health-check timeout is not cut short by a replica restart.
- **Ray logs and spilled objects are kept out of `/dev/shm`:** session files and logs are in `RAY_TMPDIR` with log rotation, and objects spill to disk.
- **Leftover temp files are removed:** uploads left in the temp directory by a crash are removed at container start.
- **The Docker healthcheck fails on non-200 answers:** any HTTP response, including 500 and 503, counted as healthy.
- **Failed diarization is reported:** after a diarization error (for example CUDA out of memory), the transcript was returned without speakers and without any sign of the failure. The failure is now marked in the response, and an out-of-memory error is retried once.
- **Ray errors map to 503:** object store full, out of disk and replica failures are answered with 503 and `Retry-After` instead of 500.
- **Replicate strategy on CPU:** a GPU was requested for the full-pipeline replica even with `DEVICE=cpu`, so the replica could not start on a machine without one.

**New settings**

- Asynchronous jobs: `ASYNC_JOBS_ENABLED` (default `true`), `MAX_ASYNC_JOBS` (default 16), `JOB_RESULT_TTL_SECONDS` (default `JOB_TTL_SECONDS`).
- Admission control: `MAX_AUDIO_SECONDS`, `MAX_INFLIGHT_AUDIO_SECONDS`, `MAX_QUEUED_AUDIO_SECONDS`, `MAX_QUEUED_REQUESTS`, `QUEUE_WAIT_TIMEOUT_SECONDS` (all off by default).
- Job files: `JOB_DIR`, `JOB_TTL_SECONDS`.
- Health and watchdog: `HEALTH_MIN_SHM_FREE_PCT`, `HEALTH_MIN_DISK_FREE_MB`, `HEALTH_STAGE_TIMEOUT_SECONDS`, `WATCHDOG_ENABLED`, `WATCHDOG_INTERVAL_SECONDS`, `WATCHDOG_FAILURES`, `WATCHDOG_TIMEOUT_SECONDS`, `WATCHDOG_SHM_MAX_USED_PCT`, `WATCHDOG_SHM_FAILURES`.
- Ray Serve: `RAY_TMPDIR`, `RAY_SPILL_DIR`, `RAY_ROTATION_MAX_BYTES`, `RAY_ROTATION_BACKUP_COUNT`, `STAGE_MAX_ONGOING_REQUESTS` (default 2; it was Ray's default of 5), `PIPELINE_MAX_ONGOING_REQUESTS`.
- Diarization: `DIARIZE_ON_ERROR`.
- Metrics: `whisperx_inflight_audio_seconds`, `whisperx_queued_requests`, `whisperx_queued_audio_seconds`, `whisperx_admission_rejections_total`.

### v0.4.4 (2026-10-23)

**Upgrade notes**

- No configuration changes are required. Existing model cache volumes are reused without a new download, and transcripts, speaker labels and word timestamps on the test recordings are identical to v0.4.3.
- Ray Serve mode: the Ray Dashboard stays unreachable by default, as in earlier releases. Compose files from earlier releases publish port 8265 on all interfaces; before enabling the dashboard with `RAY_DASHBOARD_HOST=0.0.0.0`, change that line to the `127.0.0.1` mapping in the current `docker-compose.yml`.
- Multipart limits of the updated web stack: a request with a form field larger than 1 MB, or with more than 1000 fields or files, receives a 400 response. Uploaded audio files are not affected by the 1 MB limit; `MAX_FILE_SIZE_MB` still applies to them.
- Newer releases of transformers (5.20), huggingface_hub (2.3) and CTranslate2 (4.8.3) are included in the image. The PyTorch version is unchanged.

**Security**

- **Web stack updated for known vulnerabilities:** FastAPI 0.104.1 is replaced by 0.162.0, Starlette 0.27.0 by 1.7.0 (minimum 1.3.1), python-multipart 0.0.6 by 0.0.32, Uvicorn 0.24.0 by 0.54.0 and Pydantic 2.5.0 by 2.14.1. Fixes for the Starlette and python-multipart advisories on multipart parsing (denial of service and arbitrary file write, including CVE-2024-47874 and CVE-2026-24486) are included. Malformed multipart requests receive a 400 response, as do requests with a form field larger than 1 MB or with more than 1000 fields or files. No changes to the request or response format were needed.
- **Minimum versions for dependencies with advisories:** Ray `>=2.56.0`, anyio `>=4.14.2`, urllib3 `>=2.8.0`, aiohttp `>=3.14.4`, nltk `>=3.10.3`, msgpack `>=1.2.1`, fsspec `>=2026.6.0`, lightning `>=2.6.6`, certifi `>=2026.10.19` and virtualenv `>=21.14.9`. Ubuntu security updates (including openssl 3.0.2-0ubuntu1.30) are applied during the build. The PyTorch versions are unchanged.
- **Ray Dashboard bound to localhost:** port 8265 is published on `127.0.0.1` only in the Compose files, since jobs submitted through the dashboard are executed as code inside the container. The dashboard remains unreachable by default, as in earlier releases, and is enabled with `RAY_DASHBOARD_HOST=0.0.0.0`. Ray token authentication, enabled by default in current Ray releases, also applies to the dashboard. See [Ray Dashboard Access](#ray-dashboard-access) for SSH tunnel access and the token.

**Fixes**

- **Ray Serve HTTP port set explicitly:** `RAY_SERVE_DEFAULT_HTTP_PORT` is ignored by Ray 2.59, and with the Ray update the API would have started on port 8000 instead of 9000 in Ray Serve mode. The HTTP address is now passed to `serve.start()`.
- **Ray Dashboard can be opened through the published port:** in earlier images, the dashboard was bound to the container's loopback interface and could not be opened from the host, although port 8265 was published. It can now be enabled with `RAY_DASHBOARD_HOST=0.0.0.0`.

**Known remaining findings**

- Pillow 12.3.0 has advisories fixed in Pillow 13.0.0, which requires Python 3.11. The image uses the Python 3.10 of Ubuntu 22.04, so this update requires a change of base image and is left for a later release.
- Ray 2.59.0, the latest release, bundles aiohttp 3.14.3 for its runtime environment agent and Jackson 2.18.8 in its Java JAR; these copies are updated only with a Ray release. pip 26.2.2, the latest release, vendors msgpack 1.1.2, urllib3 2.7.0 and setuptools 70.3.0; these are used only when packages are installed.

### v0.4.3 (2026-10-11)

**Improvements**

- **Optional VAD settings for text invented in silence and noise (Speakr #418):** `VAD_METHOD` (`pyannote` or `silero`), `VAD_ONSET` and `VAD_OFFSET` are passed to WhisperX when set. Text such as "Thank you." is invented by Whisper from background noise that is passed to the model as speech. In tests, it was removed with `VAD_ONSET=0.8` or `VAD_METHOD=silero`, each with the costs described in "Reducing Text Invented in Silence and Noise". Without these settings, the model is loaded as before, and the output was identical to v0.4.2 on all test files.

### v0.4.2 (2026-09-30)

**Reported issues fixed**

- **Requests without a model failed when `PRELOAD_MODEL` was empty (Speakr #409):** the request default was taken from `PRELOAD_MODEL`, and an unset variable is forwarded by the shipped compose files as an empty string, so every request that named no model failed with "Invalid model size ''". An empty `PRELOAD_MODEL` now only disables preloading; for such requests, the new `DEFAULT_MODEL` is used, then `PRELOAD_MODEL`, then `large-v3`.
- **Malformed number settings stopped the service at startup:** a value such as `BATCH_SIZE=16  # comment`, which `docker run --env-file` keeps whole, made the import fail. `BATCH_SIZE`, `MAX_FILE_SIZE_MB`, `MAX_QUEUE_SIZE` and `GPU_CONCURRENCY` now fall back to their defaults when the value is not a number.

**Improvements**

- Unknown model names receive HTTP 400 with the accepted list, where a 500 from the engine was returned before.
- On `/v1/audio/transcriptions` and `/translations`, every model listed by `/v1/models` is accepted (previously six), and `whisper-1` maps to `OPENAI_WHISPER1_MODEL` or the default model in all modes. The same name resolution now applies to `/asr`, the OpenAI endpoints and Ray Serve mode.
- The preloaded model is resolved in the same way as requested models, so aliases work and no second copy is loaded under another name. On the `qwen3` and `external` backends, no Whisper model is preloaded, and in Ray Serve mode an invalid preload name is logged without stopping the replica.
- New optional settings `ALLOWED_MODELS` and `MAX_LOADED_MODELS`. A model in use is never unloaded by the cap or by the idle sweep, and a request for a loaded model is served without waiting for another model to load.
- Tests for the documented setup: unit tests, a check of the settings produced by `docker-compose.yml` with the setup guide's `.env`, and a service check on stand-in ML packages, run in GitHub Actions on every push without a GPU.
- Documentation: model defaults explained, `PRELOAD_MODEL` added to the setup guide's `.env`, and inline comments moved off value lines, since `docker run --env-file` keeps them as part of the value.

### v0.3.2 (2026-05-03)

**Reported issues fixed**

- **Pascal/Blackwell image variants (#15):** `:latest` ships PyTorch 2.7.1 (cu126, supports Pascal through Hopper). New `:blackwell` tag ships PyTorch 2.8.0 (cu128) for RTX 50xx. The `Dockerfile` now exposes `TORCH_VERSION` and `TORCH_INDEX_URL` build args and re-pins torch *after* the WhisperX install so the requested version sticks (the upstream fork was silently upgrading torch to 2.8 and breaking Pascal). CI publishes both variants per release with separate buildx cache scopes; either job can fail independently.
- **Device-aware `BATCH_SIZE` default (#12):** the default is now 16 on cuda and 2 on cpu. The hardcoded 16 was OOM-killing CPU runs (exit 137) on audio longer than ~30 minutes.
- **Idle model eviction (#16):** new `MODEL_KEEP_ALIVE_SECONDS` env var (default 0/disabled) unloads Whisper models that have been idle longer than the configured window. `MODEL_EVICTION_INTERVAL_SECONDS` controls sweep frequency (floor 30s). The next request that needs an evicted model reloads it transparently.
- **Real Prometheus `/metrics` (#13):** `/metrics` now returns OpenMetrics text instead of JSON. New histograms and counters cover request duration, status, audio duration/size, in-flight requests, loaded model count, model evictions, and VRAM. The previous JSON shape is preserved at `/queue-metrics` for callers that depended on it. See [Prometheus Metrics](#prometheus-metrics) for the full table and the Ray Serve caveat.

**Other fixes shipped in v0.3.2**

- `/asr` accepts the OpenAI-style aliases advertised by `/v1/models` (`whisper-1`, `whisper-tiny`, `whisper-large-v3`, ...). Previously these returned a 500 because the raw value was passed straight through to `faster_whisper.WhisperModel`.
- `/v1/models` is sourced from `faster_whisper.available_models()` instead of a hardcoded list, so the advertised set stays in sync with whatever engine version is installed (about 20 canonical names plus the `whisper-1` alias).
- `app.*` log records from Ray Serve replicas now flow into the per-replica log file. Previously `Loading WhisperX model: X`, `Starting transcription...`, etc. were silently dropped because Ray Serve disables propagation on its own logger.
- `whisperx_model_evictions_total{model="<name>"}` is pre-registered with value 0 each time a model loads, so dashboards can graph the metric from the first load onward instead of only after the first eviction. Simple mode only; see the Ray Serve caveat.

**New env vars**

| Variable | Default | Description |
|---|---|---|
| `MODEL_KEEP_ALIVE_SECONDS` | `0` (disabled) | Idle window after which a Whisper model is unloaded |
| `MODEL_EVICTION_INTERVAL_SECONDS` | `60` (floor 30) | Sweep cadence for the eviction daemon |

**New build args**

| Build arg | Default | Description |
|---|---|---|
| `TORCH_VERSION` | `2.7.1` | PyTorch version to install |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu126` | PyTorch wheel index URL |

**Tests**

`tests/test_v0_3_2.sh` and `tests/test_keep_alive.sh` cover the new endpoints and eviction logic; both verified end-to-end against a Ray Serve container.

### v0.3.0 (2026-02-28)
- Thread-safe model loading with double-checked locking for concurrent request safety
- Add Ray Serve mode for high-throughput ASR with cross-request batching
- Two pipeline strategies: `replicate` (full pipeline per GPU) and `split` (stage per GPU)
- Multi-GPU support via `NUM_GPU_REPLICAS` and per-stage replica/fraction config
- Refactor pipeline into shared stage functions (`app/pipeline.py`)
- Add async GPU queue with semaphore for simple mode (non-blocking event loop)
- Add `/metrics` endpoint for pipeline monitoring
- Add `SERVE_MODE`, `PIPELINE_STRATEGY`, `NUM_GPU_REPLICAS` env vars
- Add `entrypoint.sh` for automatic mode switching in Docker
- GPU pinning via `NVIDIA_VISIBLE_DEVICES` in local compose overrides
- Add stress test suite (`tests/stress_test.py`)

### v0.2.0 (2025-01-21)
- Add /v1/models and /v1/audio/transcriptions endpoints for OpenAI API compatibility
- Add diarize parameter for broader API compatibility
- Add offline mode support and fix model caching
- Use Query() parameters to work as a drop-in replacement for other Whisper ASR services

### v0.1.1alpha (2025-11-23)
- Initial release
- WhisperX integration with API wrapper
- Speaker diarization support
- Docker deployment
