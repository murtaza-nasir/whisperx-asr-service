# WhisperX ASR Service

## Build & Dev Commands

```bash
# Local dev (uv, CPU)
uv sync --extra cpu

# Local dev (uv, GPU/CUDA)
uv sync --extra gpu

# Production (prebuilt image)
docker compose up -d

# Dev build (local source, live reload)
docker compose -f docker-compose.dev.yml up --build

# Rebuild after changes
docker compose -f docker-compose.dev.yml up --build --force-recreate

# Stress test
python tests/stress_test.py

# Quick test
curl -F "audio_file=@test.wav" http://localhost:9000/asr
```

## Architecture

FastAPI service wrapping WhisperX with two serve modes:

**Simple mode** (`SERVE_MODE=simple`, default): Single uvicorn process. Async GPU queue runs pipeline in thread pool executor with `GPU_CONCURRENCY` concurrent slots. Per-stage locks in `pipeline.py` serialize access to non-thread-safe stages (transcribe, diarize) while allowing pipeline parallelism — multiple requests can be in different stages simultaneously (e.g., one transcribing while another aligns).

**Ray Serve mode** (`SERVE_MODE=ray`): Cross-request batching with `@serve.batch`. Two strategies:
- **Replicate** (`PIPELINE_STRATEGY=replicate`, default): Full 3-stage pipeline per GPU replica.
- **Split** (`PIPELINE_STRATEGY=split`): Each stage as separate Ray deployment with fractional GPU allocation (whisper 0.5, align 0.3, diarize 0.2).

Pipeline: Audio upload → Transcribe (Whisper) → Align (wav2vec2) → Diarize (pyannote) → JSON/SRT/VTT/conversation response.

Endpoints: `/asr` (native), `/v1/audio/transcriptions` and `/v1/audio/translations` (OpenAI-compatible), `/health`, `/metrics`.

## Module Structure

```
app/
├── __init__.py            # Package init
├── version.py             # __version__ = "0.3.1"
├── main.py                # Simple mode FastAPI app, /asr endpoint
├── pipeline.py            # Shared 3-stage pipeline: transcribe/align/diarize + model caching
├── queue.py               # Async GPU queue (semaphore + ThreadPoolExecutor)
├── upload.py              # Streaming file upload utility, FileTooLargeError, size constants
├── schemas.py             # Pydantic models (OpenAI-compatible responses)
├── openai_compat.py       # /v1/audio/* endpoints, model mapping
├── serve_app.py           # Ray Serve ingress (ASRIngress class)
└── serve_deployments.py   # Ray Serve deployments (FullPipeline, Whisper, Align, Diarize)
```

## Key Patterns

- **Thread-safe model loading**: Double-checked locking in `pipeline.py` — check cache, acquire lock, check again, load. Per-model and per-language caching.
- **Per-stage pipeline locks** (`pipeline.py`): `_transcribe_lock` serializes transcription (upstream WhisperX model mutates shared state), `_diarize_lock` serializes diarization (precautionary). `align()` is stateless and runs lock-free. An in-flight counter ensures `torch.cuda.empty_cache()` only runs when no other request is on the GPU.
- **Async GPU queue** (`queue.py`): `asyncio.Semaphore` + `ThreadPoolExecutor` keeps event loop responsive while GPU work runs in threads. Configurable via `GPU_CONCURRENCY`.
- **Graceful degradation**: Alignment and diarization catch exceptions and return partial results rather than failing the request.
- **Diarization off by default**: Diarization only runs when explicitly requested via `diarize=true` or `enable_diarization=true`.
- **Conversation output**: `output_format=conversation` merges consecutive segments from the same speaker into conversation turns.
- **Ray Serve batching**: `@serve.batch` on split-mode deployments collects requests up to `max_batch_size` or `batch_wait_timeout_s` (0.1s default).
- **Streaming uploads** (`upload.py`): Files are streamed to disk in chunks (default 8MB) to avoid loading entire uploads into memory. Size validated during streaming.
- **OpenAI API compatibility**: Model name mapping (whisper-1 → large-v3), response format translation, standard error responses.

## Critical Rules

- **HF_TOKEN required**: Diarization needs a HuggingFace token with access to pyannote models. Set via env var or `.env` file.
- **WhisperX as submodule**: `whisperx-custom/` is a git submodule. Clone with `--recurse-submodules`. Dockerfile and `uv sync` both use the local submodule path.
- **Version tag must match `app/version.py`**: Currently `0.3.1`. The entrypoint.sh prints this on startup.
- **GPU memory**: `large-v3` needs ~10GB VRAM. Service clears GPU memory between pipeline stages via `gc.collect()` + conditional `torch.cuda.empty_cache()` (skipped when other requests are in-flight).
- **Shared memory**: Ray mode requires `shm_size: 8g` in docker-compose for Ray object store.
- **Base image**: `nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04` with PyTorch 2.3.0 + CUDA 12.1 (Docker). Local dev via `uv` uses PyTorch 2.8.0 with cpu/gpu extras.

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVE_MODE` | simple | `simple` or `ray` |
| `DEVICE` | cuda | `cuda` or `cpu` |
| `COMPUTE_TYPE` | float16 | `float16`, `float32`, `int8` |
| `BATCH_SIZE` | 16 | Whisper batch size |
| `HF_TOKEN` | (required) | HuggingFace auth token |
| `PRELOAD_MODEL` | large-v3 | Model to load on startup |
| `PIPELINE_STRATEGY` | replicate | `replicate` or `split` (ray mode) |
| `GPU_CONCURRENCY` | 20 | Concurrent GPU runs (simple mode) |
| `NUM_GPU_REPLICAS` | 1 | Pipeline replicas (ray mode) |
| `UPLOAD_CHUNK_SIZE_BYTES` | 8388608 | Upload streaming chunk size (8MB) |

## Plans

Implementation plans are stored in `plans/` as numbered markdown files (e.g., `plans/001-feature-name.md`). Reference these for architectural context on past decisions.

Before implementing any plan make sure we capture it in `plans/` through creating new ones or updating existing ones.
