"""
WhisperX ASR API Service
Compatible with openai-whisper-asr-webservice API endpoints
"""

import os
import time
import tempfile
import logging
import warnings
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, Response
import whisperx

from app.version import __version__
from app.pipeline import (
    DEVICE,
    COMPUTE_TYPE,
    BATCH_SIZE,
    HF_TOKEN,
    DEFAULT_MODEL,
    InvalidModelError,
    load_whisper_model,
    clear_gpu_memory,
    format_timestamp,
    sanitize_float_values,
    run_pipeline,
    resolve_model_name,
    preload_whisper_model,
    describe_model_settings,
    _whisper_models as loaded_models,
)
from app.queue import run_in_queue, get_queue_metrics
from app import metrics as prom_metrics
from app.security import SecurityMiddleware, ENABLE_DOCS, MAX_FILE_SIZE_MB

# Suppress pyannote pooling warnings about degrees of freedom
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# uvicorn applies logging.config.dictConfig() before our basicConfig runs,
# so set the "app" namespace level explicitly here.
logging.getLogger("app").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

SERVE_MODE = os.getenv("SERVE_MODE", "simple")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: preload models on startup."""
    prom_metrics.SERVICE_INFO.info({
        "version": __version__,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "serve_mode": SERVE_MODE,
    })
    preload_whisper_model("startup")
    yield


# Initialize FastAPI app
app = FastAPI(
    title="WhisperX ASR API",
    description="Automatic Speech Recognition API with Speaker Diarization using WhisperX",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

# Shared HTTP security boundary (auth, rate limiting, CORS, headers).
app.add_middleware(SecurityMiddleware)

logger.info(f"WhisperX ASR Service v{__version__} initialized on device: {DEVICE}")
logger.info(f"Compute type: {COMPUTE_TYPE}, Batch size: {BATCH_SIZE}")
logger.info(f"Models: {describe_model_settings()}; serve mode: {SERVE_MODE}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "WhisperX ASR API",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "serve_mode": SERVE_MODE,
    }


@app.post("/asr")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe"),
    language: Optional[str] = Query(None),
    initial_prompt: Optional[str] = Query(None),
    hotwords: Optional[str] = Query(None),
    word_timestamps: bool = Query(True),
    output_format: str = Query("json"),
    output: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    num_speakers: Optional[int] = Query(None),
    min_speakers: Optional[int] = Query(None),
    max_speakers: Optional[int] = Query(None),
    diarize: Optional[bool] = Query(None),
    enable_diarization: Optional[bool] = Query(None),
    return_speaker_embeddings: Optional[bool] = Query(None),
):
    """
    Main ASR endpoint compatible with openai-whisper-asr-webservice

    Args:
        audio_file: Audio file to transcribe
        task: transcribe or translate
        language: Language code (e.g., 'en', 'es', 'fr')
        initial_prompt: Optional prompt to guide the model
        word_timestamps: Return word-level timestamps
        output_format: json, text, srt, vtt, or tsv
        model: WhisperX model name (tiny, base, small, medium, large-v3, turbo, ...);
            omitted or empty uses the server's default model
        num_speakers: Exact number of speakers (if known, overrides min/max)
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        diarize: Enable speaker diarization (compatible with whisper-asr-webservice)
        enable_diarization: Alias for diarize (deprecated, use diarize instead)
        return_speaker_embeddings: Return speaker embeddings (256-dimensional vectors)
    """
    temp_audio_path = None
    request_started = time.time()
    metric_status = "error"
    prom_metrics.ACTIVE_TRANSCRIPTIONS.inc()

    try:
        # Handle legacy parameter names
        if output is not None:
            output_format = output

        # Map OpenAI-style aliases (whisper-tiny, whisper-large-v3, whisper-1, ...)
        # to canonical faster-whisper names so /asr accepts the same identifiers
        # advertised by /v1/models. No model means the default model; an
        # unknown one is a client error.
        try:
            model = resolve_model_name(model)
        except InvalidModelError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Resolve diarization toggle
        if diarize is not None or enable_diarization is not None:
            should_diarize = (diarize is True) or (enable_diarization is True)
        else:
            should_diarize = True
        if return_speaker_embeddings is None:
            return_speaker_embeddings = False

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as temp_file:
            temp_audio_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)

        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        prom_metrics.AUDIO_SIZE_MB.observe(file_size_mb)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB. "
                       f"Large files may cause out-of-memory errors."
            )

        if file_size_mb > 100:
            logger.warning(f"Processing large file ({file_size_mb:.1f}MB) - may consume significant VRAM")

        logger.info(f"Processing audio file: {audio_file.filename} ({file_size_mb:.1f}MB), model: {model}, language: {language}")

        # Load audio
        audio = whisperx.load_audio(temp_audio_path)
        prom_metrics.AUDIO_DURATION.observe(len(audio) / 16000.0)

        # Run pipeline through the async queue (GPU semaphore)
        result, speaker_embeddings = await run_in_queue(
            run_pipeline,
            audio,
            model_name=model,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
            should_diarize=should_diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )

        detected_language = result.get("language", language or "en")

        # Format output based on requested format
        if output_format == "json":
            response_data = {
                "text": result.get("segments", []),
                "language": detected_language,
                "segments": result.get("segments", []),
                "word_segments": result.get("word_segments", [])
            }

            if return_speaker_embeddings and speaker_embeddings:
                response_data["speaker_embeddings"] = sanitize_float_values(speaker_embeddings)
                logger.info(f"Including speaker embeddings in response: {list(speaker_embeddings.keys())}")

            metric_status = "ok"
            return JSONResponse(content=response_data)

        elif output_format == "text":
            text = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
            metric_status = "ok"
            return {"text": text}

        elif output_format == "srt":
            srt_content = []
            for i, segment in enumerate(result.get("segments", []), 1):
                start_time = format_timestamp(segment.get("start", 0))
                end_time = format_timestamp(segment.get("end", 0))
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")

                if speaker:
                    text = f"[{speaker}] {text}"

                srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")

            metric_status = "ok"
            return {"srt": "\n".join(srt_content)}

        elif output_format == "vtt":
            vtt_content = ["WEBVTT\n"]
            for segment in result.get("segments", []):
                start_time = format_timestamp(segment.get("start", 0)).replace(',', '.')
                end_time = format_timestamp(segment.get("end", 0)).replace(',', '.')
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")

                if speaker:
                    text = f"[{speaker}] {text}"

                vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")

            metric_status = "ok"
            return {"vtt": "\n".join(vtt_content)}

        elif output_format == "tsv":
            tsv_content = ["start\tend\ttext\tspeaker"]
            for segment in result.get("segments", []):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")
                tsv_content.append(f"{start}\t{end}\t{text}\t{speaker}")

            metric_status = "ok"
            return {"tsv": "\n".join(tsv_content)}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported output format: {output_format}")

    except HTTPException as e:
        metric_status = f"http_{e.status_code}"
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        prom_metrics.ACTIVE_TRANSCRIPTIONS.dec()
        prom_metrics.REQUEST_DURATION.labels(endpoint="/asr").observe(time.time() - request_started)
        prom_metrics.REQUESTS_TOTAL.labels(endpoint="/asr", status=metric_status).inc()
        prom_metrics.refresh_vram()
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "loaded_models": list(loaded_models.keys()),
        "serve_mode": SERVE_MODE,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics in OpenMetrics text format."""
    prom_metrics.LOADED_MODELS.set(len(loaded_models))
    prom_metrics.refresh_vram()
    body, content_type = prom_metrics.render()
    return Response(content=body, media_type=content_type)


@app.get("/queue-metrics")
async def queue_metrics():
    """Queue and pipeline state (JSON; the old /metrics shape)."""
    data = {
        "serve_mode": SERVE_MODE,
        "device": DEVICE,
        "loaded_models": list(loaded_models.keys()),
    }
    if SERVE_MODE == "simple":
        data["queue"] = get_queue_metrics()
    return data


# Register OpenAI-compatible API routers
# Import here to avoid circular imports (openai_compat imports from this module)
from app.openai_compat import router as openai_router, models_router
app.include_router(openai_router)
app.include_router(models_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
