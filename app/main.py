"""
WhisperX ASR API Service
Compatible with openai-whisper-asr-webservice API endpoints
"""

import os
import time
import logging
import warnings
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Header
from fastapi.responses import JSONResponse, Response

from app.version import __version__
from app.pipeline import (
    _env_int,
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
    DiarizationError,
    _whisper_models as loaded_models,
)
from app.queue import run_in_queue, get_queue_metrics
from app import metrics as prom_metrics
import app.health as health_mod
from app.admission import get_controller, AdmissionRejected, AudioTooLong
from app.audio_io import JOB_DIR, UploadTooLarge, AudioDecodeError, load_pcm
import app.jobs as jobs_mod
from app.job_runner import (
    JobManager,
    disabled_response,
    not_found_response,
    too_many_response,
    poll_after,
)
from app.response_format import format_asr_response
from app.request_flow import (
    admitted_job,
    busy_response,
    too_large_response,
    diarization_fields,
    MaxUploadSizeMiddleware,
    startup_sweep,
)

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

MAX_FILE_SIZE_MB = _env_int("MAX_FILE_SIZE_MB", 1000)
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
    prom_metrics.bind_admission(get_controller())
    os.makedirs(JOB_DIR, exist_ok=True)
    startup_sweep()
    if jobs_mod.async_jobs_enabled():
        get_jobs().recover()
    preload_whisper_model("startup")
    yield


# Initialize FastAPI app
app = FastAPI(
    title="WhisperX ASR API",
    description="Automatic Speech Recognition API with Speaker Diarization using WhisperX",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(MaxUploadSizeMiddleware, max_file_size_mb=MAX_FILE_SIZE_MB)


def _run_pipeline_on_file(pcm_path, **kwargs):
    """Worker-thread entry: load the decoded job file, then run the pipeline."""
    return run_pipeline(load_pcm(pcm_path), **kwargs)


async def _execute_job(pcm_path, params, progress):
    """Stage runner for one asynchronous job in simple mode.

    Simple mode runs the whole pipeline in one call, so there is one stage.
    """
    progress("pipeline")
    result, speaker_embeddings = await run_in_queue(
        _run_pipeline_on_file,
        pcm_path,
        model_name=params.get("model"),
        language=params.get("language"),
        task=params.get("task") or "transcribe",
        initial_prompt=params.get("initial_prompt"),
        hotwords=params.get("hotwords"),
        word_timestamps=params.get("word_timestamps", True),
        should_diarize=params.get("should_diarize", True),
        num_speakers=params.get("num_speakers"),
        min_speakers=params.get("min_speakers"),
        max_speakers=params.get("max_speakers"),
        return_speaker_embeddings=params.get("return_speaker_embeddings", False),
    )
    return result, speaker_embeddings, {}


_jobs: Optional[JobManager] = None


def get_jobs() -> JobManager:
    """Process-wide job manager."""
    global _jobs
    if _jobs is None:
        _jobs = JobManager(get_controller(), _execute_job)
    return _jobs

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
        "capabilities": jobs_mod.capabilities(),
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

        async with admitted_job(audio_file, get_controller(), MAX_FILE_SIZE_MB) as job:
            prom_metrics.AUDIO_SIZE_MB.observe(job.size_mb)
            prom_metrics.AUDIO_DURATION.observe(job.duration)
            if job.size_mb > 100:
                logger.warning(f"Processing large file ({job.size_mb:.1f}MB) - may consume significant VRAM")
            logger.info(
                f"Processing audio file: {audio_file.filename} ({job.size_mb:.1f}MB, "
                f"{job.duration:.0f}s), model: {model}, language: {language}"
            )

            # Run pipeline through the async queue (GPU semaphore)
            result, speaker_embeddings = await run_in_queue(
                _run_pipeline_on_file,
                job.pcm_path,
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
        diarization, diarization_header = diarization_fields(result)
        headers = {"X-Diarization-Status": diarization_header} if diarization_header else None

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
            if diarization:
                response_data["diarization"] = diarization

            metric_status = "ok"
            return JSONResponse(content=response_data, headers=headers)

        elif output_format == "text":
            text = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
            metric_status = "ok"
            return JSONResponse(content={"text": text}, headers=headers)

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
            return JSONResponse(content={"srt": "\n".join(srt_content)}, headers=headers)

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
            return JSONResponse(content={"vtt": "\n".join(vtt_content)}, headers=headers)

        elif output_format == "tsv":
            tsv_content = ["start\tend\ttext\tspeaker"]
            for segment in result.get("segments", []):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")
                tsv_content.append(f"{start}\t{end}\t{text}\t{speaker}")

            metric_status = "ok"
            return JSONResponse(content={"tsv": "\n".join(tsv_content)}, headers=headers)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported output format: {output_format}")

    except HTTPException as e:
        metric_status = f"http_{e.status_code}"
        raise
    except UploadTooLarge as e:
        metric_status = "http_413"
        return too_large_response(e)
    except AudioTooLong as e:
        metric_status = "http_413"
        prom_metrics.ADMISSION_REJECTIONS.labels(reason="too_long").inc()
        return too_large_response(e, code="audio_too_long")
    except AdmissionRejected as e:
        metric_status = "http_503"
        prom_metrics.ADMISSION_REJECTIONS.labels(reason=e.reason).inc()
        return busy_response(e)
    except DiarizationError as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        prom_metrics.ACTIVE_TRANSCRIPTIONS.dec()
        prom_metrics.REQUEST_DURATION.labels(endpoint="/asr").observe(time.time() - request_started)
        prom_metrics.REQUESTS_TOTAL.labels(endpoint="/asr", status=metric_status).inc()
        prom_metrics.refresh_vram()


@app.get("/health")
async def health_check():
    """200 when the service can take work; 503 with reasons when the job
    directory's disk is nearly full."""
    reasons, resources = health_mod.check_resources(disk_paths=[JOB_DIR])
    code, status, reasons = health_mod.evaluate(reasons)
    controller = get_controller()
    return JSONResponse(status_code=code, content={
        "status": status,
        "device": DEVICE,
        "loaded_models": list(loaded_models.keys()),
        "serve_mode": SERVE_MODE,
        "reasons": reasons,
        "resources": resources,
        "capabilities": jobs_mod.capabilities(),
        "admission": {**controller.snapshot(), "saturated": controller.saturated(),
                      "limits": controller.settings()},
        "jobs": get_jobs().store.counts(),
    })


@app.post("/jobs")
async def submit_job(
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe"),
    language: Optional[str] = Query(None),
    initial_prompt: Optional[str] = Query(None),
    hotwords: Optional[str] = Query(None),
    word_timestamps: bool = Query(True),
    model: Optional[str] = Query(None),
    num_speakers: Optional[int] = Query(None),
    min_speakers: Optional[int] = Query(None),
    max_speakers: Optional[int] = Query(None),
    diarize: Optional[bool] = Query(None),
    enable_diarization: Optional[bool] = Query(None),
    return_speaker_embeddings: Optional[bool] = Query(None),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """Accept an upload and return a job id."""
    if not jobs_mod.async_jobs_enabled():
        return disabled_response()
    try:
        model = resolve_model_name(model)
    except InvalidModelError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if diarize is not None or enable_diarization is not None:
        should_diarize = (diarize is True) or (enable_diarization is True)
    else:
        should_diarize = True
    params = {
        "model": model,
        "language": language,
        "task": task,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
        "word_timestamps": word_timestamps,
        "should_diarize": should_diarize,
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "return_speaker_embeddings": bool(return_speaker_embeddings),
    }
    manager = get_jobs()
    try:
        job, deduplicated = await manager.submit(
            audio_file, params, MAX_FILE_SIZE_MB, idempotency_key,
        )
    except UploadTooLarge as e:
        return too_large_response(e)
    except AudioTooLong as e:
        prom_metrics.ADMISSION_REJECTIONS.labels(reason="too_long").inc()
        return too_large_response(e, code="audio_too_long")
    except jobs_mod.TooManyJobs as e:
        prom_metrics.ADMISSION_REJECTIONS.labels(reason="too_many_jobs").inc()
        return too_many_response(e, get_controller().retry_after())
    except AudioDecodeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    body = manager.snapshot(job, deduplicated=deduplicated)
    body["poll_after_seconds"] = poll_after(job)
    code = 200 if deduplicated else 202
    return JSONResponse(status_code=code, content=body,
                        headers={"Location": body["status_url"]})


@app.get("/jobs")
async def list_jobs(limit: int = Query(50, ge=1, le=500)):
    """Jobs this process knows about, newest first.

    Off unless JOBS_LIST_ENABLED is set: a listing would make job ids
    enumerable, and an id is the only gate on its transcript.
    """
    if not jobs_mod.async_jobs_enabled() or not jobs_mod.jobs_list_enabled():
        return disabled_response()
    manager = get_jobs()
    manager.store.forget_expired()
    return {
        "jobs": [manager.snapshot(j) for j in manager.store.list()[:limit]],
        "counts": manager.store.counts(),
    }


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """Status and progress of one job."""
    if not jobs_mod.async_jobs_enabled():
        return disabled_response()
    manager = get_jobs()
    try:
        job = manager.store.get(job_id)
    except jobs_mod.JobNotFound:
        return not_found_response(job_id)
    body = manager.snapshot(job)
    body["poll_after_seconds"] = poll_after(job)
    headers = {"Retry-After": str(body["poll_after_seconds"])} if job.active else None
    return JSONResponse(content=body, headers=headers)


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str, output_format: str = Query("json"),
                     output: Optional[str] = Query(None)):
    """The finished transcription, in the requested format."""
    if not jobs_mod.async_jobs_enabled():
        return disabled_response()
    if output is not None:
        output_format = output
    manager = get_jobs()
    try:
        job = manager.store.get(job_id)
    except jobs_mod.JobNotFound:
        return not_found_response(job_id)
    if job.active:
        body = manager.snapshot(job)
        body["poll_after_seconds"] = poll_after(job)
        return JSONResponse(status_code=202, content=body,
                            headers={"Retry-After": str(body["poll_after_seconds"])})
    if job.status != jobs_mod.DONE:
        return JSONResponse(status_code=409, content=manager.snapshot(job))
    try:
        payload = job.load_result()
    except (OSError, ValueError):
        return not_found_response(job_id)
    result = payload.get("result") or {}
    embeddings = payload.get("speaker_embeddings")
    detected_language = result.get("language", job.params.get("language") or "en")
    diarization, diarization_header = diarization_fields(result)
    headers = {"X-Diarization-Status": diarization_header} if diarization_header else None
    return format_asr_response(
        result, detected_language, output_format,
        bool(job.params.get("return_speaker_embeddings")), embeddings, diarization,
        headers=headers,
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued or running job, or drop a finished transcript."""
    if not jobs_mod.async_jobs_enabled():
        return disabled_response()
    manager = get_jobs()
    try:
        job = manager.store.get(job_id)
    except jobs_mod.JobNotFound:
        return not_found_response(job_id)
    await manager.cancel(job)
    return JSONResponse(content=manager.snapshot(job))


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
    data["admission"] = get_controller().snapshot()
    return data


# Register OpenAI-compatible API routers
# Import here to avoid circular imports (openai_compat imports from this module)
from app.openai_compat import router as openai_router, models_router
app.include_router(openai_router)
app.include_router(models_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
