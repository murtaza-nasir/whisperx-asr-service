"""
Ray Serve application entry point.

Wraps the existing FastAPI app with @serve.ingress so all current endpoints
(/asr, /v1/audio/transcriptions, /health, etc.) are preserved.  The ASR
pipeline stages run as independent Ray Serve deployments with cross-request
batching.

Start with:
    serve run app.serve_app:app
"""

import os
import time
import asyncio
import logging
import warnings
from typing import Optional, List

from fastapi import (
    FastAPI, File, UploadFile, Form, Query, Header, HTTPException, Request,
)
from fastapi.responses import JSONResponse, PlainTextResponse, Response
import ray
import ray.exceptions
import ray.serve.exceptions
from ray import serve

from app.version import __version__
from app.pipeline import (
    DEVICE,
    COMPUTE_TYPE,
    BATCH_SIZE,
    DEFAULT_MODEL,
    InvalidModelError,
    format_timestamp,
    sanitize_float_values,
    resolve_model_name,
    list_available_models,
    describe_model_settings,
    DiarizationError,
    _env_int,
    _whisper_models as loaded_models,
)
from app import metrics as prom_metrics
import app.health as health_mod
from app.admission import get_controller, AdmissionRejected, AudioTooLong
from app.audio_io import JOB_DIR, UploadTooLarge, AudioDecodeError
import app.jobs as jobs_mod
from app.response_format import format_asr_response
from app.job_runner import (
    JobManager,
    disabled_response,
    not_found_response,
    too_many_response,
    poll_after,
)
from app.request_flow import (
    admitted_job,
    busy_response,
    unavailable_response,
    too_large_response,
    diarization_fields,
    MaxUploadSizeMiddleware,
    startup_sweep,
)
from app.schemas import (
    ResponseFormat,
    TranscriptionWord,
    TranscriptionSegment,
    TranscriptionVerboseJsonResponse,
    OpenAIErrorDetail,
    OpenAIErrorResponse,
)
from app.serve_deployments import (
    PIPELINE_STRATEGY,
    STAGE_MAX_ONGOING_REQUESTS,
    PIPELINE_MAX_ONGOING_REQUESTS,
    FullPipelineDeployment,
    WhisperDeployment,
    AlignDeployment,
    DiarizeDeployment,
)

warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = _env_int("MAX_FILE_SIZE_MB", 1000)

def _error_types(module, names):
    return [t for t in (getattr(module, n, None) for n in names) if isinstance(t, type)]


# Ray errors that mean "the service cannot take this request right now"
# (object store or disk full, a replica died or is restarting). They are
# answered with 503 and Retry-After so clients back off and try again.
_UNAVAILABLE_ERRORS = tuple(
    _error_types(ray.exceptions, (
        "ObjectStoreFullError", "OutOfDiskError", "RayActorError", "ActorDiedError",
        "ActorUnavailableError", "ObjectLostError", "OutOfMemoryError",
    ))
    + _error_types(ray.serve.exceptions, (
        "BackPressureError", "DeploymentUnavailableError", "ReplicaUnavailableError",
    ))
)

# Shared memory holds only the Ray object store; the session dir and logs
# live in RAY_TMPDIR. Both are checked by /health.
SHM_PATH = os.getenv("RAY_PLASMA_DIR") or "/dev/shm"
RAY_TMPDIR = os.getenv("RAY_TMPDIR") or ""

# Model names are resolved by app.pipeline.resolve_model_name, shared with
# the simple-mode endpoints.
logger.info(f"Models: {describe_model_settings()}")

def _build_available_models():
    """
    Build the /v1/models response from faster-whisper's authoritative list,
    so this stays in sync with whatever engine version is installed.

    The OpenAI-compat alias `whisper-1` is added on top because some client
    SDKs hard-code it; everything else is canonical.
    """
    models = [{"id": "whisper-1", "object": "model", "owned_by": "openai"}]
    for name in list_available_models():
        models.append({"id": name, "object": "model", "owned_by": "whisperx"})
    return models


AVAILABLE_MODELS = _build_available_models()


def create_openai_error(status_code, message, error_type="invalid_request_error",
                        param=None, code=None):
    error_response = OpenAIErrorResponse(
        error=OpenAIErrorDetail(message=message, type=error_type, param=param, code=code)
    )
    return JSONResponse(status_code=status_code, content=error_response.model_dump())


fastapi_app = FastAPI(
    title="WhisperX ASR API (Ray Serve)",
    description="Automatic Speech Recognition API with Speaker Diarization using WhisperX",
    version=__version__,
)
fastapi_app.add_middleware(MaxUploadSizeMiddleware, max_file_size_mb=MAX_FILE_SIZE_MB)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
    # The ingress is a lightweight async FastAPI router -- it reads the file,
    # then awaits a GPU replica.  It must accept many concurrent requests so
    # the proxy never blocks while GPU replicas are busy.
    max_ongoing_requests=100,
)
@serve.ingress(fastapi_app)
class ASRIngress:

    def __init__(self, pipeline_handle=None, whisper_handle=None,
                 align_handle=None, diarize_handle=None):
        self._pipeline = pipeline_handle
        self._whisper = whisper_handle
        self._align = align_handle
        self._diarize = diarize_handle
        prom_metrics.SERVICE_INFO.info({
            "version": __version__,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "serve_mode": "ray",
        })
        self._admission = get_controller()
        prom_metrics.bind_admission(self._admission)
        if pipeline_handle is not None:
            self._stages = {"pipeline": (pipeline_handle, PIPELINE_MAX_ONGOING_REQUESTS)}
        else:
            self._stages = {
                "whisper": (whisper_handle, STAGE_MAX_ONGOING_REQUESTS),
                "align": (align_handle, STAGE_MAX_ONGOING_REQUESTS),
                "diarize": (diarize_handle, STAGE_MAX_ONGOING_REQUESTS),
            }
        # Calls this ingress has sent to each stage and not yet received.
        self._stage_calls = {name: 0 for name in self._stages}
        self._health_cache = None
        self._stage_models: List[str] = []
        os.makedirs(JOB_DIR, exist_ok=True)
        startup_sweep()
        # Asynchronous jobs. recover() reports jobs that were in flight when
        # an earlier process stopped as interrupted instead of losing them.
        self._jobs = JobManager(self._admission, self._execute_job)
        if jobs_mod.async_jobs_enabled():
            self._jobs.recover()
        try:
            if len([n for n in ray.nodes() if n.get("Alive")]) > 1:
                logger.warning(
                    "The Ray cluster has more than one node. Stages read the decoded "
                    f"audio from JOB_DIR ({JOB_DIR}), so it must be a path shared by every node."
                )
        except Exception:
            pass

    async def _call(self, stage: str, method, *args, **kwargs):
        """Call a stage and count it while in flight (for the health ping)."""
        self._stage_calls[stage] += 1
        try:
            return await method.remote(*args, **kwargs)
        finally:
            self._stage_calls[stage] -= 1

    async def _ping_stages(self) -> dict:
        timeout = health_mod.HEALTH_STAGE_TIMEOUT_SECONDS

        async def one(name, handle, slots):
            # A stage whose request slots are all taken by this ingress is
            # working; a ping would wait behind those calls, so skip it.
            if self._stage_calls.get(name, 0) >= slots:
                return name, {"ok": True, "busy": True, "pinged": False}
            try:
                info = await asyncio.wait_for(handle.ping.remote(), timeout=timeout)
                return name, {"ok": True, "busy": bool(info.get("busy")), "pinged": True,
                              "loaded_models": info.get("loaded_models", [])}
            except asyncio.TimeoutError:
                return name, {"ok": False, "error": f"no answer within {timeout:g}s"}
            except Exception as e:
                return name, {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}

        results = await asyncio.gather(*(one(n, h, s) for n, (h, s) in self._stages.items()))
        return dict(results)

    # ------------------------------------------------------------------
    # Basic endpoints
    # ------------------------------------------------------------------
    @fastapi_app.get("/")
    async def root(self):
        return {
            "status": "running",
            "service": "WhisperX ASR API",
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "serve_mode": "ray",
            "capabilities": jobs_mod.capabilities(),
        }

    @fastapi_app.get("/health")
    async def health_check(self):
        """200 when the service can take work; 503 with reasons when shared
        memory or disk is low or a stage does not answer. Results are reused
        for 2 seconds so frequent probes stay cheap."""
        now = time.monotonic()
        if self._health_cache and now - self._health_cache[0] < 2.0:
            code, body = self._health_cache[1]
        else:
            shm = [SHM_PATH] + ([RAY_TMPDIR] if RAY_TMPDIR else [])
            reasons, resources = health_mod.check_resources(shm_paths=shm, disk_paths=[JOB_DIR])
            stages = await self._ping_stages()
            code, status, reasons = health_mod.evaluate(reasons, stages)
            models = sorted({m for s in stages.values() for m in s.get("loaded_models", [])})
            if models or any(s.get("pinged") for s in stages.values()):
                self._stage_models = models
            body = {
                "status": status,
                "device": DEVICE,
                "loaded_models": self._stage_models,
                "serve_mode": "ray",
                "pipeline_strategy": PIPELINE_STRATEGY,
                "reasons": reasons,
                "stages": stages,
                "resources": resources,
                "capabilities": jobs_mod.capabilities(),
                "admission": {**self._admission.snapshot(),
                              "saturated": self._admission.saturated(),
                              "limits": self._admission.settings()},
                "jobs": self._jobs.store.counts(),
            }
            self._health_cache = (now, (code, body))
        body = dict(body, admission={**self._admission.snapshot(),
                                     "saturated": self._admission.saturated(),
                                     "limits": self._admission.settings()},
                    jobs=self._jobs.store.counts())
        return JSONResponse(status_code=code, content=body)

    @fastapi_app.get("/metrics")
    async def metrics(self):
        """Prometheus metrics in OpenMetrics text format.

        Note: in Ray Serve mode this only reflects the ingress process. Whisper
        models load inside replica processes, so whisperx_loaded_models and
        whisperx_vram_allocated_bytes will read 0 here. HTTP-level counters
        (requests, durations, audio sizes) are accurate.
        """
        prom_metrics.LOADED_MODELS.set(len(loaded_models))
        prom_metrics.refresh_vram()
        body, content_type = prom_metrics.render()
        return Response(content=body, media_type=content_type)

    @fastapi_app.get("/queue-metrics")
    async def queue_metrics(self):
        """Pipeline state (JSON; the old /metrics shape)."""
        return {
            "serve_mode": "ray",
            "device": DEVICE,
            "loaded_models": self._stage_models,
            "queue": self._admission.snapshot(),
        }

    # ------------------------------------------------------------------
    # Asynchronous jobs
    # ------------------------------------------------------------------
    @fastapi_app.post("/jobs")
    async def submit_job(
        self,
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
        """Accept an upload and return a job id.

        The upload is received and decoded before the response, so audio that
        is too long is refused here with 413 and audio that cannot be decoded
        with 400. Waiting for the pipeline happens afterwards, so the client
        may disconnect. A submit identical to one already queued, running or
        finished answers 200 with that job id and `deduplicated` true.
        """
        request_started = time.time()
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
        metric_status = "error"
        try:
            try:
                job, deduplicated = await self._jobs.submit(
                    audio_file, params, MAX_FILE_SIZE_MB, idempotency_key,
                )
            except UploadTooLarge as e:
                metric_status = "http_413"
                return too_large_response(e)
            except AudioTooLong as e:
                metric_status = "http_413"
                prom_metrics.ADMISSION_REJECTIONS.labels(reason="too_long").inc()
                return too_large_response(e, code="audio_too_long")
            except jobs_mod.TooManyJobs as e:
                metric_status = "http_503"
                prom_metrics.ADMISSION_REJECTIONS.labels(reason="too_many_jobs").inc()
                return too_many_response(e, self._admission.retry_after())
            except AudioDecodeError as e:
                metric_status = "http_400"
                raise HTTPException(status_code=400, detail=str(e))
            body = self._jobs.snapshot(job, deduplicated=deduplicated)
            body["poll_after_seconds"] = poll_after(job)
            code = 200 if deduplicated else 202
            metric_status = f"http_{code}"
            logger.info(
                f"Job {job.id} {'reused' if deduplicated else 'accepted'} for "
                f"{job.filename} ({job.audio_seconds or 0:.0f}s)"
            )
            return JSONResponse(status_code=code, content=body,
                                headers={"Location": body["status_url"]})
        except HTTPException as e:
            metric_status = f"http_{e.status_code}"
            raise
        finally:
            prom_metrics.REQUEST_DURATION.labels(endpoint="/jobs").observe(
                time.time() - request_started)
            prom_metrics.REQUESTS_TOTAL.labels(endpoint="/jobs", status=metric_status).inc()

    @fastapi_app.get("/jobs")
    async def list_jobs(self, limit: int = Query(50, ge=1, le=500)):
        """Jobs this process knows about, newest first.

        Off unless JOBS_LIST_ENABLED is set: a listing would make job ids
        enumerable, and an id is the only gate on its transcript.
        """
        if not jobs_mod.async_jobs_enabled() or not jobs_mod.jobs_list_enabled():
            return disabled_response()
        self._jobs.store.forget_expired()
        return {
            "jobs": [self._jobs.snapshot(j) for j in self._jobs.store.list()[:limit]],
            "counts": self._jobs.store.counts(),
        }

    @fastapi_app.get("/jobs/{job_id}")
    async def job_status(self, job_id: str):
        """Status and progress of one job. Cheap: no stage calls, and
        admission control does not apply."""
        if not jobs_mod.async_jobs_enabled():
            return disabled_response()
        try:
            job = self._jobs.store.get(job_id)
        except jobs_mod.JobNotFound:
            return not_found_response(job_id)
        body = self._jobs.snapshot(job)
        body["poll_after_seconds"] = poll_after(job)
        headers = {"Retry-After": str(body["poll_after_seconds"])} if job.active else {}
        return JSONResponse(content=body, headers=headers)

    @fastapi_app.get("/jobs/{job_id}/result")
    async def job_result(
        self,
        job_id: str,
        output_format: str = Query("json"),
        output: Optional[str] = Query(None),
    ):
        """The finished transcription, in the requested format.

        202 with Retry-After while the job is unfinished, 409 when it ended
        without a result, 404 when it is unknown or has expired.
        """
        if not jobs_mod.async_jobs_enabled():
            return disabled_response()
        if output is not None:
            output_format = output
        try:
            job = self._jobs.store.get(job_id)
        except jobs_mod.JobNotFound:
            return not_found_response(job_id)
        if job.active:
            body = self._jobs.snapshot(job)
            body["poll_after_seconds"] = poll_after(job)
            return JSONResponse(status_code=202, content=body,
                                headers={"Retry-After": str(body["poll_after_seconds"])})
        if job.status != jobs_mod.DONE:
            return JSONResponse(status_code=409, content=self._jobs.snapshot(job))
        try:
            payload = job.load_result()
        except (OSError, ValueError):
            return not_found_response(job_id)
        result = payload.get("result") or {}
        embeddings = payload.get("speaker_embeddings")
        detected_language = result.get("language", job.params.get("language") or "en")
        diarization, diarization_header = diarization_fields(result)
        response = self._format_asr_response(
            result, detected_language, output_format,
            bool(job.params.get("return_speaker_embeddings")), embeddings, diarization,
        )
        if diarization_header:
            response.headers["X-Diarization-Status"] = diarization_header
        return response

    @fastapi_app.delete("/jobs/{job_id}")
    async def cancel_job(self, job_id: str):
        """Cancel a queued or running job, or drop a finished transcript.
        Answers with the job's final state."""
        if not jobs_mod.async_jobs_enabled():
            return disabled_response()
        try:
            job = self._jobs.store.get(job_id)
        except jobs_mod.JobNotFound:
            return not_found_response(job_id)
        await self._jobs.cancel(job)
        return JSONResponse(content=self._jobs.snapshot(job))

    async def _execute_job(self, pcm_path, params: dict, progress):
        """Stage runner for one asynchronous job."""
        result, embeddings = await self._run_stages(pcm_path, params, progress)
        return result, embeddings, {}

    async def _run_stages(self, audio, params: dict, progress=None):
        """Run the configured stages over a decoded job file.

        Used by /asr and by asynchronous jobs, so both take exactly the same
        path through the pipeline. `progress` is called with the stage name as
        each one starts.
        """
        model = params.get("model")
        language = params.get("language")
        task = params.get("task") or "transcribe"
        initial_prompt = params.get("initial_prompt")
        hotwords = params.get("hotwords")
        word_timestamps = params.get("word_timestamps", True)
        should_diarize = params.get("should_diarize", True)
        num_speakers = params.get("num_speakers")
        min_speakers = params.get("min_speakers")
        max_speakers = params.get("max_speakers")
        return_speaker_embeddings = params.get("return_speaker_embeddings", False)

        def _stage(name):
            if progress is not None:
                progress(name)

        if self._pipeline:
            _stage("pipeline")
            result, speaker_embeddings = await self._call(
                "pipeline", self._pipeline.run,
                audio, model_name=model, language=language,
                task=task, initial_prompt=initial_prompt,
                hotwords=hotwords,
                word_timestamps=word_timestamps,
                should_diarize=should_diarize,
                num_speakers=num_speakers, min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_speaker_embeddings=return_speaker_embeddings,
            )
        else:
            _stage("whisper")
            result = await self._call(
                "whisper", self._whisper.transcribe,
                audio, model_name=model, language=language,
                task=task, initial_prompt=initial_prompt,
                hotwords=hotwords,
            )
            # qwen3 and text-only external segments are coarse chunks
            # until aligned, so speaker assignment needs word timestamps
            # even if the caller did not request them.
            internally_aligned = (
                result.get("_asr_backend") == "qwen3"
                or result.get("_qwen_align")
            )
            if word_timestamps or (should_diarize and internally_aligned):
                _stage("align")
                result = await self._call("align", self._align.align, audio, result)
            speaker_embeddings = None
            if should_diarize:
                _stage("diarize")
                result, speaker_embeddings = await self._call(
                    "diarize", self._diarize.diarize,
                    audio, result,
                    num_speakers=num_speakers, min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    return_speaker_embeddings=return_speaker_embeddings,
                )
            if not word_timestamps and internally_aligned:
                result.pop("word_segments", None)
                for seg in result.get("segments", []):
                    seg.pop("words", None)
            result.pop("_asr_backend", None)
            result.pop("_language_name", None)
            result.pop("_qwen_align", None)
        return result, speaker_embeddings

    # ------------------------------------------------------------------
    # /asr endpoint
    # ------------------------------------------------------------------
    @fastapi_app.post("/asr")
    async def transcribe_audio(
        self,
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
        request_started = time.time()
        metric_status = "error"
        prom_metrics.ACTIVE_TRANSCRIPTIONS.inc()
        try:
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

            if diarize is not None or enable_diarization is not None:
                should_diarize = (diarize is True) or (enable_diarization is True)
            else:
                should_diarize = True
            if return_speaker_embeddings is None:
                return_speaker_embeddings = False

            async with admitted_job(audio_file, self._admission, MAX_FILE_SIZE_MB) as job:
                prom_metrics.AUDIO_SIZE_MB.observe(job.size_mb)
                prom_metrics.AUDIO_DURATION.observe(job.duration)
                logger.info(
                    f"Processing {audio_file.filename} ({job.size_mb:.1f}MB, "
                    f"{job.duration:.0f}s), model: {model}"
                )
                audio = job.pcm_path

                result, speaker_embeddings = await self._run_stages(audio, {
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
                    "return_speaker_embeddings": return_speaker_embeddings,
                })

            detected_language = result.get("language", language or "en")
            diarization, diarization_header = diarization_fields(result)
            metric_status = "ok"
            response = self._format_asr_response(
                result, detected_language, output_format,
                return_speaker_embeddings, speaker_embeddings, diarization,
            )
            if diarization_header:
                response.headers["X-Diarization-Status"] = diarization_header
            return response

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
        except _UNAVAILABLE_ERRORS as e:
            metric_status = "http_503"
            logger.error(f"Transcription unavailable: {type(e).__name__}: {e}")
            return unavailable_response(f"{type(e).__name__}: {e}")
        except DiarizationError as e:
            logger.error(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            prom_metrics.ACTIVE_TRANSCRIPTIONS.dec()
            prom_metrics.REQUEST_DURATION.labels(endpoint="/asr").observe(time.time() - request_started)
            prom_metrics.REQUESTS_TOTAL.labels(endpoint="/asr", status=metric_status).inc()

    # ------------------------------------------------------------------
    # OpenAI-compat: /v1/audio/transcriptions
    # ------------------------------------------------------------------
    @fastapi_app.post("/v1/audio/transcriptions")
    async def create_transcription(
        self,
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        hotwords: Optional[str] = Form(None),
        response_format: ResponseFormat = Form(ResponseFormat.JSON),
        temperature: float = Form(0.0, ge=0.0, le=1.0),
    ):
        form_data = await request.form()
        timestamp_granularities = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            timestamp_granularities = []
        if response_format == ResponseFormat.VERBOSE_JSON and not timestamp_granularities:
            timestamp_granularities = ["segment"]

        return await self._process_openai_audio(
            file=file, model=model, language=language, prompt=prompt,
            response_format=response_format, temperature=temperature,
            timestamp_granularities=timestamp_granularities, task="transcribe",
            hotwords=hotwords,
        )

    # ------------------------------------------------------------------
    # OpenAI-compat: /v1/audio/translations
    # ------------------------------------------------------------------
    @fastapi_app.post("/v1/audio/translations")
    async def create_translation(
        self,
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        hotwords: Optional[str] = Form(None),
        response_format: ResponseFormat = Form(ResponseFormat.JSON),
        temperature: float = Form(0.0, ge=0.0, le=1.0),
    ):
        form_data = await request.form()
        timestamp_granularities = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            timestamp_granularities = []
        if response_format == ResponseFormat.VERBOSE_JSON and not timestamp_granularities:
            timestamp_granularities = ["segment"]

        return await self._process_openai_audio(
            file=file, model=model, language=None, prompt=prompt,
            response_format=response_format, temperature=temperature,
            timestamp_granularities=timestamp_granularities, task="translate",
            hotwords=hotwords,
        )

    # ------------------------------------------------------------------
    # OpenAI-compat: /v1/models
    # ------------------------------------------------------------------
    @fastapi_app.get("/v1/models")
    async def list_models(self):
        return {"object": "list", "data": AVAILABLE_MODELS}

    @fastapi_app.get("/v1/models/{model_id}")
    async def get_model(self, model_id: str):
        for m in AVAILABLE_MODELS:
            if m["id"] == model_id:
                return m
        return create_openai_error(404, f"Model '{model_id}' not found",
                                   code="model_not_found")

    # ------------------------------------------------------------------
    # Shared OpenAI-compat processing
    # ------------------------------------------------------------------
    async def _process_openai_audio(
        self, file, model, language, prompt, response_format,
        temperature, timestamp_granularities, task, hotwords=None,
    ):
        try:
            try:
                whisperx_model = resolve_model_name(model)
            except InvalidModelError as e:
                return create_openai_error(400, str(e), param="model")

            if timestamp_granularities and response_format != ResponseFormat.VERBOSE_JSON:
                return create_openai_error(
                    400,
                    "timestamp_granularities requires response_format='verbose_json'",
                    param="timestamp_granularities",
                )

            if temperature < 0 or temperature > 1:
                return create_openai_error(400, "temperature must be between 0 and 1",
                                           param="temperature")

            effective_hotwords = hotwords or prompt
            need_word_timestamps = (
                response_format == ResponseFormat.VERBOSE_JSON
                and "word" in timestamp_granularities
            )

            async with admitted_job(file, self._admission, MAX_FILE_SIZE_MB) as job:
                duration = job.duration
                logger.info(
                    f"OpenAI-compat: Processing {file.filename} ({job.size_mb:.1f}MB, "
                    f"{duration:.0f}s), model: {whisperx_model}, task: {task}"
                )
                audio = job.pcm_path
                if self._pipeline:
                    result, _ = await self._call(
                        "pipeline", self._pipeline.run,
                        audio, model_name=whisperx_model, language=language, task=task,
                        hotwords=effective_hotwords,
                        word_timestamps=need_word_timestamps,
                        should_diarize=False,
                    )
                else:
                    result = await self._call(
                        "whisper", self._whisper.transcribe,
                        audio, model_name=whisperx_model, language=language, task=task,
                        hotwords=effective_hotwords,
                    )
                    if need_word_timestamps:
                        result = await self._call("align", self._align.align, audio, result)

            detected_language = result.get("language", language or "en")

            # Format response
            if response_format == ResponseFormat.JSON:
                full_text = " ".join([
                    seg.get("text", "").strip()
                    for seg in result.get("segments", [])
                ]).strip()
                return JSONResponse(content={"text": full_text})

            elif response_format == ResponseFormat.TEXT:
                full_text = " ".join([
                    seg.get("text", "").strip()
                    for seg in result.get("segments", [])
                ]).strip()
                return PlainTextResponse(content=full_text)

            elif response_format == ResponseFormat.SRT:
                srt_content = []
                for i, segment in enumerate(result.get("segments", []), 1):
                    start_time = format_timestamp(segment.get("start", 0))
                    end_time = format_timestamp(segment.get("end", 0))
                    text = segment.get("text", "").strip()
                    srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
                return PlainTextResponse(content="\n".join(srt_content), media_type="text/plain")

            elif response_format == ResponseFormat.VTT:
                vtt_content = ["WEBVTT\n"]
                for segment in result.get("segments", []):
                    start_time = format_timestamp(segment.get("start", 0)).replace(",", ".")
                    end_time = format_timestamp(segment.get("end", 0)).replace(",", ".")
                    text = segment.get("text", "").strip()
                    vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
                return PlainTextResponse(content="\n".join(vtt_content), media_type="text/vtt")

            elif response_format == ResponseFormat.VERBOSE_JSON:
                include_words = "word" in timestamp_granularities
                include_segments = "segment" in timestamp_granularities or not timestamp_granularities

                full_text = " ".join([
                    seg.get("text", "").strip()
                    for seg in result.get("segments", [])
                ]).strip()

                segments = []
                if include_segments:
                    for idx, seg in enumerate(result.get("segments", [])):
                        segments.append(TranscriptionSegment(
                            id=idx,
                            seek=int(seg.get("start", 0) * 100),
                            start=seg.get("start", 0.0),
                            end=seg.get("end", 0.0),
                            text=seg.get("text", "").strip(),
                        ))

                words = None
                if include_words:
                    words = []
                    word_segments = result.get("word_segments", [])
                    if not word_segments:
                        for seg in result.get("segments", []):
                            word_segments.extend(seg.get("words", []))
                    for wd in word_segments:
                        if "word" in wd and "start" in wd and "end" in wd:
                            words.append(TranscriptionWord(
                                word=wd["word"].strip(),
                                start=wd.get("start", 0.0),
                                end=wd.get("end", 0.0),
                            ))

                resp = TranscriptionVerboseJsonResponse(
                    task=task, language=detected_language, duration=duration,
                    text=full_text, segments=segments, words=words,
                )
                return JSONResponse(content=resp.model_dump(exclude_none=True))

            return create_openai_error(400, f"Unsupported response format: {response_format}")

        except HTTPException as e:
            return create_openai_error(e.status_code, e.detail)
        except UploadTooLarge as e:
            return too_large_response(e, openai=True)
        except AudioTooLong as e:
            prom_metrics.ADMISSION_REJECTIONS.labels(reason="too_long").inc()
            return too_large_response(e, openai=True, code="audio_too_long")
        except AdmissionRejected as e:
            prom_metrics.ADMISSION_REJECTIONS.labels(reason=e.reason).inc()
            return busy_response(e, openai=True)
        except _UNAVAILABLE_ERRORS as e:
            logger.error(f"OpenAI-compat unavailable: {type(e).__name__}: {e}")
            return unavailable_response(f"{type(e).__name__}: {e}", openai=True)
        except Exception as e:
            logger.error(f"OpenAI-compat error: {e}", exc_info=True)
            return create_openai_error(500, f"Internal server error: {e}",
                                       error_type="server_error")

    # ------------------------------------------------------------------
    # /asr response formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_asr_response(result, detected_language, output_format,
                             return_speaker_embeddings, speaker_embeddings,
                             diarization=None):
        return format_asr_response(
            result, detected_language, output_format,
            return_speaker_embeddings, speaker_embeddings, diarization,
        )


# ------------------------------------------------------------------
# Bind deployments into the application graph
# ------------------------------------------------------------------
if PIPELINE_STRATEGY == "split":
    logger.info("Using split strategy: separate deployments per stage")
    app = ASRIngress.bind(
        pipeline_handle=None,
        whisper_handle=WhisperDeployment.bind(),
        align_handle=AlignDeployment.bind(),
        diarize_handle=DiarizeDeployment.bind(),
    )
else:
    logger.info("Using replicate strategy: full pipeline per GPU")
    app = ASRIngress.bind(
        pipeline_handle=FullPipelineDeployment.bind(),
    )
