"""
Submitting, running and reading back asynchronous jobs.

The HTTP layer (app/serve_app.py in Ray Serve mode, app/main.py in simple
mode) supplies one coroutine that runs the stages for a decoded file. Everything
else -- receiving the upload, the length check, deduplication, the admission
wait, the stage progress, error mapping, storing the result and cleaning up --
happens here, so both modes behave the same.

The upload is received and decoded inside the submit request, so a file that
is too long or cannot be decoded is refused immediately with 413 or 400.
Waiting for the pipeline happens afterwards in a task owned by the server, so
the client may disconnect.

No ML imports.
"""

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi.responses import JSONResponse

import app.audio_io as audio_io
import app.jobs as jobs_mod
from app.admission import AdmissionController
from app.audio_io import AudioJob

logger = logging.getLogger(__name__)

# How long a client should wait before polling again, by job state.
POLL_QUEUED_SECONDS = 10
POLL_RUNNING_SECONDS = 15

# Errors that mean "not now" rather than "not ever".
_UNAVAILABLE_NAMES = (
    "ObjectStoreFullError", "OutOfDiskError", "RayActorError", "ActorDiedError",
    "ActorUnavailableError", "ObjectLostError", "BackPressureError",
    "DeploymentUnavailableError", "ReplicaUnavailableError",
)
_OOM_NAMES = ("OutOfMemoryError", "OutOfDiskError", "ObjectStoreFullError")


def classify_error(exc: BaseException) -> str:
    """Map an exception to a job error code.

    Classes are matched by name along the exception's inheritance chain
    rather than with isinstance, so an error raised through a second import
    of its module is still classified. Ray's errors are matched by name for
    the same reason: this module must import without Ray.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "AudioTooLong" in names:
        return "audio_too_long"
    if "UploadTooLarge" in names:
        return "file_too_large"
    if "AudioDecodeError" in names:
        return "decode_failed"
    if "AdmissionRejected" in names:
        return "unavailable"
    if names & set(_OOM_NAMES) or "CUDA out of memory" in str(exc):
        return "out_of_memory"
    if any("OutOfMemory" in name for name in names):
        return "out_of_memory"
    if names & set(_UNAVAILABLE_NAMES):
        return "unavailable"
    if "DiarizationError" in names:
        return "stage_failed"
    return "internal"


def disabled_response() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": "not_found",
                 "detail": "Asynchronous jobs are disabled (ASYNC_JOBS_ENABLED=false)."},
    )


def not_found_response(job_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": "not_found",
                 "detail": f"No job {job_id}. It may have expired or the service restarted."},
    )


def too_many_response(exc: jobs_mod.TooManyJobs, retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": "busy", "reason": "too_many_jobs",
                 "retry_after": retry_after, "detail": str(exc)},
        headers={"Retry-After": str(retry_after)},
    )


def poll_after(job: jobs_mod.Job) -> int:
    if job.status == jobs_mod.QUEUED:
        return POLL_QUEUED_SECONDS
    eta = (job.progress(jobs_mod.INITIAL_REALTIME_FACTOR) or {}).get("eta_seconds")
    if isinstance(eta, int) and eta > 0:
        return max(POLL_RUNNING_SECONDS, min(300, eta // 4))
    return POLL_RUNNING_SECONDS


class JobManager:
    """Owns the store, the admission controller and the running tasks."""

    def __init__(self, controller: AdmissionController,
                 execute: Callable[..., Awaitable],
                 store: Optional[jobs_mod.JobStore] = None):
        self._controller = controller
        self._execute = execute
        self._store = store or jobs_mod.get_store()
        self._tasks: Dict[str, asyncio.Task] = {}

    @property
    def store(self) -> jobs_mod.JobStore:
        return self._store

    def recover(self) -> dict:
        return self._store.recover()

    def snapshot(self, job: jobs_mod.Job, deduplicated: bool = False) -> dict:
        return job.snapshot(self._store.rate, deduplicated=deduplicated)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------
    async def submit(self, upload, params: Dict[str, Any], max_file_size_mb: float,
                     idempotency_key: Optional[str] = None) -> tuple:
        """Receive, decode and register a job.

        Returns (job, deduplicated). Raises UploadTooLarge, AudioTooLong,
        AudioDecodeError or TooManyJobs, which the endpoint turns into 413,
        400 or 503.
        """
        self._store.forget_expired()
        max_bytes = int(max_file_size_mb * 1024 * 1024) if max_file_size_mb else None
        params_hash = jobs_mod.params_key(params)
        filename = getattr(upload, "filename", None)

        staging = AudioJob()
        await staging.__aenter__()
        keep_staging = False
        try:
            await staging.receive(upload, max_bytes)
            estimate = await staging.probe()
            if estimate is not None:
                self._controller.check_length(estimate)
            content_hash = (None if idempotency_key
                            else await asyncio.to_thread(jobs_mod.hash_file,
                                                         staging.upload_path))
            await staging.decode()
            self._controller.check_length(staging.duration)

            key = jobs_mod.dedupe_key(params_hash, content_hash, idempotency_key)
            existing = self._store.find_duplicate(key)
            if existing is not None:
                existing.submissions += 1
                existing.save()
                logger.info(f"Job {existing.id} reused for a duplicate submit "
                            f"({existing.submissions} submissions)")
                return existing, True

            job = self._store.create(params=params, key=key, filename=filename)
            # The staging directory becomes the job's directory, so the
            # decoded audio is not copied again.
            os.replace(staging.dir, job.dir)
            keep_staging = True
            job.audio_seconds = staging.duration
            job.size_bytes = staging.size_bytes
            job.save()
        finally:
            if not keep_staging:
                staging.cleanup()

        pcm_path = os.path.join(job.dir, audio_io.PCM_NAME)
        task = asyncio.create_task(self._run(job, pcm_path))
        job.task = task
        self._tasks[job.id] = task
        task.add_done_callback(lambda _t, jid=job.id: self._tasks.pop(jid, None))
        return job, False

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    async def _run(self, job: jobs_mod.Job, pcm_path: str) -> None:
        """Wait for admission, run the stages, store the result.

        The admission queue bounds and the queue wait timeout are not applied:
        the job is already accepted, and MAX_ASYNC_JOBS bounds the backlog.
        The in-flight audio limit still applies, so GPU work stays serialized
        with the synchronous requests.
        """
        try:
            async with self._controller.slot(job.audio_seconds or 0.0,
                                             enforce_queue_bounds=False,
                                             wait_timeout=None):
                job.start()
                result, embeddings, extras = await self._execute(
                    pcm_path, job.params, lambda stage: job.enter_stage(stage)
                )
                payload = {"result": result, "speaker_embeddings": embeddings}
                payload.update(extras or {})
                job.finish(payload)
                logger.info(f"Job {job.id} done in "
                            f"{(job.finished_at or 0) - (job.started_at or 0):.1f}s")
        except asyncio.CancelledError:
            job.mark_cancelled()
            job.drop_result()
            raise
        except BaseException as exc:  # noqa: BLE001 - every failure becomes job state
            code = classify_error(exc)
            logger.error(f"Job {job.id} failed ({code}): {type(exc).__name__}: {exc}",
                         exc_info=(code == "internal"))
            job.fail(code, str(exc) or type(exc).__name__)
        finally:
            # The decoded audio is the large file; the result and the state stay.
            try:
                os.unlink(pcm_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------
    async def cancel(self, job: jobs_mod.Job) -> bool:
        """Cancel a queued or running job. Returns True when it was active.

        Cancelling the task cancels the stage call it is awaiting, so the GPU
        stage is released as soon as the deployment honours it. A result that
        was already stored is dropped.
        """
        task = self._tasks.get(job.id) or job.task
        if job.active and task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except BaseException:
                pass
            if job.active:
                job.mark_cancelled()
            job.drop_result()
            return True
        if job.status == jobs_mod.DONE:
            # Finished before the cancel arrived: drop the stored transcript.
            job.mark_cancelled()
            job.drop_result()
            return True
        return False
