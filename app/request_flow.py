"""
Request handling shared by simple mode (app/main.py, app/openai_compat.py)
and Ray Serve mode (app/serve_app.py): receive the upload into a job dir,
check its length, wait for admission, decode once, and clean up.

No ML imports at module level.
"""

import contextlib
import logging
import os
import tempfile
from typing import Optional

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

import app.audio_io as audio_io
from app.admission import AdmissionController, AdmissionRejected, AudioTooLong
from app.audio_io import AudioJob, AudioDecodeError, UploadTooLarge

logger = logging.getLogger(__name__)

# Allowance for multipart framing and form fields on top of the file itself
# when the Content-Length header is checked.
_MULTIPART_SLACK_BYTES = 1 << 20

# Paths that receive audio uploads.
UPLOAD_PATHS = ("/asr", "/v1/audio/transcriptions", "/v1/audio/translations")


@contextlib.asynccontextmanager
async def admitted_job(upload, controller: AdmissionController, max_file_size_mb: float):
    """Yield a decoded AudioJob once the request is admitted.

    Raises UploadTooLarge, AudioTooLong, AdmissionRejected or AudioDecodeError.
    The job dir is removed and the admission slot released on exit, also
    when the request is cancelled.
    """
    max_bytes = int(max_file_size_mb * 1024 * 1024) if max_file_size_mb else None
    async with AudioJob() as job:
        await job.receive(upload, max_bytes)
        estimate = await job.probe()
        if estimate is None:
            # No container duration (raw streams, some codecs): decode first.
            await job.decode()
            controller.check_length(job.duration)
            async with controller.slot(job.duration):
                yield job
            return
        controller.check_length(estimate)
        async with controller.slot(estimate):
            await job.decode()
            controller.check_length(job.duration)
            yield job


def busy_response(exc: AdmissionRejected, openai: bool = False) -> JSONResponse:
    headers = {"Retry-After": str(exc.retry_after)}
    if openai:
        content = {"error": {
            "message": str(exc), "type": "server_error", "param": None,
            "code": "server_busy",
        }, "retry_after": exc.retry_after, "queue": exc.snapshot}
    else:
        content = exc.body()
    return JSONResponse(status_code=503, content=content, headers=headers)


def unavailable_response(message: str, retry_after: int = 30, openai: bool = False) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)}
    if openai:
        content = {"error": {"message": message, "type": "server_error", "param": None,
                             "code": "service_unavailable"}}
    else:
        content = {"error": "unavailable", "detail": message, "retry_after": retry_after}
    return JSONResponse(status_code=503, content=content, headers=headers)


def too_large_response(exc: Exception, openai: bool = False, code: str = "file_too_large") -> JSONResponse:
    if openai:
        content = {"error": {"message": str(exc), "type": "invalid_request_error",
                             "param": None, "code": code}}
    else:
        content = {"detail": str(exc)}
    return JSONResponse(status_code=413, content=content)


def diarization_fields(result: dict):
    """Pop the internal diarization status from a pipeline result.
    Returns (response_field_or_None, header_value_or_None)."""
    status = result.pop("_diarization", None)
    if not status:
        return None, None
    return status, status.get("status")


class MaxUploadSizeMiddleware:
    """Refuse an upload whose Content-Length already exceeds MAX_FILE_SIZE_MB,
    before the body is read. Bodies without a length are checked while the
    file is copied into the job dir."""

    def __init__(self, app: ASGIApp, max_file_size_mb: float):
        self.app = app
        self.max_bytes = int(max_file_size_mb * 1024 * 1024) if max_file_size_mb else 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (self.max_bytes and scope.get("type") == "http"
                and scope.get("method") == "POST"
                and scope.get("path", "").rstrip("/") in UPLOAD_PATHS):
            length = None
            for key, value in scope.get("headers") or []:
                if key == b"content-length":
                    try:
                        length = int(value)
                    except ValueError:
                        pass
                    break
            if length is not None and length > self.max_bytes + _MULTIPART_SLACK_BYTES:
                limit_mb = self.max_bytes / (1024 * 1024)
                message = (f"File too large ({length / (1024 * 1024):.1f}MB). "
                           f"Maximum allowed: {limit_mb:.0f}MB.")
                if scope["path"].startswith("/v1/"):
                    content = {"error": {"message": message, "type": "invalid_request_error",
                                         "param": None, "code": "file_too_large"}}
                else:
                    content = {"detail": message}
                response = JSONResponse(status_code=413, content=content,
                                        headers={"Connection": "close"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def startup_sweep() -> None:
    """Remove job dirs older than JOB_TTL_SECONDS (in-process start)."""
    stats = audio_io.sweep(temp_dir=tempfile.gettempdir())
    if stats["job_dirs"] or stats["temp_files"]:
        logger.info(f"Removed {stats['job_dirs']} old job dirs and {stats['temp_files']} "
                    f"temp files ({stats['bytes'] / 1e6:.1f} MB)")


__all__ = [
    "admitted_job", "busy_response", "unavailable_response", "too_large_response",
    "diarization_fields", "MaxUploadSizeMiddleware", "startup_sweep",
    "UploadTooLarge", "AudioTooLong", "AdmissionRejected", "AudioDecodeError",
]
