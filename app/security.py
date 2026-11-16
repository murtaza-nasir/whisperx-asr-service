"""
Optional, environment-driven HTTP security layer shared by both serving modes.

Every protection is off by default so existing deployments and clients keep
working unchanged. This module centralises the configuration parsing and the
shared middleware so simple/uvicorn and Ray Serve stay in sync.
"""

import os
import time
import math
import hmac
import logging
from collections import deque
from typing import List, Optional, Deque, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid integer for %s=%r; using %d", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _split_csv(name: str) -> List[str]:
    return [item.strip() for item in _env_str(name).split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Configuration (read once at import time, consistent with the rest of app/)
# ---------------------------------------------------------------------------
API_KEY = _env_str("API_KEY").strip() or None
API_KEYS: List[str] = _split_csv("API_KEYS")
if API_KEY and API_KEY not in API_KEYS:
    API_KEYS.insert(0, API_KEY)

RATE_LIMIT_REQUESTS = _env_int("RATE_LIMIT_REQUESTS", 0)
RATE_LIMIT_WINDOW_SECONDS = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)

ALLOWED_ORIGINS: List[str] = _split_csv("ALLOWED_ORIGINS")

MAX_PROMPT_CHARS = _env_int("MAX_PROMPT_CHARS", 4096)

ALLOW_LOCAL_MODEL_PATHS = _env_bool("ALLOW_LOCAL_MODEL_PATHS", False)

ENABLE_DOCS = _env_bool("ENABLE_DOCS", True)

TRUST_PROXY = _env_bool("TRUST_PROXY", False)

MAX_FILE_SIZE_MB = _env_int("MAX_FILE_SIZE_MB", 1000)


# ---------------------------------------------------------------------------
# Security headers (always applied)
# ---------------------------------------------------------------------------
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
}


# ---------------------------------------------------------------------------
# Rate limiting state (in-memory, per process)
# ---------------------------------------------------------------------------
_RATE_LIMIT_BUCKETS: Dict[str, Deque[float]] = {}


def _client_ip(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return ""


def _is_protected_path(path: str) -> bool:
    return (
        path == "/asr"
        or path.startswith("/v1/audio/")
        or path in ("/metrics", "/queue-metrics")
    )


def _is_transcription_path(path: str) -> bool:
    return path == "/asr" or path.startswith("/v1/audio/")


def _extract_api_key(request: Request) -> Optional[str]:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    x_api_key = request.headers.get("x-api-key")
    if x_api_key is not None:
        return x_api_key.strip()
    return None


def _constant_time_key_match(supplied: str) -> bool:
    if not supplied or not API_KEYS:
        return False
    for key in API_KEYS:
        if hmac.compare_digest(supplied, key):
            return True
    return False


def _check_rate_limit(client_ip: str) -> Optional[int]:
    """Return Retry-After seconds if the request must be rejected, else None.

    Records the current request when allowed.
    """
    if RATE_LIMIT_REQUESTS <= 0:
        return None
    window = max(1, RATE_LIMIT_WINDOW_SECONDS)
    now = time.monotonic()
    bucket = _RATE_LIMIT_BUCKETS.get(client_ip)
    if bucket is None:
        bucket = deque()
        _RATE_LIMIT_BUCKETS[client_ip] = bucket
    while bucket and now - bucket[0] >= window:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_REQUESTS:
        retry_after = math.ceil(max(0.0, window - (now - bucket[0])))
        if retry_after < 1:
            retry_after = 1
        return retry_after
    bucket.append(now)
    return None


def _apply_security_headers(response: Response, request: Request) -> None:
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )


def _cors_simple_headers(origin: Optional[str]) -> dict:
    if not origin or not ALLOWED_ORIGINS or origin not in ALLOWED_ORIGINS:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Vary": "Origin",
    }


def _preflight_cors_headers(origin: str, request: Request) -> dict:
    headers = {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
        "Access-Control-Max-Age": "600",
        "Vary": "Origin, Access-Control-Request-Method, Access-Control-Request-Headers",
    }
    requested_headers = request.headers.get("access-control-request-headers")
    if requested_headers:
        headers["Access-Control-Allow-Headers"] = requested_headers
    else:
        headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-API-Key"
    return headers


class SecurityMiddleware(BaseHTTPMiddleware):
    """Shared HTTP-boundary middleware.

    Enforces security headers, optional API key auth, per-IP rate limiting and
    allowlist-based CORS for both serving modes.
    """

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        client_ip = _client_ip(request)
        path = request.url.path

        # CORS preflight: never requires an API key.
        is_preflight = (
            request.method == "OPTIONS"
            and "access-control-request-method" in request.headers
        )
        if is_preflight:
            if origin and origin in ALLOWED_ORIGINS:
                headers = _preflight_cors_headers(origin, request)
                status_code = 200
            else:
                headers = {}
                status_code = 400
            response = Response(status_code=status_code, headers=headers)
            _apply_security_headers(response, request)
            return response

        # API key authentication.
        if API_KEYS and _is_protected_path(path):
            supplied = _extract_api_key(request)
            if not _constant_time_key_match(supplied or ""):
                logger.warning(
                    "Authentication failed: client_ip=%s path=%s", client_ip, path
                )
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "Not authenticated"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
                for key, value in _cors_simple_headers(origin).items():
                    response.headers[key] = value
                _apply_security_headers(response, request)
                return response

        # Per-IP rate limiting on transcription endpoints.
        if _is_transcription_path(path):
            retry_after = _check_rate_limit(client_ip)
            if retry_after is not None:
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )
                for key, value in _cors_simple_headers(origin).items():
                    response.headers[key] = value
                _apply_security_headers(response, request)
                return response

        response = await call_next(request)
        _apply_security_headers(response, request)
        for key, value in _cors_simple_headers(origin).items():
            response.headers[key] = value
        return response


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------
_UPLOAD_CHUNK_SIZE = 64 * 1024


def sanitize_filename(name: Optional[str]) -> str:
    """Strip control characters and truncate an uploaded file name for safe
    logging. Never returns newlines or other control characters."""
    if not name:
        return ""
    cleaned = "".join(
        ch for ch in name
        if ch.isprintable() and ord(ch) not in (0x7F,) and not (0x80 <= ord(ch) <= 0x9F)
    )
    # Strip path components to avoid log confusion.
    cleaned = cleaned.replace("\\", "/").split("/")[-1]
    if len(cleaned) > 200:
        cleaned = cleaned[:200]
    return cleaned


sanitize_file_name = sanitize_filename
sanitize_upload_filename = sanitize_filename
_sanitise_filename = sanitize_filename


async def stream_upload_to_temp(upload_file, temp_path: str, max_file_size_mb: int = None,
                               chunk_size: int = _UPLOAD_CHUNK_SIZE) -> int:
    """Stream an uploaded file to temp_path in chunks."""
    if max_file_size_mb is None:
        max_file_size_mb = MAX_FILE_SIZE_MB
    max_bytes = int(max_file_size_mb) * 1024 * 1024
    total = 0
    with open(temp_path, "wb") as out:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large ({total / (1024 * 1024):.1f}MB). "
                        f"Maximum allowed: {max_file_size_mb}MB."
                    ),
                )
            out.write(chunk)
    return total


stream_upload = stream_upload_to_temp
save_upload_to_temp = stream_upload_to_temp
read_upload_to_temp = stream_upload_to_temp