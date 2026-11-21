"""
Optional, environment-driven HTTP security layer shared by both serving modes.

Every protection is off by default so existing deployments and clients keep
working unchanged. This module centralises the configuration parsing and the
shared middleware so simple/uvicorn and Ray Serve stay in sync.
"""

import hmac
import logging
import math
import os
import time
from collections import deque

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

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


def _split_csv(name: str) -> list[str]:
    return [item.strip() for item in _env_str(name).split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Configuration (read once at import time, consistent with the rest of app/)
# ---------------------------------------------------------------------------
API_KEY = _env_str("API_KEY").strip() or None
API_KEYS: list[str] = _split_csv("API_KEYS")
if API_KEY and API_KEY not in API_KEYS:
    API_KEYS.insert(0, API_KEY)

RATE_LIMIT_REQUESTS = _env_int("RATE_LIMIT_REQUESTS", 0)
RATE_LIMIT_WINDOW_SECONDS = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)

ALLOWED_ORIGINS: list[str] = _split_csv("ALLOWED_ORIGINS")

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
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}


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
        or path.startswith(("/v1/audio/", "/jobs/"))
        or path in ("/metrics", "/queue-metrics", "/jobs")
    )


def _is_transcription_path(method: str, path: str) -> bool:
    return (
        path == "/asr"
        or path.startswith("/v1/audio/")
        or (method == "POST" and path == "/jobs")
    )


def _extract_api_key(request: Request) -> str | None:
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


def _check_rate_limit(client_ip: str) -> int | None:
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
        return max(1, math.ceil(max(0.0, window - (now - bucket[0]))))
    bucket.append(now)
    return None


def _apply_security_headers(response: Response, request: Request) -> None:
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )


def _cors_simple_headers(origin: str | None) -> dict:
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
        headers["Access-Control-Allow-Headers"] = (
            "Authorization, Content-Type, X-API-Key"
        )
    return headers


def _rejection(
    request: Request,
    origin: str | None,
    status_code: int,
    content: dict,
    headers: dict,
) -> Response:
    response = JSONResponse(status_code=status_code, content=content, headers=headers)
    for key, value in _cors_simple_headers(origin).items():
        response.headers[key] = value
    _apply_security_headers(response, request)
    return response


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
                return _rejection(
                    request,
                    origin,
                    401,
                    {"detail": "Not authenticated"},
                    {"WWW-Authenticate": "Bearer"},
                )

        # Per-IP rate limiting on transcription endpoints.
        if _is_transcription_path(request.method, path):
            retry_after = _check_rate_limit(client_ip)
            if retry_after is not None:
                return _rejection(
                    request,
                    origin,
                    429,
                    {"detail": "Rate limit exceeded"},
                    {"Retry-After": str(retry_after)},
                )

        response = await call_next(request)
        _apply_security_headers(response, request)
        for key, value in _cors_simple_headers(origin).items():
            response.headers[key] = value
        return response


# ---------------------------------------------------------------------------
# Request validation helpers (shared by the serving modes)
# ---------------------------------------------------------------------------
def local_model_paths_allowed() -> bool:
    """Read at call time so tests can change the flag on this module."""
    return ALLOW_LOCAL_MODEL_PATHS


def prompt_too_long(text: str | None) -> bool:
    return text is not None and len(text) > MAX_PROMPT_CHARS


def check_prompt_lengths(initial_prompt: str | None, hotwords: str | None) -> None:
    """Raise HTTP 400 when a prompt or hotword string exceeds MAX_PROMPT_CHARS."""
    if prompt_too_long(initial_prompt):
        raise HTTPException(
            status_code=400,
            detail=f"initial_prompt exceeds {MAX_PROMPT_CHARS} characters",
        )
    if prompt_too_long(hotwords):
        raise HTTPException(
            status_code=400,
            detail=f"hotwords exceeds {MAX_PROMPT_CHARS} characters",
        )


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------
def sanitize_filename(name: str | None) -> str:
    """Strip control characters and truncate an uploaded file name for safe
    logging. Never returns newlines or other control characters."""
    if not name:
        return ""
    cleaned = "".join(
        ch
        for ch in name
        if ch.isprintable() and ord(ch) not in (0x7F,) and not (0x80 <= ord(ch) <= 0x9F)
    )
    # Strip path components to avoid log confusion.
    cleaned = cleaned.replace("\\", "/").split("/")[-1]
    if len(cleaned) > 200:
        cleaned = cleaned[:200]
    return cleaned
