"""
Optional, environment-driven HTTP security layer shared by both serving modes.

Every protection is off by default so existing deployments and clients keep
working unchanged. This module centralises the configuration parsing and the
shared middleware so simple/uvicorn and Ray Serve stay in sync.
"""

import os
import logging
from typing import List

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

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


class SecurityMiddleware(BaseHTTPMiddleware):
    """Shared HTTP-boundary middleware.

    Ticket 01 delivers the foundation: security headers on every response and
    the wiring point for the protections added by later tickets (auth, rate
    limiting, CORS, ...). Each later ticket extends ``dispatch``.
    """

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value

        # HSTS only when TLS is actually in use.
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response
