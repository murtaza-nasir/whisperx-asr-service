"""Shared, typed environment-variable parsing.

Centralises the small parsers that the rest of ``app/`` uses to read
configuration from the environment.  Every parser treats an unset, empty or
whitespace-only value the same way (as "not set"), so callers do not have to
repeat that normalisation, and unparseable values fall back to the default
with a warning instead of raising.

This module intentionally has no heavy (ML) imports so that lightweight
modules such as ``app.queue`` can depend on it without pulling in torch.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def env_str(name: str, default: str = "") -> str:
    """String env var; unset/empty/whitespace values give *default*."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_or_none(name: str) -> Optional[str]:
    """Read an env var, treating unset OR empty/whitespace as None.

    Compose forwards optional vars as ``${VAR:-}``, which sets them to an
    empty string rather than leaving them unset, so a plain ``os.getenv()``
    would return ``""`` and downstream ``float()`` would crash.  Normalise
    that to ``None`` here.
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def env_int(name: str, default: int = 0) -> int:
    """Integer env var; unset, empty or unparseable values give the default."""
    value = env_or_none(name)
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"{name}={value!r} is not an integer; using {default}")
        return default


def env_float(
    name: str,
    low: Optional[float] = None,
    high: Optional[float] = None,
    default: Optional[float] = None,
) -> Optional[float]:
    """Float env var.

    Unset/empty/whitespace values give *default*.  Unparseable values give
    *default* with a warning.  When *low*/*high* are given, values outside
    the open interval ``(low, high)`` also give *default* with a warning.
    """
    value = env_or_none(name)
    if value is None:
        return default
    try:
        number = float(value)
    except (ValueError, TypeError):
        logger.warning(f"{name}={value!r} is not a number; using the default")
        return default
    if low is not None and not (low < number):
        logger.warning(f"{name}={value!r} is outside {low}-{high}; using the default")
        return default
    if high is not None and not (number < high):
        logger.warning(f"{name}={value!r} is outside {low}-{high}; using the default")
        return default
    return number


def env_bool(name: str, default: bool = False) -> bool:
    """Boolean env var; unset/empty/whitespace values give *default*."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")
