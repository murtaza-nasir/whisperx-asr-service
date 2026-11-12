"""
Health evaluation shared by simple mode and Ray Serve mode.

/health returns 503 with status "degraded" and a list of reasons when shared
memory or disk space runs low, or when a pipeline stage does not answer a
ping. A busy queue is reported, but does not make the service unhealthy.

No ML imports, so the CI tests use it directly.
"""

import os
import logging
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"{name}={value!r} is not a number; using {default}")
        return default


HEALTH_MIN_SHM_FREE_PCT = _env_float("HEALTH_MIN_SHM_FREE_PCT", 5.0)
HEALTH_MIN_DISK_FREE_MB = _env_float("HEALTH_MIN_DISK_FREE_MB", 500.0)
HEALTH_STAGE_TIMEOUT_SECONDS = _env_float("HEALTH_STAGE_TIMEOUT_SECONDS", 5.0)


def disk_usage(path: str) -> Optional[dict]:
    """Free and total space of the filesystem holding `path`, or None."""
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    total = st.f_blocks * st.f_frsize
    free = st.f_bavail * st.f_frsize
    if total <= 0:
        return None
    return {
        "path": path,
        "total_mb": round(total / 1e6, 1),
        "free_mb": round(free / 1e6, 1),
        "free_pct": round(100.0 * free / total, 2),
    }


def check_resources(
    shm_paths: Iterable[str] = (),
    disk_paths: Iterable[str] = (),
    min_shm_free_pct: float = HEALTH_MIN_SHM_FREE_PCT,
    min_disk_free_mb: float = HEALTH_MIN_DISK_FREE_MB,
    usage=disk_usage,
) -> Tuple[List[str], Dict[str, dict]]:
    """Return (reasons, details). Each path that cannot be read is skipped.

    `shm_paths` are checked by percentage (memory-backed filesystems with a
    fixed size); `disk_paths` by free megabytes.
    """
    reasons: List[str] = []
    details: Dict[str, dict] = {}
    seen = set()
    for path in shm_paths:
        info = usage(path)
        if info is None or path in seen:
            continue
        seen.add(path)
        details[path] = info
        if min_shm_free_pct > 0 and info["free_pct"] < min_shm_free_pct:
            reasons.append(
                f"low space on {path}: {info['free_pct']}% free "
                f"(minimum {min_shm_free_pct}%)"
            )
    for path in disk_paths:
        info = usage(path)
        if info is None or path in seen:
            continue
        seen.add(path)
        details[path] = info
        if min_disk_free_mb > 0 and info["free_mb"] < min_disk_free_mb:
            reasons.append(
                f"low disk space on {path}: {info['free_mb']} MB free "
                f"(minimum {min_disk_free_mb:.0f} MB)"
            )
    return reasons, details


def evaluate(
    resource_reasons: List[str],
    stages: Optional[Dict[str, dict]] = None,
) -> Tuple[int, str, List[str]]:
    """Combine resource checks and stage pings into (status_code, status,
    reasons). A stage entry is {"ok": bool, "error": str?, "busy": bool?}."""
    reasons = list(resource_reasons)
    for name, info in (stages or {}).items():
        if not info.get("ok"):
            reasons.append(f"stage {name} not responding: {info.get('error', 'no answer')}")
    if reasons:
        return 503, "degraded", reasons
    return 200, "healthy", []
