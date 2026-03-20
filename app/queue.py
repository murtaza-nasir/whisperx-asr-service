"""
Simple async queue with GPU semaphore for running without Ray.

Provides backpressure and ensures only one pipeline invocation runs on the GPU
at a time.  Used when SERVE_MODE=simple (default).
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "32"))
GPU_CONCURRENCY = int(os.getenv("GPU_CONCURRENCY", "20"))

# Thread pool for running blocking pipeline calls without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=GPU_CONCURRENCY)
_gpu_semaphore: Optional[asyncio.Semaphore] = None

# Metrics counters
_requests_queued = 0
_requests_in_flight = 0


def _get_semaphore() -> asyncio.Semaphore:
    global _gpu_semaphore
    if _gpu_semaphore is None:
        _gpu_semaphore = asyncio.Semaphore(GPU_CONCURRENCY)
    return _gpu_semaphore


async def run_in_queue(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Submit a blocking function to run on the GPU with backpressure.

    Acquires the GPU semaphore, then runs ``fn(*args, **kwargs)`` in a
    thread-pool executor so the event loop stays responsive.
    """
    global _requests_queued, _requests_in_flight

    sem = _get_semaphore()
    _requests_queued += 1
    logger.debug(f"Queue: {_requests_queued} waiting, {_requests_in_flight} in flight")

    await sem.acquire()
    _requests_queued -= 1
    _requests_in_flight += 1
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor, lambda: fn(*args, **kwargs)
        )
        return result
    finally:
        _requests_in_flight -= 1
        sem.release()


def get_queue_metrics() -> dict:
    """Return current queue/in-flight counters."""
    return {
        "requests_queued": _requests_queued,
        "requests_in_flight": _requests_in_flight,
        "gpu_concurrency": GPU_CONCURRENCY,
        "max_queue_size": MAX_QUEUE_SIZE,
    }
