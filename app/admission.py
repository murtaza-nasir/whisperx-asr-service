"""
Admission control by audio duration.

Requests are admitted while the audio seconds already in the pipeline plus
the new file stay within MAX_INFLIGHT_AUDIO_SECONDS. Other requests wait in
a bounded FIFO queue (MAX_QUEUED_REQUESTS, MAX_QUEUED_AUDIO_SECONDS) for at
most QUEUE_WAIT_TIMEOUT_SECONDS. When the queue is full or the wait times
out, the request is rejected with a Retry-After estimate (HTTP 503). A file
longer than MAX_AUDIO_SECONDS is refused outright (HTTP 413).

A single file larger than the in-flight limit is admitted once the pipeline
is idle, so it cannot wait forever.

Every limit is off (0) by default. No ML imports, so CI tests run it as is.
"""

import asyncio
import collections
import contextlib
import logging
import os
import time
from typing import Deque, Optional

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        logger.warning(f"{name}={value!r} is not a number; using {default}")
        return default


# Marks "use the configured value" so a caller can pass wait_timeout=None
# to wait without a deadline.
_UNSET = object()

RETRY_AFTER_MIN_SECONDS = 30
RETRY_AFTER_MAX_SECONDS = 3600
# Starting throughput guess (audio seconds per wall-clock second) until the
# first requests have been measured.
INITIAL_REALTIME_FACTOR = 20.0


class AudioTooLong(Exception):
    def __init__(self, seconds: float, limit: float):
        self.seconds = seconds
        self.limit = limit
        super().__init__(
            f"Audio too long ({seconds:.0f}s). Maximum allowed: {limit:.0f}s."
        )


class AdmissionRejected(Exception):
    def __init__(self, reason: str, retry_after: int, snapshot: dict):
        self.reason = reason
        self.retry_after = retry_after
        self.snapshot = snapshot
        super().__init__(f"Server busy ({reason}); retry after {retry_after}s")

    def body(self) -> dict:
        return {
            "error": "busy",
            "reason": self.reason,
            "retry_after": self.retry_after,
            "queue": self.snapshot,
        }


class _Waiter:
    __slots__ = ("seconds", "future", "enqueued")

    def __init__(self, seconds: float, future: asyncio.Future):
        self.seconds = seconds
        self.future = future
        self.enqueued = time.monotonic()


class AdmissionController:
    def __init__(
        self,
        max_audio_seconds: float = 0,
        max_inflight_audio_seconds: float = 0,
        max_queued_audio_seconds: float = 0,
        max_queued_requests: int = 0,
        queue_wait_timeout_seconds: float = 0,
    ):
        self.max_audio_seconds = max_audio_seconds
        self.max_inflight = max_inflight_audio_seconds
        self.max_queued_seconds = max_queued_audio_seconds
        self.max_queued_requests = int(max_queued_requests)
        self.wait_timeout = queue_wait_timeout_seconds
        self.inflight_seconds = 0.0
        self.inflight_requests = 0
        self._queue: Deque[_Waiter] = collections.deque()
        self.rejections = 0
        self._rate = INITIAL_REALTIME_FACTOR

    @classmethod
    def from_env(cls) -> "AdmissionController":
        return cls(
            max_audio_seconds=_env_float("MAX_AUDIO_SECONDS", 0),
            max_inflight_audio_seconds=_env_float("MAX_INFLIGHT_AUDIO_SECONDS", 0),
            max_queued_audio_seconds=_env_float("MAX_QUEUED_AUDIO_SECONDS", 0),
            max_queued_requests=int(_env_float("MAX_QUEUED_REQUESTS", 0)),
            queue_wait_timeout_seconds=_env_float("QUEUE_WAIT_TIMEOUT_SECONDS", 0),
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def queued_requests(self) -> int:
        return len(self._queue)

    @property
    def queued_seconds(self) -> float:
        return sum(w.seconds for w in self._queue)

    def settings(self) -> dict:
        return {
            "max_audio_seconds": self.max_audio_seconds,
            "max_inflight_audio_seconds": self.max_inflight,
            "max_queued_audio_seconds": self.max_queued_seconds,
            "max_queued_requests": self.max_queued_requests,
            "queue_wait_timeout_seconds": self.wait_timeout,
        }

    def snapshot(self) -> dict:
        return {
            "inflight_requests": self.inflight_requests,
            "inflight_audio_seconds": round(self.inflight_seconds, 1),
            "queued_requests": self.queued_requests,
            "queued_audio_seconds": round(self.queued_seconds, 1),
            "estimated_wait_seconds": self.estimate_wait(),
            "admission_rejections_total": self.rejections,
        }

    def saturated(self) -> bool:
        if not self.max_inflight:
            return False
        return self.inflight_seconds >= self.max_inflight or bool(self._queue)

    def estimate_wait(self, extra_seconds: float = 0.0) -> int:
        backlog = self.inflight_seconds + self.queued_seconds + extra_seconds
        return int(backlog / max(self._rate, 0.01))

    def retry_after(self, extra_seconds: float = 0.0) -> int:
        return int(min(RETRY_AFTER_MAX_SECONDS,
                       max(RETRY_AFTER_MIN_SECONDS, self.estimate_wait(extra_seconds))))

    def observe(self, audio_seconds: float, elapsed_seconds: float) -> None:
        """Feed one finished request into the throughput estimate."""
        if audio_seconds <= 0 or elapsed_seconds <= 0:
            return
        rate = audio_seconds / elapsed_seconds
        self._rate = 0.7 * self._rate + 0.3 * rate

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------
    def check_length(self, seconds: Optional[float]) -> None:
        if seconds is not None and self.max_audio_seconds and seconds > self.max_audio_seconds:
            raise AudioTooLong(seconds, self.max_audio_seconds)

    def _fits(self, seconds: float) -> bool:
        if not self.max_inflight:
            return True
        if self.inflight_requests == 0:
            return True  # an oversized file still runs when the pipeline is idle
        return self.inflight_seconds + seconds <= self.max_inflight

    def _grant(self, seconds: float) -> None:
        self.inflight_seconds += seconds
        self.inflight_requests += 1

    def _reject(self, reason: str, seconds: float) -> AdmissionRejected:
        self.rejections += 1
        return AdmissionRejected(reason, self.retry_after(seconds), self.snapshot())

    def _wake(self) -> None:
        while self._queue:
            head = self._queue[0]
            if head.future.done():  # cancelled waiter
                self._queue.popleft()
                continue
            if not self._fits(head.seconds):
                return
            self._queue.popleft()
            self._grant(head.seconds)
            head.future.set_result(True)

    async def acquire(self, seconds: float, *, enforce_queue_bounds: bool = True,
                      wait_timeout=_UNSET) -> float:
        """Wait for room for `seconds` of audio. Returns the reserved amount,
        to be passed to release().

        `enforce_queue_bounds=False` and `wait_timeout=None` are used by
        asynchronous jobs, which are already accepted and bounded by
        MAX_ASYNC_JOBS: they wait for the in-flight limit without a deadline
        instead of being rejected after the fact. Synchronous callers pass
        neither and behave exactly as before.
        """
        seconds = max(0.0, float(seconds or 0.0))
        timeout = self.wait_timeout if wait_timeout is _UNSET else wait_timeout
        self.check_length(seconds)
        if not self._queue and self._fits(seconds):
            self._grant(seconds)
            return seconds
        if enforce_queue_bounds:
            if self.max_queued_requests and len(self._queue) >= self.max_queued_requests:
                raise self._reject("queue_full", seconds)
            if self.max_queued_seconds and self.queued_seconds + seconds > self.max_queued_seconds:
                raise self._reject("queue_audio_full", seconds)

        loop = asyncio.get_running_loop()
        waiter = _Waiter(seconds, loop.create_future())
        self._queue.append(waiter)
        try:
            if timeout:
                await asyncio.wait_for(asyncio.shield(waiter.future), timeout=timeout)
            else:
                await waiter.future
        except asyncio.TimeoutError:
            if waiter.future.done() and not waiter.future.cancelled():
                return seconds  # granted at the same moment the timer fired
            waiter.future.cancel()
            self._drop(waiter)
            raise self._reject("queue_timeout", seconds)
        except BaseException:
            if waiter.future.done() and not waiter.future.cancelled():
                # Granted, then the request was cancelled: give the room back.
                self.release(seconds)
            else:
                waiter.future.cancel()
                self._drop(waiter)
            raise
        return seconds

    def _drop(self, waiter: _Waiter) -> None:
        try:
            self._queue.remove(waiter)
        except ValueError:
            pass
        self._wake()

    def release(self, seconds: float, elapsed_seconds: Optional[float] = None) -> None:
        self.inflight_seconds = max(0.0, self.inflight_seconds - seconds)
        self.inflight_requests = max(0, self.inflight_requests - 1)
        if self.inflight_requests == 0:
            self.inflight_seconds = 0.0
        if elapsed_seconds is not None:
            self.observe(seconds, elapsed_seconds)
        self._wake()

    @contextlib.asynccontextmanager
    async def slot(self, seconds: float, **kwargs):
        reserved = await self.acquire(seconds, **kwargs)
        started = time.monotonic()
        ok = False
        try:
            yield reserved
            ok = True
        finally:
            self.release(reserved, (time.monotonic() - started) if ok else None)


_controller: Optional[AdmissionController] = None


def get_controller() -> AdmissionController:
    """Process-wide controller (one per ingress or simple-mode server)."""
    global _controller
    if _controller is None:
        _controller = AdmissionController.from_env()
    return _controller


def reset_controller() -> None:
    global _controller
    _controller = None
