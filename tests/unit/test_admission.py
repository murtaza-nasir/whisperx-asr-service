"""Admission control by audio seconds: admit, queue, reject, release."""

import asyncio

import pytest

from app.admission import (
    AdmissionController,
    AdmissionRejected,
    AudioTooLong,
    RETRY_AFTER_MAX_SECONDS,
    RETRY_AFTER_MIN_SECONDS,
)


def run(coro):
    return asyncio.run(coro)


def test_everything_is_off_by_default(monkeypatch):
    for key in ("MAX_AUDIO_SECONDS", "MAX_INFLIGHT_AUDIO_SECONDS", "MAX_QUEUED_AUDIO_SECONDS",
                "MAX_QUEUED_REQUESTS", "QUEUE_WAIT_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    c = AdmissionController.from_env()
    assert c.settings() == {
        "max_audio_seconds": 0, "max_inflight_audio_seconds": 0,
        "max_queued_audio_seconds": 0, "max_queued_requests": 0,
        "queue_wait_timeout_seconds": 0,
    }

    async def scenario():
        for _ in range(50):
            await c.acquire(6 * 3600)
        assert c.inflight_requests == 50 and c.queued_requests == 0
        assert not c.saturated()

    run(scenario())


@pytest.mark.parametrize("value", ["", "  ", "abc"])
def test_empty_or_bad_values_fall_back_to_off(monkeypatch, value):
    monkeypatch.setenv("MAX_INFLIGHT_AUDIO_SECONDS", value)
    monkeypatch.setenv("MAX_QUEUED_REQUESTS", value)
    c = AdmissionController.from_env()
    assert c.max_inflight == 0 and c.max_queued_requests == 0


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MAX_AUDIO_SECONDS", "28800")
    monkeypatch.setenv("MAX_INFLIGHT_AUDIO_SECONDS", "21600")
    monkeypatch.setenv("MAX_QUEUED_AUDIO_SECONDS", "43200")
    monkeypatch.setenv("MAX_QUEUED_REQUESTS", "8")
    monkeypatch.setenv("QUEUE_WAIT_TIMEOUT_SECONDS", "1200")
    c = AdmissionController.from_env()
    assert c.settings() == {
        "max_audio_seconds": 28800, "max_inflight_audio_seconds": 21600,
        "max_queued_audio_seconds": 43200, "max_queued_requests": 8,
        "queue_wait_timeout_seconds": 1200,
    }


def test_a_file_over_the_length_limit_is_refused():
    c = AdmissionController(max_audio_seconds=3600)
    with pytest.raises(AudioTooLong):
        c.check_length(3601)
    c.check_length(3600)
    c.check_length(None)  # unknown length is checked after decoding
    with pytest.raises(AudioTooLong):
        run(c.acquire(7200))


def test_admit_queue_then_reject_with_retry_after():
    """4 long files against a 4 h in-flight budget and a queue of one."""
    c = AdmissionController(max_inflight_audio_seconds=4 * 3600, max_queued_requests=1)

    async def scenario():
        assert await c.acquire(2 * 3600) == 7200      # 1: admitted
        assert await c.acquire(2 * 3600) == 7200      # 2: admitted, budget now full
        third = asyncio.create_task(c.acquire(3 * 3600))
        await asyncio.sleep(0)
        assert not third.done() and c.queued_requests == 1
        with pytest.raises(AdmissionRejected) as info:
            await c.acquire(1800)                     # 4: queue full
        rejected = info.value
        assert rejected.reason == "queue_full"
        assert RETRY_AFTER_MIN_SECONDS <= rejected.retry_after <= RETRY_AFTER_MAX_SECONDS
        body = rejected.body()
        assert body["error"] == "busy" and body["queue"]["queued_requests"] == 1
        assert c.saturated()

        c.release(7200)                               # 1 finishes: 3 still does not fit
        await asyncio.sleep(0)
        assert not third.done()
        c.release(7200)                               # 2 finishes: 3 runs
        assert await third == 3 * 3600
        assert c.inflight_requests == 1 and c.queued_requests == 0
        c.release(3 * 3600)
        assert c.inflight_seconds == 0 and c.inflight_requests == 0
        assert c.rejections == 1

    run(scenario())


def test_queue_is_first_in_first_out():
    c = AdmissionController(max_inflight_audio_seconds=100)

    async def scenario():
        await c.acquire(100)
        order = []

        async def wait(name, seconds):
            await c.acquire(seconds)
            order.append(name)

        big = asyncio.create_task(wait("big", 90))
        await asyncio.sleep(0)
        small = asyncio.create_task(wait("small", 5))
        await asyncio.sleep(0)
        c.release(100)
        await asyncio.gather(big, small)
        assert order == ["big", "small"]

    run(scenario())


def test_an_oversized_file_runs_when_the_pipeline_is_idle():
    c = AdmissionController(max_inflight_audio_seconds=3600)

    async def scenario():
        assert await c.acquire(6 * 3600) == 6 * 3600
        waiting = asyncio.create_task(c.acquire(60))
        await asyncio.sleep(0)
        assert not waiting.done()
        c.release(6 * 3600)
        await waiting

    run(scenario())


def test_queued_audio_seconds_limit():
    c = AdmissionController(max_inflight_audio_seconds=100, max_queued_audio_seconds=150)

    async def scenario():
        await c.acquire(100)
        first = asyncio.create_task(c.acquire(100))
        await asyncio.sleep(0)
        with pytest.raises(AdmissionRejected) as info:
            await c.acquire(60)
        assert info.value.reason == "queue_audio_full"
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    run(scenario())


def test_a_wait_that_times_out_is_rejected_and_leaves_the_queue():
    c = AdmissionController(max_inflight_audio_seconds=100, queue_wait_timeout_seconds=0.05)

    async def scenario():
        await c.acquire(100)
        with pytest.raises(AdmissionRejected) as info:
            await c.acquire(50)
        assert info.value.reason == "queue_timeout"
        assert c.queued_requests == 0
        c.release(100)
        assert await c.acquire(50) == 50

    run(scenario())


def test_a_cancelled_waiter_gives_its_place_back():
    c = AdmissionController(max_inflight_audio_seconds=100)

    async def scenario():
        await c.acquire(100)
        waiter = asyncio.create_task(c.acquire(80))
        behind = asyncio.create_task(c.acquire(10))
        await asyncio.sleep(0)
        assert c.queued_requests == 2
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        c.release(100)
        await behind
        assert c.inflight_requests == 1 and c.inflight_seconds == 10

    run(scenario())


def test_the_slot_is_released_on_error_and_cancel():
    c = AdmissionController(max_inflight_audio_seconds=100)

    async def failing():
        async with c.slot(60):
            raise RuntimeError("stage failed")

    async def scenario():
        with pytest.raises(RuntimeError):
            await failing()
        assert c.inflight_requests == 0

        started = asyncio.Event()

        async def long_request():
            async with c.slot(60):
                started.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(long_request())
        await started.wait()
        assert c.inflight_requests == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert c.inflight_requests == 0 and c.inflight_seconds == 0

    run(scenario())


def test_retry_after_follows_measured_throughput():
    c = AdmissionController(max_inflight_audio_seconds=100)
    for _ in range(20):
        c.observe(3600, 36)          # 100x realtime
    c.inflight_seconds, c.inflight_requests = 36000, 1
    assert 300 <= c.retry_after() <= 400
    c.inflight_seconds = 10
    assert c.retry_after() == RETRY_AFTER_MIN_SECONDS
    c.inflight_seconds = 10 ** 9
    assert c.retry_after() == RETRY_AFTER_MAX_SECONDS
