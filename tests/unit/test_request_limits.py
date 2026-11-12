"""Simple-mode endpoints: size and length limits, admission (503 with
Retry-After), job dir cleanup, the diarization status and /health.

The pipeline is faked; audio is decoded for real (ffmpeg, or the stub
whisperx decoder when ffmpeg is missing).
"""

import asyncio
import io
import math
import os
import struct
import time
import wave

import httpx
import pytest

from test_model_selection import MODEL_ENV, _drop_app_modules

LIMIT_ENV = ("MAX_FILE_SIZE_MB", "MAX_AUDIO_SECONDS", "MAX_INFLIGHT_AUDIO_SECONDS",
             "MAX_QUEUED_AUDIO_SECONDS", "MAX_QUEUED_REQUESTS", "QUEUE_WAIT_TIMEOUT_SECONDS",
             "JOB_DIR", "HEALTH_MIN_DISK_FREE_MB", "DIARIZE_ON_ERROR")


def _wav(seconds=2.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(i / 7.0)))
                               for i in range(int(seconds * rate))))
    return buf.getvalue()


@pytest.fixture
def app_for(monkeypatch, tmp_path):
    def _make(pipeline_seconds=0.0, result_extra=None, **env):
        for key in MODEL_ENV + LIMIT_ENV:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("JOB_DIR", str(tmp_path / "jobs"))
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        _drop_app_modules()
        import app.main as main
        import app.openai_compat as oc
        os.makedirs(str(tmp_path / "jobs"), exist_ok=True)
        calls = []

        def fake_pipeline(audio, model_name=None, **kwargs):
            calls.append(len(audio))
            time.sleep(pipeline_seconds)
            result = {"language": "en", "segments": [{"text": "hi", "start": 0.0, "end": 1.0}],
                      "word_segments": []}
            result.update(result_extra or {})
            return result, None

        def fake_transcribe_and_align(audio, model, *args, **kwargs):
            calls.append(audio)
            time.sleep(pipeline_seconds)
            return {"language": "en", "segments": [{"text": "hi", "start": 0.0, "end": 1.0}]}

        monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(oc, "_run_transcribe_and_align", fake_transcribe_and_align)
        return main.app, calls, tmp_path / "jobs"
    yield _make
    _drop_app_modules()


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                             timeout=60)


def _post_asr(client, seconds=2.0, params=None):
    return client.post("/asr", params=params or {"diarize": "false"},
                       files={"audio_file": ("a.wav", _wav(seconds), "audio/wav")})


def test_the_stage_gets_the_decoded_audio_and_the_job_dir_is_removed(app_for):
    app, calls, jobs = app_for()

    async def scenario():
        async with _client(app) as c:
            r = await _post_asr(c, 2.0)
            assert r.status_code == 200, r.text
            r = await c.post("/v1/audio/transcriptions", data={"model": "whisper-1"},
                             files={"file": ("a.wav", _wav(1.0), "audio/wav")})
            assert r.status_code == 200, r.text

    asyncio.run(scenario())
    assert calls[0] == 32000                     # 2 s at 16 kHz, loaded in the worker
    assert isinstance(calls[1], str) and calls[1].endswith("audio.s16")
    assert os.listdir(jobs) == []


def test_the_job_dir_is_removed_when_the_pipeline_fails(app_for, monkeypatch):
    app, _, jobs = app_for()
    import app.main as main

    def broken(*a, **k):
        raise RuntimeError("stage failed")

    monkeypatch.setattr(main, "run_pipeline", broken)

    async def scenario():
        async with _client(app) as c:
            r = await _post_asr(c)
            assert r.status_code == 500

    asyncio.run(scenario())
    assert os.listdir(jobs) == []


def test_an_upload_over_max_file_size_is_a_413(app_for):
    app, calls, jobs = app_for(MAX_FILE_SIZE_MB="1")

    async def scenario():
        async with _client(app) as c:
            r = await _post_asr(c, 60.0)        # ~1.9 MB
            assert r.status_code == 413, r.text
            r = await c.post("/v1/audio/transcriptions", data={"model": "whisper-1"},
                             files={"file": ("a.wav", _wav(60.0), "audio/wav")})
            assert r.status_code == 413
            assert r.json()["error"]["code"] == "file_too_large"

    asyncio.run(scenario())
    assert calls == [] and os.listdir(jobs) == []


def test_audio_over_max_audio_seconds_is_a_413(app_for):
    app, calls, _ = app_for(MAX_AUDIO_SECONDS="5")

    async def scenario():
        async with _client(app) as c:
            r = await _post_asr(c, 8.0)
            assert r.status_code == 413, r.text
            assert "too long" in r.json()["detail"]
            r = await c.post("/v1/audio/transcriptions", data={"model": "whisper-1"},
                             files={"file": ("a.wav", _wav(8.0), "audio/wav")})
            assert r.status_code == 413 and r.json()["error"]["code"] == "audio_too_long"
            r = await _post_asr(c, 3.0)
            assert r.status_code == 200

    asyncio.run(scenario())
    assert len(calls) == 1


def test_overload_is_a_503_with_retry_after_and_the_queue_drains(app_for):
    """In-flight budget of 3 s, queue of one: of three 2 s requests the first
    runs, the second waits, the third is refused."""
    app, calls, jobs = app_for(pipeline_seconds=1.5, MAX_INFLIGHT_AUDIO_SECONDS="3",
                               MAX_QUEUED_REQUESTS="1")

    async def scenario():
        async with _client(app) as c:
            first = asyncio.create_task(_post_asr(c, 2.0))
            await asyncio.sleep(0.3)
            second = asyncio.create_task(_post_asr(c, 2.0))
            await asyncio.sleep(0.3)
            third = await _post_asr(c, 2.0)
            assert third.status_code == 503, third.text
            assert int(third.headers["Retry-After"]) >= 30
            body = third.json()
            assert body["error"] == "busy" and body["queue"]["queued_requests"] == 1

            health = await c.get("/health")
            assert health.status_code == 200
            assert health.json()["admission"]["saturated"] is True

            r1, r2 = await first, await second
            assert r1.status_code == r2.status_code == 200
            health = (await c.get("/health")).json()["admission"]
            assert health["inflight_requests"] == 0 and health["queued_requests"] == 0
            assert health["admission_rejections_total"] == 1

    asyncio.run(scenario())
    assert len(calls) == 2 and os.listdir(jobs) == []


def test_openai_endpoint_overload_is_a_503(app_for):
    app, _, _ = app_for(pipeline_seconds=1.5, MAX_INFLIGHT_AUDIO_SECONDS="1",
                        MAX_QUEUED_REQUESTS="1", QUEUE_WAIT_TIMEOUT_SECONDS="0.2")

    async def scenario():
        async with _client(app) as c:
            first = asyncio.create_task(_post_asr(c, 2.0))
            await asyncio.sleep(0.3)
            r = await c.post("/v1/audio/transcriptions", data={"model": "whisper-1"},
                             files={"file": ("a.wav", _wav(1.0), "audio/wav")})
            assert r.status_code == 503
            assert r.headers["Retry-After"]
            assert r.json()["error"]["code"] == "server_busy"
            assert (await first).status_code == 200

    asyncio.run(scenario())


@pytest.mark.parametrize("output", ["json", "text", "srt"])
def test_a_failed_diarization_is_reported(app_for, output):
    app, _, _ = app_for(result_extra={"_diarization": {"status": "failed", "error": "CUDA out of memory"}})

    async def scenario():
        async with _client(app) as c:
            r = await _post_asr(c, params={"diarize": "true", "output": output})
            assert r.status_code == 200
            assert r.headers["X-Diarization-Status"] == "failed"
            if output == "json":
                assert r.json()["diarization"] == {"status": "failed", "error": "CUDA out of memory"}

    asyncio.run(scenario())


def test_health_is_503_when_the_job_disk_is_low(app_for):
    app, _, _ = app_for(HEALTH_MIN_DISK_FREE_MB="1e15")

    async def scenario():
        async with _client(app) as c:
            r = await c.get("/health")
            assert r.status_code == 503
            body = r.json()
            assert body["status"] == "degraded" and "disk" in body["reasons"][0]

    asyncio.run(scenario())
