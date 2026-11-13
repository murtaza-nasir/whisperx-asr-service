"""Asynchronous jobs over HTTP, on stand-in ML packages.

Submit, poll, fetch, deduplicate and cancel against the real app, plus the
checks that the synchronous surface is unchanged: every field the previous
version answered with is still there, and the new ones are additions.
"""

import io
import math
import os
import socket
import struct
import subprocess
import sys
import time
import wave

import httpx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
MINIMAL_ENV = os.path.join(REPO_ROOT, "tests", "defaults", "minimal.env")

POLL_TIMEOUT = 120


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wav(seconds=3.0, freq=440.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate)))
                          for i in range(int(seconds * rate)))
        w.writeframes(frames)
    return buf.getvalue()


def _start(overrides=()):
    port = _free_port()
    argv = [sys.executable, os.path.join(HERE, "stub_server.py"), "--env-file", MINIMAL_ENV,
            "--port", str(port), "--timeout", "60"]
    for item in overrides:
        argv += ["--set", item]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=REPO_ROOT)
    deadline, log = time.time() + 90, []
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        log.append(line)
        if line.startswith("READY "):
            break
    if proc.poll() is not None:
        pytest.fail("stub server exited:\n" + "".join(log[-60:]))
    return proc, f"http://127.0.0.1:{port}"


def _stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def server():
    """The job API with the listing on, so the tests can read /jobs."""
    proc, url = _start(overrides=["JOBS_LIST_ENABLED=true"])
    yield url
    _stop(proc)


@pytest.fixture(scope="module")
def default_server():
    """Defaults as shipped: jobs on, the listing off."""
    proc, url = _start()
    yield url
    _stop(proc)


@pytest.fixture(scope="module")
def short_limit_server():
    """A server that refuses anything longer than one second."""
    proc, url = _start(overrides=["MAX_AUDIO_SECONDS=1", "JOBS_LIST_ENABLED=true"])
    yield url
    _stop(proc)


@pytest.fixture(scope="module")
def no_jobs_server():
    proc, url = _start(overrides=["ASYNC_JOBS_ENABLED=false"])
    yield url
    _stop(proc)


def submit(url, audio=None, params=None, headers=None, filename="a.wav"):
    return httpx.post(
        url + "/jobs",
        params=params or {},
        files={"audio_file": (filename, audio if audio is not None else _wav(), "audio/wav")},
        headers=headers or {},
        timeout=60,
    )


def wait_for(url, job_id, statuses=("done", "error", "cancelled", "interrupted")):
    deadline = time.time() + POLL_TIMEOUT
    body = {}
    while time.time() < deadline:
        r = httpx.get(f"{url}/jobs/{job_id}", timeout=10)
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in statuses:
            return body
        time.sleep(0.2)
    pytest.fail(f"job {job_id} stayed {body.get('status')}")


# ----------------------------------------------------------------------
# The happy path
# ----------------------------------------------------------------------
def test_a_submit_returns_a_job(server):
    r = submit(server)
    assert r.status_code == 202, r.text
    body = r.json()
    assert len(body["job_id"]) == 32
    assert body["status"] in ("queued", "running")
    assert body["deduplicated"] is False
    assert body["status_url"] == f"/jobs/{body['job_id']}"
    assert body["result_url"] == f"/jobs/{body['job_id']}/result"
    assert body["poll_after_seconds"] >= 1
    assert r.headers["location"] == body["status_url"]
    assert body["audio_seconds"] == pytest.approx(3.0, abs=0.2)


def test_the_job_finishes_and_the_result_matches_the_sync_endpoint(server):
    audio = _wav(2.0)
    sync = httpx.post(server + "/asr", params={"output": "json"},
                      files={"audio_file": ("a.wav", audio, "audio/wav")}, timeout=60)
    assert sync.status_code == 200, sync.text

    job_id = submit(server, audio=audio).json()["job_id"]
    assert wait_for(server, job_id)["status"] == "done"
    r = httpx.get(f"{server}/jobs/{job_id}/result", timeout=30)
    assert r.status_code == 200, r.text
    assert r.json() == sync.json()


def test_progress_reaches_one_and_says_it_is_no_longer_an_estimate(server):
    job_id = submit(server).json()["job_id"]
    body = wait_for(server, job_id)
    assert body["status"] == "done"
    assert body["progress"]["fraction"] == 1.0
    assert body["progress"]["fraction_estimated"] is False
    assert body["progress"]["eta_seconds"] == 0
    assert body["finished_at"] and body["started_at"]
    assert body["expires_at"] > time.time()


def test_the_format_is_chosen_when_the_result_is_fetched(server):
    job_id = submit(server).json()["job_id"]
    wait_for(server, job_id)
    as_json = httpx.get(f"{server}/jobs/{job_id}/result", timeout=30)
    as_srt = httpx.get(f"{server}/jobs/{job_id}/result",
                       params={"output_format": "srt"}, timeout=30)
    as_vtt = httpx.get(f"{server}/jobs/{job_id}/result",
                       params={"output": "vtt"}, timeout=30)
    assert "segments" in as_json.json()
    assert as_srt.json()["srt"].strip()
    assert as_vtt.json()["vtt"].startswith("WEBVTT")


def test_an_unfinished_job_answers_202_with_retry_after(server):
    job_id = submit(server, audio=_wav(5.0)).json()["job_id"]
    r = httpx.get(f"{server}/jobs/{job_id}/result", timeout=10)
    if r.status_code == 202:
        assert int(r.headers["retry-after"]) >= 1
        assert r.json()["status"] in ("queued", "running")
    else:
        assert r.status_code == 200  # already finished, the stub is fast
    wait_for(server, job_id)


def test_the_job_list_shows_it(server):
    job_id = submit(server).json()["job_id"]
    wait_for(server, job_id)
    body = httpx.get(server + "/jobs", timeout=10).json()
    assert job_id in {j["job_id"] for j in body["jobs"]}
    assert set(body["counts"]) == {"total", "active", "queued", "running"}


# ----------------------------------------------------------------------
# Deduplication
# ----------------------------------------------------------------------
def test_the_same_bytes_and_parameters_reuse_the_job(server):
    audio = _wav(2.5, freq=330.0)
    first = submit(server, audio=audio).json()
    second = submit(server, audio=audio)
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["job_id"] == first["job_id"]
    assert body["deduplicated"] is True
    assert body["submissions"] >= 2


def test_other_parameters_start_another_job(server):
    audio = _wav(2.5, freq=350.0)
    first = submit(server, audio=audio).json()
    second = submit(server, audio=audio, params={"diarize": "false"}).json()
    assert second["job_id"] != first["job_id"]
    assert second["deduplicated"] is False


def test_other_bytes_start_another_job(server):
    first = submit(server, audio=_wav(2.0, freq=300.0)).json()
    second = submit(server, audio=_wav(2.0, freq=500.0)).json()
    assert first["job_id"] != second["job_id"]


def test_an_idempotency_key_reuses_the_job_for_other_bytes(server):
    headers = {"Idempotency-Key": "recording-41"}
    first = submit(server, audio=_wav(2.0, freq=210.0), headers=headers).json()
    second = submit(server, audio=_wav(3.0, freq=640.0), headers=headers)
    assert second.status_code == 200
    assert second.json()["job_id"] == first["job_id"]
    assert second.json()["deduplicated"] is True


def test_an_idempotency_key_with_other_parameters_starts_another_job(server):
    headers = {"Idempotency-Key": "recording-42"}
    audio = _wav(2.0, freq=220.0)
    first = submit(server, audio=audio, headers=headers).json()
    second = submit(server, audio=audio, params={"word_timestamps": "false"},
                    headers=headers).json()
    assert second["job_id"] != first["job_id"]


def test_a_duplicate_of_a_finished_job_needs_no_second_run(server):
    audio = _wav(2.0, freq=777.0)
    first = submit(server, audio=audio).json()["job_id"]
    wait_for(server, first)
    again = submit(server, audio=audio).json()
    assert again["job_id"] == first
    assert again["status"] == "done"
    assert httpx.get(f"{server}/jobs/{first}/result", timeout=30).status_code == 200


# ----------------------------------------------------------------------
# Cancel, errors, absent jobs
# ----------------------------------------------------------------------
def test_cancelling_a_finished_job_drops_the_transcript(server):
    job_id = submit(server, audio=_wav(2.0, freq=990.0)).json()["job_id"]
    wait_for(server, job_id)
    r = httpx.delete(f"{server}/jobs/{job_id}", timeout=30)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "cancelled"
    assert httpx.get(f"{server}/jobs/{job_id}/result", timeout=10).status_code == 409


def test_a_cancelled_job_is_not_reused(server):
    audio = _wav(2.0, freq=995.0)
    job_id = submit(server, audio=audio).json()["job_id"]
    wait_for(server, job_id)
    httpx.delete(f"{server}/jobs/{job_id}", timeout=30)
    again = submit(server, audio=audio).json()
    assert again["job_id"] != job_id


def test_an_unknown_job_is_a_404(server):
    assert httpx.get(server + "/jobs/" + "0" * 32, timeout=10).status_code == 404
    assert httpx.get(server + "/jobs/" + "0" * 32 + "/result", timeout=10).status_code == 404
    assert httpx.delete(server + "/jobs/" + "0" * 32, timeout=10).status_code == 404


def test_an_unknown_model_is_a_400(server):
    r = submit(server, params={"model": "large-v4"})
    assert r.status_code == 400, r.text


def test_audio_longer_than_the_limit_is_a_413(short_limit_server):
    r = submit(short_limit_server, audio=_wav(4.0))
    assert r.status_code == 413, r.text
    body = r.json()
    assert "Maximum allowed" in (body.get("detail") or "")


def test_the_limit_is_reported_before_any_work(short_limit_server):
    """413 arrives from the submit itself, so no job is created."""
    before = httpx.get(short_limit_server + "/jobs", timeout=10).json()["counts"]["total"]
    submit(short_limit_server, audio=_wav(4.0))
    after = httpx.get(short_limit_server + "/jobs", timeout=10).json()["counts"]["total"]
    assert after == before


def test_the_listing_is_off_by_default(default_server):
    """A listing would make job ids enumerable, so it needs the setting."""
    job_id = submit(default_server).json()["job_id"]
    assert httpx.get(default_server + "/jobs", timeout=10).status_code == 404
    caps = httpx.get(default_server + "/health", timeout=10).json()["capabilities"]
    assert caps["async_jobs"] is True
    assert caps["list_jobs"] is False
    # The job itself is still reachable by its id, and the counts have no ids.
    assert httpx.get(f"{default_server}/jobs/{job_id}", timeout=10).status_code == 200
    counts = httpx.get(default_server + "/health", timeout=10).json()["jobs"]
    assert set(counts) == {"total", "active", "queued", "running"}
    wait_for(default_server, job_id)


def test_the_listing_reports_the_capability_when_on(server):
    caps = httpx.get(server + "/health", timeout=10).json()["capabilities"]
    assert caps["list_jobs"] is True


def test_the_surface_can_be_turned_off(no_jobs_server):
    assert submit(no_jobs_server).status_code == 404
    assert httpx.get(no_jobs_server + "/jobs", timeout=10).status_code == 404
    caps = httpx.get(no_jobs_server + "/health", timeout=10).json()["capabilities"]
    assert caps["async_jobs"] is False


def test_the_synchronous_endpoint_still_works_with_jobs_off(no_jobs_server):
    r = httpx.post(no_jobs_server + "/asr", params={"output": "json"},
                   files={"audio_file": ("a.wav", _wav(1.0), "audio/wav")}, timeout=60)
    assert r.status_code == 200, r.text
    assert r.json()["segments"]


# ----------------------------------------------------------------------
# Backwards compatibility of the existing surface
# ----------------------------------------------------------------------
def test_root_keeps_its_fields_and_gains_capabilities(server):
    body = httpx.get(server + "/", timeout=10).json()
    for key in ("status", "service", "device", "compute_type", "serve_mode"):
        assert key in body, key
    assert body["status"] == "running"
    assert body["capabilities"]["async_jobs"] is True


def test_health_keeps_its_fields(server):
    body = httpx.get(server + "/health", timeout=10).json()
    for key in ("status", "device", "loaded_models", "serve_mode", "reasons",
                "resources", "admission"):
        assert key in body, key
    assert set(body["admission"]) >= {
        "inflight_requests", "inflight_audio_seconds", "queued_requests",
        "queued_audio_seconds", "estimated_wait_seconds",
        "admission_rejections_total", "saturated", "limits",
    }
    assert set(body["admission"]["limits"]) == {
        "max_audio_seconds", "max_inflight_audio_seconds", "max_queued_audio_seconds",
        "max_queued_requests", "queue_wait_timeout_seconds",
    }


def test_health_gains_capabilities_and_job_counts(server):
    body = httpx.get(server + "/health", timeout=10).json()
    assert body["capabilities"]["jobs_api_version"] == 1
    assert set(body["jobs"]) == {"total", "active", "queued", "running"}


def test_the_asr_response_shape_is_unchanged(server):
    r = httpx.post(server + "/asr",
                   params={"output": "json", "diarize": "true",
                           "return_speaker_embeddings": "true", "min_speakers": "1",
                           "max_speakers": "1"},
                   files={"audio_file": ("a.wav", _wav(2.0), "audio/wav")}, timeout=60)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"text", "language", "segments", "word_segments",
                         "speaker_embeddings"}
    assert body["segments"] and "text" in body["segments"][0]


def test_the_openai_endpoint_is_unchanged(server):
    r = httpx.post(server + "/v1/audio/transcriptions",
                   data={"model": "whisper-1", "response_format": "json"},
                   files={"file": ("a.wav", _wav(2.0), "audio/wav")}, timeout=60)
    assert r.status_code == 200, r.text
    assert r.json()["text"].strip()


def test_queue_metrics_is_unchanged(server):
    body = httpx.get(server + "/queue-metrics", timeout=10).json()
    assert {"serve_mode", "device", "loaded_models", "queue"} <= set(body)


def test_metrics_still_renders(server):
    r = httpx.get(server + "/metrics", timeout=10)
    assert r.status_code == 200
    assert "whisperx_" in r.text
