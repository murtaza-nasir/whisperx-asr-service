"""Asynchronous jobs: keys, state, persistence, recovery, progress, bounds."""

import asyncio
import json
import os
import time

import pytest

import app.jobs as jobs_mod
from app.admission import AdmissionController, AdmissionRejected
from app.audio_io import AudioDecodeError, UploadTooLarge
from app.job_runner import classify_error, poll_after


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in ("ASYNC_JOBS_ENABLED", "MAX_ASYNC_JOBS", "JOB_RESULT_TTL_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    jobs_mod.reset_store()
    yield
    jobs_mod.reset_store()


@pytest.fixture
def store(tmp_path):
    return jobs_mod.JobStore(root=str(tmp_path))


def make_job(store, params=None, key="k1", filename="a.mp3", seconds=600.0):
    job = store.create(params=params or {"model": "large-v3"}, key=key, filename=filename)
    job.audio_seconds = seconds
    job.size_bytes = 1234
    job.save()
    return job


# ----------------------------------------------------------------------
# Defaults and capabilities
# ----------------------------------------------------------------------
def test_async_jobs_are_on_by_default():
    assert jobs_mod.async_jobs_enabled() is True
    caps = jobs_mod.capabilities()
    assert caps["async_jobs"] is True
    assert caps["jobs_api_version"] == jobs_mod.JOBS_API_VERSION
    assert caps["dedupe"] is True and caps["idempotency_key"] is True
    assert caps["cancel"] is True and caps["progress"] is True
    assert caps["job_result_ttl_seconds"] > 0


def test_the_surface_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("ASYNC_JOBS_ENABLED", "false")
    assert jobs_mod.async_jobs_enabled() is False
    caps = jobs_mod.capabilities()
    assert caps["async_jobs"] is False
    assert caps["jobs_api_version"] is None
    assert caps["job_result_ttl_seconds"] is None


def test_job_ids_are_random_and_long():
    ids = {jobs_mod.new_job_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids)


# ----------------------------------------------------------------------
# Dedupe keys
# ----------------------------------------------------------------------
def test_the_same_bytes_and_parameters_share_a_key():
    params = jobs_mod.params_key({"model": "large-v3", "language": "en"})
    assert (jobs_mod.dedupe_key(params, content_hash="abc")
            == jobs_mod.dedupe_key(params, content_hash="abc"))


def test_other_parameters_give_another_key():
    a = jobs_mod.params_key({"model": "large-v3", "should_diarize": True})
    b = jobs_mod.params_key({"model": "large-v3", "should_diarize": False})
    assert a != b
    assert jobs_mod.dedupe_key(a, content_hash="abc") != jobs_mod.dedupe_key(b, content_hash="abc")


def test_parameter_order_does_not_matter():
    assert (jobs_mod.params_key({"a": 1, "b": 2})
            == jobs_mod.params_key({"b": 2, "a": 1}))


def test_an_idempotency_key_replaces_the_content_hash():
    params = jobs_mod.params_key({"model": "large-v3"})
    with_key = jobs_mod.dedupe_key(params, content_hash="abc", idempotency_key="rec-7")
    other_bytes = jobs_mod.dedupe_key(params, content_hash="zzz", idempotency_key="rec-7")
    assert with_key == other_bytes
    assert with_key != jobs_mod.dedupe_key(params, content_hash="abc")


def test_an_idempotency_key_with_other_parameters_is_a_new_job():
    a = jobs_mod.params_key({"model": "large-v3"})
    b = jobs_mod.params_key({"model": "turbo"})
    assert (jobs_mod.dedupe_key(a, idempotency_key="rec-7")
            != jobs_mod.dedupe_key(b, idempotency_key="rec-7"))


def test_hash_file_reads_in_chunks(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"x" * (3 * jobs_mod.HASH_CHUNK_BYTES + 7))
    import hashlib
    assert jobs_mod.hash_file(str(path)) == hashlib.sha256(b"x" * (3 * jobs_mod.HASH_CHUNK_BYTES + 7)).hexdigest()


# ----------------------------------------------------------------------
# State and persistence
# ----------------------------------------------------------------------
def test_a_new_job_is_queued_and_saved(store):
    job = make_job(store)
    assert job.status == jobs_mod.QUEUED
    assert os.path.isfile(job.state_path)
    state = json.loads(open(job.state_path).read())
    assert state["id"] == job.id and state["status"] == jobs_mod.QUEUED


def test_the_state_round_trips(store):
    job = make_job(store, params={"model": "turbo", "language": "de"})
    job.start()
    job.enter_stage("whisper")
    loaded = jobs_mod.Job.from_state(json.loads(open(job.state_path).read()), job.dir)
    assert loaded.id == job.id
    assert loaded.status == jobs_mod.RUNNING
    assert loaded.stage == "whisper"
    assert loaded.params == {"model": "turbo", "language": "de"}
    assert loaded.audio_seconds == job.audio_seconds


def test_finishing_stores_the_result(store):
    job = make_job(store)
    job.start()
    job.finish({"result": {"segments": [{"text": "hello"}]}, "speaker_embeddings": None})
    assert job.status == jobs_mod.DONE
    assert job.finished_at is not None
    assert job.load_result()["result"]["segments"][0]["text"] == "hello"


def test_failing_records_an_error_object(store):
    job = make_job(store)
    job.start()
    job.fail("decode_failed", "ffmpeg said no")
    assert job.status == jobs_mod.ERROR
    assert job.error == {"code": "decode_failed", "message": "ffmpeg said no",
                         "retryable": False}


def test_a_retryable_error_says_so(store):
    job = make_job(store)
    job.fail("out_of_memory", "CUDA out of memory")
    assert job.error["retryable"] is True


def test_cancelling_drops_the_result(store):
    job = make_job(store)
    job.start()
    job.finish({"result": {}, "speaker_embeddings": None})
    job.mark_cancelled()
    job.drop_result()
    assert job.status == jobs_mod.CANCELLED
    assert not os.path.exists(job.result_path)


# ----------------------------------------------------------------------
# Lookup and dedupe through the store
# ----------------------------------------------------------------------
def test_a_duplicate_finds_the_running_job(store):
    job = make_job(store, key="same")
    job.start()
    assert store.find_duplicate("same") is job


def test_a_duplicate_finds_the_finished_job(store):
    job = make_job(store, key="same")
    job.start()
    job.finish({"result": {}, "speaker_embeddings": None})
    assert store.find_duplicate("same") is job


def test_a_failed_job_is_not_served_as_a_duplicate(store):
    job = make_job(store, key="same")
    job.fail("stage_failed", "replica died")
    assert store.find_duplicate("same") is None


def test_a_cancelled_job_is_not_served_as_a_duplicate(store):
    job = make_job(store, key="same")
    job.mark_cancelled()
    assert store.find_duplicate("same") is None


def test_a_finished_job_without_its_file_is_not_a_duplicate(store):
    job = make_job(store, key="same")
    job.start()
    job.finish({"result": {}, "speaker_embeddings": None})
    os.unlink(job.result_path)
    assert store.find_duplicate("same") is None


def test_an_unknown_job_raises(store):
    with pytest.raises(jobs_mod.JobNotFound):
        store.get("nope")


def test_the_backlog_is_bounded(monkeypatch, store):
    monkeypatch.setenv("MAX_ASYNC_JOBS", "2")
    make_job(store, key="a")
    make_job(store, key="b")
    with pytest.raises(jobs_mod.TooManyJobs):
        make_job(store, key="c")


def test_a_finished_job_leaves_room(monkeypatch, store):
    monkeypatch.setenv("MAX_ASYNC_JOBS", "1")
    first = make_job(store, key="a")
    first.start()
    first.finish({"result": {}, "speaker_embeddings": None})
    make_job(store, key="b")  # no raise


def test_no_bound_when_zero(monkeypatch, store):
    monkeypatch.setenv("MAX_ASYNC_JOBS", "0")
    for i in range(40):
        make_job(store, key=f"k{i}")
    assert store.counts()["active"] == 40


# ----------------------------------------------------------------------
# Expiry
# ----------------------------------------------------------------------
def test_an_active_job_never_expires(store):
    job = make_job(store)
    job.start()
    assert job.expired(now=time.time() + 10_000_000) is False


def test_a_finished_job_expires_after_the_ttl(monkeypatch, store):
    monkeypatch.setenv("JOB_RESULT_TTL_SECONDS", "60")
    job = make_job(store)
    job.start()
    job.finish({"result": {}, "speaker_embeddings": None})
    assert job.expired(now=time.time() + 30) is False
    assert job.expired(now=time.time() + 120) is True


def test_sweeping_removes_the_directory(monkeypatch, store):
    monkeypatch.setenv("JOB_RESULT_TTL_SECONDS", "60")
    job = make_job(store)
    job.start()
    job.finish({"result": {}, "speaker_embeddings": None})
    assert store.forget_expired(now=time.time() + 120) == 1
    assert not os.path.exists(job.dir)
    with pytest.raises(jobs_mod.JobNotFound):
        store.get(job.id)


# ----------------------------------------------------------------------
# Recovery after a restart
# ----------------------------------------------------------------------
def test_a_job_in_flight_becomes_interrupted(tmp_path):
    first = jobs_mod.JobStore(root=str(tmp_path))
    job = make_job(first, key="a")
    job.start()
    job.enter_stage("whisper")

    second = jobs_mod.JobStore(root=str(tmp_path))
    stats = second.recover()
    assert stats == {"recovered": 1, "interrupted": 1}
    found = second.get(job.id)
    assert found.status == jobs_mod.INTERRUPTED
    assert found.error["code"] == "interrupted"
    assert found.error["retryable"] is True


def test_a_finished_job_stays_fetchable(tmp_path):
    first = jobs_mod.JobStore(root=str(tmp_path))
    job = make_job(first, key="a")
    job.start()
    job.finish({"result": {"segments": [{"text": "kept"}]}, "speaker_embeddings": None})

    second = jobs_mod.JobStore(root=str(tmp_path))
    assert second.recover() == {"recovered": 1, "interrupted": 0}
    found = second.get(job.id)
    assert found.status == jobs_mod.DONE
    assert found.load_result()["result"]["segments"][0]["text"] == "kept"
    assert second.find_duplicate("a") is found


def test_an_interrupted_job_is_not_a_duplicate(tmp_path):
    first = jobs_mod.JobStore(root=str(tmp_path))
    job = make_job(first, key="a")
    job.start()
    second = jobs_mod.JobStore(root=str(tmp_path))
    second.recover()
    assert second.find_duplicate("a") is None


def test_unreadable_state_is_ignored(tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / jobs_mod.STATE_NAME).write_text("{not json")
    store = jobs_mod.JobStore(root=str(tmp_path))
    assert store.recover() == {"recovered": 0, "interrupted": 0}


def test_directories_without_state_are_left_alone(tmp_path):
    directory = tmp_path / "plain-audio-job"
    directory.mkdir()
    (directory / "audio.s16").write_bytes(b"\x00\x00")
    store = jobs_mod.JobStore(root=str(tmp_path))
    assert store.recover() == {"recovered": 0, "interrupted": 0}
    assert directory.exists()


# ----------------------------------------------------------------------
# Progress
# ----------------------------------------------------------------------
def test_a_queued_job_reports_no_stage(store):
    job = make_job(store)
    p = job.progress(40.0)
    assert p["stage"] is None and p["fraction"] == 0.0
    assert p["fraction_estimated"] is True


def test_a_finished_job_is_complete_and_not_an_estimate(store):
    job = make_job(store)
    job.start()
    job.finish({"result": {}, "speaker_embeddings": None})
    p = job.progress(40.0)
    assert p["fraction"] == 1.0
    assert p["fraction_estimated"] is False
    assert p["eta_seconds"] == 0


def test_a_failed_job_has_no_fraction(store):
    job = make_job(store)
    job.start()
    job.fail("internal", "boom")
    p = job.progress(40.0)
    assert p["fraction"] is None
    assert p["fraction_estimated"] is False


def test_the_fraction_grows_with_the_stage(store):
    job = make_job(store, seconds=3600.0)
    job.start()
    job.enter_stage("whisper")
    whisper = job.progress(40.0)["fraction"]
    job.enter_stage("align")
    align = job.progress(40.0)["fraction"]
    job.enter_stage("diarize")
    diarize = job.progress(40.0)["fraction"]
    assert whisper < align < diarize < 1.0
    assert align == pytest.approx(0.60, abs=0.02)
    assert diarize == pytest.approx(0.85, abs=0.02)


def test_the_fraction_is_always_marked_as_an_estimate_while_running(store):
    job = make_job(store)
    job.start()
    job.enter_stage("whisper")
    assert job.progress(40.0)["fraction_estimated"] is True


def test_without_a_duration_there_is_no_fraction(store):
    job = store.create(params={}, key="k", filename="a.mp3")
    job.start()
    job.enter_stage("whisper")
    p = job.progress(40.0)
    assert p["fraction"] is None and p["eta_seconds"] is None
    assert p["stage"] == "whisper"


def test_the_rate_is_learned_from_finished_jobs(store):
    job = make_job(store, seconds=3600.0)
    job.start()
    job.started_at = time.time() - 60
    job.finish({"result": {}, "speaker_embeddings": None})
    assert store.rate > jobs_mod.INITIAL_REALTIME_FACTOR


def test_the_snapshot_carries_the_urls_and_the_error(store):
    job = make_job(store)
    job.fail("audio_too_long", "too long")
    snap = job.snapshot(40.0, deduplicated=True)
    assert snap["job_id"] == job.id
    assert snap["status_url"] == f"/jobs/{job.id}"
    assert snap["result_url"] == f"/jobs/{job.id}/result"
    assert snap["deduplicated"] is True
    assert snap["error"]["retryable"] is False
    assert snap["expires_at"] is not None


def test_poll_after_is_shorter_while_queued(store):
    job = make_job(store)
    queued = poll_after(job)
    job.start()
    job.enter_stage("whisper")
    assert queued <= poll_after(job)


# ----------------------------------------------------------------------
# Error classification
# ----------------------------------------------------------------------
def test_errors_map_to_codes():
    from app.admission import AudioTooLong
    assert classify_error(AudioTooLong(100, 50)) == "audio_too_long"
    assert classify_error(UploadTooLarge(10, 5)) == "file_too_large"
    assert classify_error(AudioDecodeError("bad")) == "decode_failed"
    assert classify_error(AdmissionRejected("queue_full", 30, {})) == "unavailable"
    assert classify_error(RuntimeError("something else")) == "internal"


def test_an_out_of_memory_message_is_recognised():
    assert classify_error(RuntimeError("CUDA out of memory")) == "out_of_memory"


def test_every_code_has_a_retryable_flag():
    for code in jobs_mod.ERROR_RETRYABLE:
        assert isinstance(jobs_mod.error_object(code, "m")["retryable"], bool)


# ----------------------------------------------------------------------
# Admission: the overrides asynchronous jobs rely on, and the old behaviour
# ----------------------------------------------------------------------
def test_the_queue_bounds_still_reject_a_synchronous_request():
    c = AdmissionController(max_inflight_audio_seconds=100, max_queued_requests=1)

    async def scenario():
        await c.acquire(100)           # fills the pipeline
        first = asyncio.create_task(c.acquire(10))
        await asyncio.sleep(0)
        with pytest.raises(AdmissionRejected):
            await c.acquire(10)        # queue of one is full
        first.cancel()
    run(scenario())


def test_a_job_is_not_rejected_by_the_queue_bounds():
    c = AdmissionController(max_inflight_audio_seconds=100, max_queued_requests=1)

    async def scenario():
        await c.acquire(100)
        waiting = asyncio.create_task(c.acquire(10))
        await asyncio.sleep(0)
        job = asyncio.create_task(c.acquire(10, enforce_queue_bounds=False))
        await asyncio.sleep(0.05)
        assert not job.done()          # queued, not rejected
        c.release(100)
        await asyncio.wait_for(asyncio.gather(waiting, job), timeout=2)
    run(scenario())


def test_a_job_can_wait_without_a_deadline():
    c = AdmissionController(max_inflight_audio_seconds=100,
                            queue_wait_timeout_seconds=0.05)

    async def scenario():
        await c.acquire(100)
        job = asyncio.create_task(
            c.acquire(10, enforce_queue_bounds=False, wait_timeout=None))
        await asyncio.sleep(0.2)       # well past the configured timeout
        assert not job.done()
        c.release(100)
        await asyncio.wait_for(job, timeout=2)
    run(scenario())


def test_the_configured_timeout_still_applies_by_default():
    c = AdmissionController(max_inflight_audio_seconds=100,
                            queue_wait_timeout_seconds=0.05)

    async def scenario():
        await c.acquire(100)
        with pytest.raises(AdmissionRejected):
            await c.acquire(10)
    run(scenario())


def test_the_slot_helper_passes_the_overrides_through():
    c = AdmissionController(max_inflight_audio_seconds=100,
                            queue_wait_timeout_seconds=0.05)

    async def scenario():
        async with c.slot(10, enforce_queue_bounds=False, wait_timeout=None):
            assert c.inflight_requests == 1
        assert c.inflight_requests == 0
    run(scenario())
