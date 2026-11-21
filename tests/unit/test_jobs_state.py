import json

import pytest

import app.jobs as jobs_mod


@pytest.fixture
def store(tmp_path):
    return jobs_mod.JobStore(root=str(tmp_path))


def persisted_status(job):
    with open(job.state_path) as handle:
        return json.load(handle)["status"]


def new_job(store):
    job = store.create(params={"model": "tiny"}, key="", filename="a.mp3")
    job.save()
    return job


def test_a_successful_job_goes_queued_running_done(store):
    job = new_job(store)
    assert job.status == "queued"
    assert persisted_status(job) == "queued"
    assert store.counts()["queued"] == 1 and store.counts()["running"] == 0

    job.start()
    assert job.status == "running"
    assert persisted_status(job) == "running"
    assert job.started_at is not None
    assert store.counts()["queued"] == 0 and store.counts()["running"] == 1

    job.finish({"result": {"segments": []}, "speaker_embeddings": None})
    assert job.status == "done"
    assert persisted_status(job) == "done"
    assert job.finished_at is not None
    assert store.counts()["active"] == 0


def test_a_failing_job_goes_queued_running_error(store):
    job = new_job(store)
    job.start()
    job.fail("decode_failed", "ffmpeg said no")
    assert job.status == "error"
    assert persisted_status(job) == "error"
    assert job.finished_at is not None
    assert job.error["code"] == "decode_failed"
    assert store.counts()["active"] == 0


def test_a_queued_job_can_be_cancelled_before_it_starts(store):
    job = new_job(store)
    job.mark_cancelled()
    assert job.status == "cancelled"
    assert persisted_status(job) == "cancelled"
    assert store.counts()["active"] == 0
