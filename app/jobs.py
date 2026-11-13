"""
Asynchronous transcription jobs.

A submit returns a job id straight away; the transcription runs in a task
owned by the server, so it survives the client's timeout or disconnect.
Progress and the finished result are read back with separate requests.

Identical submits attach to the job that is already queued, running or
finished instead of starting a second one. Two submits are identical when
the transcription parameters match and either the uploaded bytes hash to the
same value or the caller sent the same `Idempotency-Key`.

State lives in the job's own directory next to the decoded audio:
`job.json` is rewritten on every status change and `result.json` holds the
raw pipeline result. A restart therefore answers "interrupted" for jobs
that were in flight rather than forgetting them, and the existing job sweep
removes both files with the directory.

The service has no inbound authentication, so a job id is the only gate on a
transcript. Ids are 128 random bits and must stay unguessable.

Every endpoint built on this module is additive: the synchronous endpoints
keep their behaviour, and `ASYNC_JOBS_ENABLED=false` removes the surface
entirely. No ML imports, so the CI tests exercise this module as it ships.
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import shutil
import time
from typing import Any, Dict, List, Optional

import app.audio_io as audio_io

logger = logging.getLogger(__name__)

# Version of the jobs API itself, advertised in /health so a client can
# detect later changes. Raise it when a field changes meaning.
JOBS_API_VERSION = 1

# Status values. Terminal ones are never left by a job.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"

ACTIVE_STATUSES = (QUEUED, RUNNING)
TERMINAL_STATUSES = (DONE, ERROR, CANCELLED, INTERRUPTED)

STATE_NAME = "job.json"
RESULT_NAME = "result.json"

# Share of the total run spent in each stage, used for the progress estimate.
# Split mode runs all three; the replicate strategy reports "pipeline".
STAGE_WEIGHTS = {"whisper": 0.60, "align": 0.25, "diarize": 0.15, "pipeline": 1.0}
STAGE_ORDER = ("whisper", "align", "diarize")

# Audio seconds processed per wall-clock second before anything is measured.
INITIAL_REALTIME_FACTOR = 40.0

HASH_CHUNK_BYTES = 1 << 20

# Error codes reported in the job's `error` object. The flag says whether the
# same submit could succeed later.
ERROR_RETRYABLE = {
    "audio_too_long": False,
    "file_too_large": False,
    "decode_failed": False,
    "invalid_request": False,
    "out_of_memory": True,
    "unavailable": True,
    "stage_failed": True,
    "interrupted": True,
    "internal": True,
}


def _env_or_none(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = _env_or_none(name)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        logger.warning(f"{name}={value!r} is not a number; using {default}")
        return default


def _env_flag(name: str, default: bool) -> bool:
    value = _env_or_none(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def async_jobs_enabled() -> bool:
    return _env_flag("ASYNC_JOBS_ENABLED", True)


def max_async_jobs() -> int:
    """Bound on jobs that are queued or running at once (0 = no bound)."""
    return max(0, _env_int("MAX_ASYNC_JOBS", 16))


def jobs_list_enabled() -> bool:
    """Whether GET /jobs may list job ids.

    Off by default. The service has no inbound authentication, so a job id is
    the only gate on its transcript, and a listing turns an unguessable id
    into an enumerable one. The counts in /health carry no ids and stay
    available either way.
    """
    return _env_flag("JOBS_LIST_ENABLED", False)


def result_ttl_seconds() -> int:
    """How long a finished job stays fetchable. Defaults to the job TTL so
    one setting still governs everything in JOB_DIR."""
    return max(60, _env_int("JOB_RESULT_TTL_SECONDS", audio_io.JOB_TTL_SECONDS))


def capabilities() -> dict:
    """Advertised in /health and / so a client can detect this surface."""
    enabled = async_jobs_enabled()
    return {
        "async_jobs": enabled,
        "jobs_api_version": JOBS_API_VERSION if enabled else None,
        "dedupe": enabled,
        "idempotency_key": enabled,
        "progress": enabled,
        "cancel": enabled,
        "list_jobs": enabled and jobs_list_enabled(),
        "job_result_ttl_seconds": result_ttl_seconds() if enabled else None,
        "max_async_jobs": max_async_jobs() if enabled else None,
    }


def error_object(code: str, message: str) -> dict:
    return {
        "code": code,
        "message": message,
        "retryable": ERROR_RETRYABLE.get(code, True),
    }


class JobNotFound(Exception):
    pass


class TooManyJobs(Exception):
    def __init__(self, limit: int):
        self.limit = limit
        super().__init__(f"Too many jobs in flight ({limit}). Retry later.")


def new_job_id() -> str:
    """128 random bits. A job id is the only gate on its transcript."""
    return secrets.token_hex(16)


def hash_file(path: str) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def params_key(params: Dict[str, Any]) -> str:
    """Stable hash of the parameters that change the transcription.

    The output format is deliberately absent: the raw result is stored and
    formatted when it is fetched, so one job serves every format.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dedupe_key(params_hash: str, content_hash: Optional[str] = None,
               idempotency_key: Optional[str] = None) -> str:
    """Key two submits share when they must become one job.

    An `Idempotency-Key` replaces the content hash: the caller then decides
    what counts as the same audio, and a key that stays the same while the
    audio changes attaches to the first job. The parameters are always part
    of the key, so a reprocess with other settings starts a new job.
    """
    if idempotency_key:
        material = f"idem:{idempotency_key}\nparams:{params_hash}"
    else:
        material = f"content:{content_hash or ''}\nparams:{params_hash}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Job:
    """One asynchronous transcription."""

    __slots__ = (
        "id", "status", "created_at", "started_at", "finished_at", "stage",
        "stage_started_at", "audio_seconds", "size_bytes", "filename", "error",
        "submissions", "dedupe_key", "params", "dir", "task", "_store",
    )

    def __init__(self, job_id: str, directory: str, *, params: Dict[str, Any],
                 dedupe_key: str, filename: Optional[str] = None,
                 store: Optional["JobStore"] = None):
        self.id = job_id
        self.dir = directory
        self.params = params
        self.dedupe_key = dedupe_key
        self.filename = filename
        self.status = QUEUED
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.stage: Optional[str] = None
        self.stage_started_at: Optional[float] = None
        self.audio_seconds: Optional[float] = None
        self.size_bytes: Optional[int] = None
        self.error: Optional[dict] = None
        self.submissions = 1
        self.task: Optional[asyncio.Task] = None
        self._store = store

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @property
    def state_path(self) -> str:
        return os.path.join(self.dir, STATE_NAME)

    @property
    def result_path(self) -> str:
        return os.path.join(self.dir, RESULT_NAME)

    def to_state(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stage": self.stage,
            "stage_started_at": self.stage_started_at,
            "audio_seconds": self.audio_seconds,
            "size_bytes": self.size_bytes,
            "filename": self.filename,
            "error": self.error,
            "submissions": self.submissions,
            "dedupe_key": self.dedupe_key,
            "params": self.params,
        }

    @classmethod
    def from_state(cls, state: dict, directory: str,
                   store: Optional["JobStore"] = None) -> "Job":
        job = cls(state["id"], directory, params=state.get("params") or {},
                  dedupe_key=state.get("dedupe_key") or "",
                  filename=state.get("filename"), store=store)
        job.status = state.get("status") or QUEUED
        job.created_at = state.get("created_at") or time.time()
        job.started_at = state.get("started_at")
        job.finished_at = state.get("finished_at")
        job.stage = state.get("stage")
        job.stage_started_at = state.get("stage_started_at")
        job.audio_seconds = state.get("audio_seconds")
        job.size_bytes = state.get("size_bytes")
        error = state.get("error")
        job.error = error if isinstance(error, dict) else None
        job.submissions = state.get("submissions") or 1
        return job

    def save(self) -> None:
        """Write job.json atomically. Never raises: a job whose state cannot
        be written still runs, it just will not be recovered after a restart."""
        try:
            os.makedirs(self.dir, exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(self.to_state(), handle)
            os.replace(tmp, self.state_path)
        except OSError as exc:
            logger.warning(f"Could not write state for job {self.id}: {exc}")

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.status = RUNNING
        self.started_at = time.time()
        self.save()

    def enter_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_started_at = time.time()
        self.save()

    def finish(self, result: dict) -> None:
        try:
            tmp = self.result_path + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(result, handle)
            os.replace(tmp, self.result_path)
        except (OSError, TypeError, ValueError) as exc:
            self.fail("internal", f"Could not store the result: {exc}")
            return
        self.status = DONE
        self.stage = None
        self.finished_at = time.time()
        self.save()
        if self._store is not None:
            self._store.observe(self)

    def fail(self, code: str, message: str) -> None:
        self.status = ERROR
        self.error = error_object(code, message)
        self.stage = None
        self.finished_at = time.time()
        self.save()

    def mark_cancelled(self) -> None:
        self.status = CANCELLED
        self.stage = None
        self.finished_at = time.time()
        self.save()

    def mark_interrupted(self) -> None:
        self.status = INTERRUPTED
        self.error = error_object(
            "interrupted", "The service restarted while this job was in flight."
        )
        self.stage = None
        self.finished_at = time.time()
        self.save()

    def load_result(self) -> dict:
        with open(self.result_path) as handle:
            return json.load(handle)

    def drop_result(self) -> None:
        """Remove a stored result (a cancel after the stages finished)."""
        try:
            os.unlink(self.result_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def expires_at(self) -> float:
        base = self.finished_at or self.created_at
        return base + result_ttl_seconds()

    def expired(self, now: Optional[float] = None) -> bool:
        if self.active:
            return False
        return (now if now is not None else time.time()) >= self.expires_at()

    def progress(self, rate: float) -> dict:
        """Stage and an estimated fraction.

        The stage name is exact. The fraction is built from the stage weights
        and the measured throughput, so `fraction_estimated` is true and the
        value must not be read as a measurement.
        """
        if self.status == DONE:
            return {"stage": None, "fraction": 1.0, "fraction_estimated": False,
                    "elapsed_seconds": round(self._elapsed(), 1), "eta_seconds": 0}
        if self.status in (ERROR, CANCELLED, INTERRUPTED):
            return {"stage": None, "fraction": None, "fraction_estimated": False,
                    "elapsed_seconds": round(self._elapsed(), 1), "eta_seconds": None}
        if self.status == QUEUED:
            return {"stage": None, "fraction": 0.0, "fraction_estimated": True,
                    "elapsed_seconds": 0.0, "eta_seconds": None}

        elapsed = self._elapsed()
        predicted = self._predicted_total(rate)
        stage = self.stage
        fraction = None
        if predicted:
            if stage in STAGE_WEIGHTS:
                done_weight = 0.0
                for name in STAGE_ORDER:
                    if name == stage:
                        break
                    done_weight += STAGE_WEIGHTS[name]
                weight = STAGE_WEIGHTS[stage]
                stage_predicted = predicted * weight
                stage_elapsed = max(0.0, time.time() - (self.stage_started_at or time.time()))
                within = min(1.0, stage_elapsed / stage_predicted) if stage_predicted else 0.0
                fraction = min(0.99, done_weight + weight * within)
            else:
                fraction = min(0.99, elapsed / predicted)
        return {
            "stage": stage,
            "fraction": round(fraction, 3) if fraction is not None else None,
            "fraction_estimated": True,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": int(max(0.0, predicted - elapsed)) if predicted else None,
        }

    def _elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)

    def _predicted_total(self, rate: float) -> Optional[float]:
        if not self.audio_seconds:
            return None
        return self.audio_seconds / max(rate, 0.01)

    def snapshot(self, rate: float = INITIAL_REALTIME_FACTOR,
                 deduplicated: bool = False) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "audio_seconds": (round(self.audio_seconds, 1)
                              if self.audio_seconds is not None else None),
            "filename": self.filename,
            "progress": self.progress(rate),
            "error": self.error,
            "deduplicated": deduplicated,
            "submissions": self.submissions,
            "expires_at": None if self.active else self.expires_at(),
            "status_url": f"/jobs/{self.id}",
            "result_url": f"/jobs/{self.id}/result",
        }


class JobStore:
    """Jobs of one server process, kept in step with their job directories."""

    def __init__(self, root: Optional[str] = None):
        self.root = root or audio_io.JOB_DIR
        self._jobs: Dict[str, Job] = {}
        self._by_key: Dict[str, str] = {}
        self._rate = INITIAL_REALTIME_FACTOR

    # ------------------------------------------------------------------
    @property
    def rate(self) -> float:
        return self._rate

    def observe(self, job: Job) -> None:
        """Feed a finished job into the throughput estimate."""
        if not job.audio_seconds or not job.started_at or not job.finished_at:
            return
        elapsed = job.finished_at - job.started_at
        if elapsed <= 0:
            return
        self._rate = 0.7 * self._rate + 0.3 * (job.audio_seconds / elapsed)

    def counts(self) -> dict:
        active = [j for j in self._jobs.values() if j.active]
        return {
            "total": len(self._jobs),
            "active": len(active),
            "queued": len([j for j in active if j.status == QUEUED]),
            "running": len([j for j in active if j.status == RUNNING]),
        }

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None or job.expired():
            raise JobNotFound(job_id)
        return job

    def find_duplicate(self, key: str) -> Optional[Job]:
        job_id = self._by_key.get(key)
        if job_id is None:
            return None
        job = self._jobs.get(job_id)
        if job is None or job.expired():
            self._by_key.pop(key, None)
            return None
        if job.status in (ERROR, CANCELLED, INTERRUPTED):
            # A failed or cancelled attempt must not be served as a cached answer.
            self._by_key.pop(key, None)
            return None
        if job.status == DONE and not os.path.exists(job.result_path):
            self._by_key.pop(key, None)
            return None
        return job

    def list(self) -> List[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def create(self, *, params: Dict[str, Any], key: str,
               filename: Optional[str] = None) -> Job:
        """Register a new job and its directory. Raises TooManyJobs."""
        limit = max_async_jobs()
        if limit and len([j for j in self._jobs.values() if j.active]) >= limit:
            raise TooManyJobs(limit)
        job_id = new_job_id()
        job = Job(job_id, os.path.join(self.root, job_id), params=params,
                  dedupe_key=key, filename=filename, store=self)
        self._jobs[job.id] = job
        if key:
            self._by_key[key] = job.id
        return job

    def forget(self, job_id: str) -> None:
        job = self._jobs.pop(job_id, None)
        if job is None:
            return
        if job.dedupe_key and self._by_key.get(job.dedupe_key) == job_id:
            self._by_key.pop(job.dedupe_key, None)

    def forget_expired(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        gone = [j.id for j in list(self._jobs.values()) if j.expired(now)]
        for job_id in gone:
            job = self._jobs.get(job_id)
            if job is not None:
                shutil.rmtree(job.dir, ignore_errors=True)
            self.forget(job_id)
        return len(gone)

    def recover(self) -> dict:
        """Read job dirs left by an earlier process.

        A job that was queued or running when the process stopped is reported
        as interrupted, so a client polling it is told what happened instead
        of receiving 404. Finished jobs stay fetchable until their TTL.
        """
        stats = {"recovered": 0, "interrupted": 0}
        try:
            entries = os.listdir(self.root)
        except OSError:
            return stats
        for name in entries:
            directory = os.path.join(self.root, name)
            state_path = os.path.join(directory, STATE_NAME)
            if not os.path.isfile(state_path):
                continue
            try:
                with open(state_path) as handle:
                    state = json.load(handle)
                job = Job.from_state(state, directory, store=self)
            except (OSError, ValueError, KeyError) as exc:
                logger.warning(f"Ignoring unreadable job state in {directory}: {exc}")
                continue
            if job.expired():
                shutil.rmtree(directory, ignore_errors=True)
                continue
            if job.active:
                job.mark_interrupted()
                stats["interrupted"] += 1
            elif job.status == DONE and not os.path.exists(job.result_path):
                shutil.rmtree(directory, ignore_errors=True)
                continue
            self._jobs[job.id] = job
            if job.dedupe_key and job.status == DONE:
                self._by_key[job.dedupe_key] = job.id
            stats["recovered"] += 1
        if stats["recovered"]:
            logger.info(f"Recovered {stats['recovered']} job(s) from {self.root}, "
                        f"{stats['interrupted']} marked interrupted")
        return stats


_store: Optional[JobStore] = None


def get_store() -> JobStore:
    """Process-wide store (one per ingress or simple-mode server)."""
    global _store
    if _store is None:
        _store = JobStore()
    return _store


def reset_store() -> None:
    global _store
    _store = None


__all__ = [
    "Job", "JobStore", "JobNotFound", "TooManyJobs",
    "QUEUED", "RUNNING", "DONE", "ERROR", "CANCELLED", "INTERRUPTED",
    "ACTIVE_STATUSES", "TERMINAL_STATUSES", "JOBS_API_VERSION",
    "async_jobs_enabled", "max_async_jobs", "jobs_list_enabled",
    "result_ttl_seconds", "capabilities",
    "error_object", "hash_file", "params_key", "dedupe_key", "new_job_id",
    "get_store", "reset_store",
]
