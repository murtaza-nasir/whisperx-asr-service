"""
Ray Serve deployment classes for the WhisperX pipeline.

Stages receive the path of a decoded job file (see app/audio_io.py) and load
the audio themselves, so the audio never passes through the Ray object store.
Arrays are still accepted. Blocking model work runs in a worker thread, one
item at a time per replica, so health checks and pings answer during long
calls.

Two strategies are available (selected via PIPELINE_STRATEGY env var):

  replicate (default) -- FullPipelineDeployment runs all 3 stages on one GPU.
      One replica per GPU, no cross-GPU data transfer.  Set NUM_GPU_REPLICAS
      to the number of GPUs.

  split -- Separate WhisperDeployment / AlignDeployment / DiarizeDeployment
      with fractional GPU allocation.  Useful when you want independent
      scaling per stage.
"""

import os
import asyncio
import logging
from typing import Optional, List, Tuple, Union

import torch
import numpy as np
from ray import serve

from app.pipeline import (
    run_pipeline as _run_pipeline,
    transcribe as _transcribe,
    align as _align,
    diarize as _diarize,
    load_whisper_model,
    load_align_model,
    load_diarize_pipeline,
    preload_whisper_model,
    DEFAULT_MODEL,
    HF_TOKEN,
    _env_int,
    _whisper_models,
)
from app.audio_io import as_audio_array

AudioInput = Union[str, np.ndarray]

def _attach_serve_handlers_to_app_logger():
    """
    Make `app.*` log records visible in Ray Serve's per-replica log file.

    Ray Serve configures only the `ray.serve` logger and disables propagation,
    so any log emitted by `app.pipeline`, `app.serve_deployments`, etc. inside
    a deployment worker has no handler attached and is silently dropped.
    Copying the active handlers from `ray.serve` onto the `app` logger fixes
    this without changing where Ray writes its own logs.

    Call this from each deployment's __init__ -- by then Ray Serve has already
    configured the `ray.serve` logger for this replica.
    """
    serve_logger = logging.getLogger("ray.serve")
    app_logger = logging.getLogger("app")
    if app_logger.handlers:
        return
    for h in serve_logger.handlers:
        app_logger.addHandler(h)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False


logger = logging.getLogger(__name__)

# Batch configuration from env
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "4"))
ALIGN_BATCH_SIZE = int(os.getenv("ALIGN_BATCH_SIZE", "8"))
DIARIZE_BATCH_SIZE = int(os.getenv("DIARIZE_BATCH_SIZE", "2"))
BATCH_WAIT_TIMEOUT = float(os.getenv("BATCH_WAIT_TIMEOUT", "0.1"))

# Replica counts -- per-stage overrides fall back to the shared default.
_DEFAULT_REPLICAS = os.getenv("NUM_GPU_REPLICAS") or "1"
WHISPER_NUM_REPLICAS = int(os.getenv("WHISPER_NUM_REPLICAS") or _DEFAULT_REPLICAS)
ALIGN_NUM_REPLICAS = int(os.getenv("ALIGN_NUM_REPLICAS") or _DEFAULT_REPLICAS)
DIARIZE_NUM_REPLICAS = int(os.getenv("DIARIZE_NUM_REPLICAS") or _DEFAULT_REPLICAS)

PIPELINE_STRATEGY = os.getenv("PIPELINE_STRATEGY", "replicate")

# Requests a stage replica accepts at once. Each replica runs one item at a
# time; the second slot keeps the next request ready. Requests beyond this
# wait in the ingress, which holds only the job file path.
STAGE_MAX_ONGOING_REQUESTS = max(1, _env_int("STAGE_MAX_ONGOING_REQUESTS", 2))
PIPELINE_MAX_ONGOING_REQUESTS = max(1, _env_int("PIPELINE_MAX_ONGOING_REQUESTS", 5))

# The full pipeline asks Ray for one GPU per replica. On a CPU-only node
# there is no GPU to schedule, so ask for none.
_FULL_PIPELINE_GPUS = 0.0 if os.getenv("DEVICE", "").strip().lower() == "cpu" else 1.0


class _StageWorker:
    """Runs blocking stage calls in a worker thread, one at a time, so the
    replica's event loop stays free for health checks and pings."""

    def _init_worker(self, name: str):
        self._name = name
        self._lock = asyncio.Lock()
        self._busy = 0

    async def _run_blocking(self, fn, *args, **kwargs):
        async with self._lock:
            self._busy += 1
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            finally:
                self._busy -= 1

    async def _run_items(self, fn, items):
        """Run fn(*item) for each item; an exception is returned in place of
        that item's result so one failure does not fail the whole batch."""
        out = []
        for item in items:
            try:
                out.append(await self._run_blocking(fn, *item))
            except Exception as e:  # returned, then raised for that caller only
                out.append(e)
        return out

    async def ping(self) -> dict:
        return {
            "stage": self._name,
            "ready": getattr(self, "_ready", True),
            "busy": self._busy > 0,
            "loaded_models": list(_whisper_models.keys()),
        }


def _raise_if_error(value):
    if isinstance(value, BaseException):
        raise value
    return value


def _transcribe_path(audio, model_name, language, task, prompt, hotwords):
    return _transcribe(as_audio_array(audio), model_name=model_name, language=language,
                       task=task, initial_prompt=prompt, hotwords=hotwords)


def _align_path(audio, result):
    return _align(as_audio_array(audio), result)


def _diarize_path(audio, result, num_spk, min_spk, max_spk, ret_emb):
    return _diarize(as_audio_array(audio), result, num_speakers=num_spk,
                    min_speakers=min_spk, max_speakers=max_spk,
                    return_speaker_embeddings=ret_emb)


def _run_pipeline_path(audio, **kwargs):
    return _run_pipeline(as_audio_array(audio), **kwargs)


# ======================================================================
# Replicate strategy -- full pipeline on each GPU
# ======================================================================

@serve.deployment(
    num_replicas=int(_DEFAULT_REPLICAS),
    ray_actor_options={"num_gpus": _FULL_PIPELINE_GPUS},
    # Model loading (Whisper + diarization) takes 30-120s per replica.
    # Give replicas enough time to initialise before Ray marks them unhealthy.
    health_check_period_s=30,
    health_check_timeout_s=300,
    # Allow the proxy to queue a few requests per replica so it never gets
    # "failed to route" errors, but keep it small -- each pipeline run is
    # 30-60s of sequential GPU work, so large queues just add tail latency.
    max_ongoing_requests=PIPELINE_MAX_ONGOING_REQUESTS,
)
class FullPipelineDeployment(_StageWorker):
    """Runs the complete 3-stage pipeline on a single GPU.

    No @serve.batch here -- _run_pipeline processes one audio file at a time
    through 3 sequential GPU stages.  Batching would just serialize multiple
    files inside one call, inflating tail latency (request #4 waits for #1-3).
    Instead we let Ray Serve route each request to whichever replica is free.
    """

    def __init__(self):
        _attach_serve_handlers_to_app_logger()
        self._init_worker("pipeline")
        self._ready = False

        # Log which GPU this replica landed on
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            logger.info(
                f"FullPipelineDeployment: initialising on cuda:{gpu_id} ({gpu_name}), "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
            )

        # Never fatal: a bad PRELOAD_MODEL is logged and the replica still
        # starts (it used to raise here and crash-loop the replica).
        preload_whisper_model("FullPipelineDeployment")
        if HF_TOKEN:
            logger.info("FullPipelineDeployment: preloading diarization pipeline")
            load_diarize_pipeline()

        self._ready = True
        logger.info("FullPipelineDeployment: initialisation complete, ready to serve")

    def check_health(self):
        if not self._ready:
            raise RuntimeError("Replica still initialising")

    async def run(
        self,
        audio: AudioInput,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        word_timestamps: bool = True,
        should_diarize: bool = True,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_speaker_embeddings: bool = False,
    ) -> Tuple[dict, Optional[dict]]:
        return await self._run_blocking(
            _run_pipeline_path,
            audio,
            model_name=model_name,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
            should_diarize=should_diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )


# ======================================================================
# Split strategy -- separate deployments per stage
# ======================================================================


@serve.deployment(
    num_replicas=WHISPER_NUM_REPLICAS,
    ray_actor_options={"num_gpus": float(os.getenv("WHISPER_GPU_FRACTION", "0.5"))},
    health_check_period_s=30,
    health_check_timeout_s=300,
    max_ongoing_requests=STAGE_MAX_ONGOING_REQUESTS,
)
class WhisperDeployment(_StageWorker):
    """Stage 1: Transcription via WhisperX."""

    def __init__(self):
        _attach_serve_handlers_to_app_logger()
        self._init_worker("whisper")
        self._ready = False
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            logger.info(
                f"WhisperDeployment: initialising on cuda:{gpu_id}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
            )
        preload_whisper_model("WhisperDeployment")
        self._ready = True

    def check_health(self):
        if not self._ready:
            raise RuntimeError("Replica still initialising")

    @serve.batch(max_batch_size=min(WHISPER_BATCH_SIZE, STAGE_MAX_ONGOING_REQUESTS),
                 batch_wait_timeout_s=BATCH_WAIT_TIMEOUT)
    async def transcribe_batch(
        self,
        audios: List[AudioInput],
        model_names: List[str],
        languages: List[Optional[str]],
        tasks: List[str],
        initial_prompts: List[Optional[str]],
        hotwords_list: List[Optional[str]],
    ) -> List[dict]:
        return await self._run_items(_transcribe_path, zip(
            audios, model_names, languages, tasks, initial_prompts, hotwords_list
        ))

    async def transcribe(
        self,
        audio: AudioInput,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> dict:
        return _raise_if_error(await self.transcribe_batch(
            audio, model_name, language, task, initial_prompt, hotwords
        ))


@serve.deployment(
    num_replicas=ALIGN_NUM_REPLICAS,
    ray_actor_options={"num_gpus": float(os.getenv("ALIGN_GPU_FRACTION", "0.3"))},
    health_check_period_s=30,
    health_check_timeout_s=300,
    max_ongoing_requests=STAGE_MAX_ONGOING_REQUESTS,
)
class AlignDeployment(_StageWorker):
    """Stage 2: Wav2Vec2 word-level alignment."""

    def __init__(self):
        _attach_serve_handlers_to_app_logger()
        self._init_worker("align")
        self._ready = True

    @serve.batch(max_batch_size=min(ALIGN_BATCH_SIZE, STAGE_MAX_ONGOING_REQUESTS),
                 batch_wait_timeout_s=BATCH_WAIT_TIMEOUT)
    async def align_batch(
        self,
        audios: List[AudioInput],
        results: List[dict],
    ) -> List[dict]:
        return await self._run_items(_align_path, zip(audios, results))

    async def align(self, audio: AudioInput, result: dict) -> dict:
        return _raise_if_error(await self.align_batch(audio, result))


@serve.deployment(
    num_replicas=DIARIZE_NUM_REPLICAS,
    ray_actor_options={"num_gpus": float(os.getenv("DIARIZE_GPU_FRACTION", "0.2"))},
    health_check_period_s=30,
    health_check_timeout_s=300,
    max_ongoing_requests=STAGE_MAX_ONGOING_REQUESTS,
)
class DiarizeDeployment(_StageWorker):
    """Stage 3: Pyannote speaker diarization."""

    def __init__(self):
        _attach_serve_handlers_to_app_logger()
        self._init_worker("diarize")
        self._ready = False
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            logger.info(
                f"DiarizeDeployment: initialising on cuda:{gpu_id}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
            )
        if HF_TOKEN:
            logger.info("DiarizeDeployment: preloading diarization pipeline")
            load_diarize_pipeline()
        self._ready = True

    def check_health(self):
        if not self._ready:
            raise RuntimeError("Replica still initialising")

    @serve.batch(
        max_batch_size=min(DIARIZE_BATCH_SIZE, STAGE_MAX_ONGOING_REQUESTS),
        batch_wait_timeout_s=float(os.getenv("DIARIZE_BATCH_WAIT_TIMEOUT", "0.2")),
    )
    async def diarize_batch(
        self,
        audios: List[AudioInput],
        results: List[dict],
        num_speakers_list: List[Optional[int]],
        min_speakers_list: List[Optional[int]],
        max_speakers_list: List[Optional[int]],
        return_embeddings_list: List[bool],
    ) -> List[Tuple[dict, Optional[dict]]]:
        return await self._run_items(_diarize_path, zip(
            audios, results, num_speakers_list, min_speakers_list,
            max_speakers_list, return_embeddings_list,
        ))

    async def diarize(
        self,
        audio: AudioInput,
        result: dict,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_speaker_embeddings: bool = False,
    ) -> Tuple[dict, Optional[dict]]:
        return _raise_if_error(await self.diarize_batch(
            audio, result, num_speakers, min_speakers,
            max_speakers, return_speaker_embeddings,
        ))
