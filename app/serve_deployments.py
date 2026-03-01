"""
Ray Serve deployment classes for the WhisperX pipeline.

Two strategies are available (selected via PIPELINE_STRATEGY env var):

  replicate (default) -- FullPipelineDeployment runs all 3 stages on one GPU.
      One replica per GPU, no cross-GPU data transfer.  Set NUM_GPU_REPLICAS
      to the number of GPUs.

  split -- Separate WhisperDeployment / AlignDeployment / DiarizeDeployment
      with fractional GPU allocation.  Useful when you want independent
      scaling per stage.
"""

import os
import logging
from typing import Optional, List, Tuple

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
    DEFAULT_MODEL,
    HF_TOKEN,
)

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


# ======================================================================
# Replicate strategy -- full pipeline on each GPU
# ======================================================================

@serve.deployment(
    num_replicas=int(_DEFAULT_REPLICAS),
    ray_actor_options={"num_gpus": 1.0},
    # Model loading (Whisper + diarization) takes 30-120s per replica.
    # Give replicas enough time to initialise before Ray marks them unhealthy.
    health_check_period_s=30,
    health_check_timeout_s=300,
    # Allow the proxy to queue a few requests per replica so it never gets
    # "failed to route" errors, but keep it small -- each pipeline run is
    # 30-60s of sequential GPU work, so large queues just add tail latency.
    max_ongoing_requests=5,
)
class FullPipelineDeployment:
    """Runs the complete 3-stage pipeline on a single GPU.

    No @serve.batch here -- _run_pipeline processes one audio file at a time
    through 3 sequential GPU stages.  Batching would just serialize multiple
    files inside one call, inflating tail latency (request #4 waits for #1-3).
    Instead we let Ray Serve route each request to whichever replica is free.
    """

    def __init__(self):
        self._ready = False

        # Log which GPU this replica landed on
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            logger.info(
                f"FullPipelineDeployment: initialising on cuda:{gpu_id} ({gpu_name}), "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
            )

        preload = os.getenv("PRELOAD_MODEL", None)
        if preload:
            logger.info(f"FullPipelineDeployment: preloading model {preload}")
            load_whisper_model(preload)
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
        audio: np.ndarray,
        model_name: str = DEFAULT_MODEL,
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
        return _run_pipeline(
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
)
class WhisperDeployment:
    """Stage 1: Transcription via WhisperX."""

    def __init__(self):
        self._ready = False
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            logger.info(
                f"WhisperDeployment: initialising on cuda:{gpu_id}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
            )
        preload = os.getenv("PRELOAD_MODEL", None)
        if preload:
            logger.info(f"WhisperDeployment: preloading model {preload}")
            load_whisper_model(preload)
        self._ready = True

    def check_health(self):
        if not self._ready:
            raise RuntimeError("Replica still initialising")

    @serve.batch(max_batch_size=WHISPER_BATCH_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT)
    async def transcribe_batch(
        self,
        audios: List[np.ndarray],
        model_names: List[str],
        languages: List[Optional[str]],
        tasks: List[str],
        initial_prompts: List[Optional[str]],
        hotwords_list: List[Optional[str]],
    ) -> List[dict]:
        results = []
        for audio, model_name, language, task, prompt, hotwords in zip(
            audios, model_names, languages, tasks, initial_prompts, hotwords_list
        ):
            result = _transcribe(
                audio,
                model_name=model_name,
                language=language,
                task=task,
                initial_prompt=prompt,
                hotwords=hotwords,
            )
            results.append(result)
        return results

    async def transcribe(
        self,
        audio: np.ndarray,
        model_name: str = DEFAULT_MODEL,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> dict:
        return await self.transcribe_batch(
            audio, model_name, language, task, initial_prompt, hotwords
        )


@serve.deployment(
    num_replicas=ALIGN_NUM_REPLICAS,
    ray_actor_options={"num_gpus": float(os.getenv("ALIGN_GPU_FRACTION", "0.3"))},
    health_check_period_s=30,
    health_check_timeout_s=300,
)
class AlignDeployment:
    """Stage 2: Wav2Vec2 word-level alignment."""

    @serve.batch(max_batch_size=ALIGN_BATCH_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT)
    async def align_batch(
        self,
        audios: List[np.ndarray],
        results: List[dict],
    ) -> List[dict]:
        aligned = []
        for audio, result in zip(audios, results):
            aligned.append(_align(audio, result))
        return aligned

    async def align(self, audio: np.ndarray, result: dict) -> dict:
        return await self.align_batch(audio, result)


@serve.deployment(
    num_replicas=DIARIZE_NUM_REPLICAS,
    ray_actor_options={"num_gpus": float(os.getenv("DIARIZE_GPU_FRACTION", "0.2"))},
    health_check_period_s=30,
    health_check_timeout_s=300,
)
class DiarizeDeployment:
    """Stage 3: Pyannote speaker diarization."""

    def __init__(self):
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
        max_batch_size=DIARIZE_BATCH_SIZE,
        batch_wait_timeout_s=float(os.getenv("DIARIZE_BATCH_WAIT_TIMEOUT", "0.2")),
    )
    async def diarize_batch(
        self,
        audios: List[np.ndarray],
        results: List[dict],
        num_speakers_list: List[Optional[int]],
        min_speakers_list: List[Optional[int]],
        max_speakers_list: List[Optional[int]],
        return_embeddings_list: List[bool],
    ) -> List[Tuple[dict, Optional[dict]]]:
        outputs = []
        for audio, result, num_spk, min_spk, max_spk, ret_emb in zip(
            audios, results, num_speakers_list, min_speakers_list,
            max_speakers_list, return_embeddings_list,
        ):
            out = _diarize(
                audio, result,
                num_speakers=num_spk,
                min_speakers=min_spk,
                max_speakers=max_spk,
                return_speaker_embeddings=ret_emb,
            )
            outputs.append(out)
        return outputs

    async def diarize(
        self,
        audio: np.ndarray,
        result: dict,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_speaker_embeddings: bool = False,
    ) -> Tuple[dict, Optional[dict]]:
        return await self.diarize_batch(
            audio, result, num_speakers, min_speakers,
            max_speakers, return_speaker_embeddings,
        )
