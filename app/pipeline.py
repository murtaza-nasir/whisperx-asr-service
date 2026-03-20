"""
Shared ASR pipeline stage functions.

Extracts the 3-stage WhisperX pipeline (transcribe -> align -> diarize) into
reusable functions consumed by both the legacy FastAPI endpoints and the
Ray Serve deployments.
"""

import os
import gc
import math
import logging
import threading
import warnings
from typing import Optional, Dict, Any, Tuple

# Suppress pyannote's torchcodec warning -- we decode audio via whisperx.load_audio (ffmpeg),
# not pyannote's built-in decoder, so the missing torchcodec is irrelevant.
warnings.filterwarnings("ignore", message=".*torchcodec.*")

import numpy as np
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at import time, same as before)
# ---------------------------------------------------------------------------
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("CACHE_DIR", "/.cache")
DEFAULT_MODEL = os.getenv("PRELOAD_MODEL", "large-v3")

_model_load_lock = threading.Lock()
_transcribe_lock = threading.Lock()
_diarize_lock = threading.Lock()

# Track concurrent GPU operations so clear_gpu_memory() only calls
# torch.cuda.empty_cache() when no other request is using the GPU.
_gpu_in_flight = 0
_gpu_in_flight_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model caches
# ---------------------------------------------------------------------------
_whisper_models: Dict[str, Any] = {}
_align_models: Dict[str, Tuple[Any, Any]] = {}
_diarize_pipeline: Optional[DiarizationPipeline] = None


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def gpu_in_flight_enter():
    """Increment the in-flight GPU request counter."""
    global _gpu_in_flight
    with _gpu_in_flight_lock:
        _gpu_in_flight += 1


def gpu_in_flight_exit():
    """Decrement the in-flight GPU request counter."""
    global _gpu_in_flight
    with _gpu_in_flight_lock:
        _gpu_in_flight -= 1


def clear_gpu_memory():
    """Clear GPU memory cache to prevent VRAM buildup.

    Always runs gc.collect(), but only calls torch.cuda.empty_cache() when no
    other requests are currently using the GPU — avoids disrupting in-flight
    CUDA operations from concurrent pipeline runs.
    """
    if DEVICE == "cuda":
        gc.collect()
        with _gpu_in_flight_lock:
            if _gpu_in_flight == 0:
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared")
            else:
                logger.debug("Skipping empty_cache, %d GPU ops in flight", _gpu_in_flight)


# ---------------------------------------------------------------------------
# Stage 0 -- model loading
# ---------------------------------------------------------------------------
def load_whisper_model(model_name: str):
    """Load WhisperX model with caching (thread-safe)."""
    if model_name not in _whisper_models:
        with _model_load_lock:
            if model_name not in _whisper_models:
                logger.info(f"Loading WhisperX model: {model_name}")
                model = whisperx.load_model(
                    model_name,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    download_root=CACHE_DIR,
                )
                _whisper_models[model_name] = model
                logger.info(f"Model {model_name} loaded successfully")
    return _whisper_models[model_name]


def load_align_model(language_code: str):
    """Load alignment model with per-language caching (thread-safe)."""
    if language_code not in _align_models:
        with _model_load_lock:
            if language_code not in _align_models:
                logger.info(f"Loading alignment model for language: {language_code}")
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=DEVICE,
                    model_dir=CACHE_DIR,
                )
                _align_models[language_code] = (model_a, metadata)
                logger.info(f"Alignment model for {language_code} loaded")
    return _align_models[language_code]


def load_diarize_pipeline() -> DiarizationPipeline:
    """Load diarization pipeline (singleton, thread-safe)."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        with _model_load_lock:
            if _diarize_pipeline is None:
                logger.info("Loading diarization pipeline: pyannote/speaker-diarization-3.1")
                _diarize_pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-3.1",
                    device=torch.device(DEVICE),
                )
                logger.info("Diarization pipeline loaded")
    return _diarize_pipeline


# ---------------------------------------------------------------------------
# Stage 1 -- Transcription
# ---------------------------------------------------------------------------
def transcribe(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
) -> dict:
    """Run WhisperX transcription and return raw result dict.

    Serialized via _transcribe_lock because the cached model singleton is not
    thread-safe: per-request options (hotwords, initial_prompt) are mutated on
    shared state, and upstream FasterWhisperPipeline.transcribe() mutates
    self.tokenizer and self.options during execution.
    """
    whisper_model = load_whisper_model(model_name)

    with _transcribe_lock:
        # Set per-request options on the model's transcription options.
        # The model is cached/shared, so we must reset after transcription.
        if hotwords is not None:
            whisper_model.options.hotwords = hotwords
        if initial_prompt is not None:
            whisper_model.options.initial_prompt = initial_prompt

        transcribe_options: Dict[str, Any] = {
            "batch_size": BATCH_SIZE,
            "language": language,
            "task": task,
        }

        logger.info("Starting transcription...")
        try:
            result = whisper_model.transcribe(audio, **transcribe_options)
        finally:
            if hotwords is not None:
                whisper_model.options.hotwords = None
            if initial_prompt is not None:
                whisper_model.options.initial_prompt = None

    detected_language = result.get("language", language or "en")
    logger.info(f"Transcription complete. Detected language: {detected_language}")

    clear_gpu_memory()
    return result


# ---------------------------------------------------------------------------
# Stage 2 -- Alignment
# ---------------------------------------------------------------------------
def align(audio: np.ndarray, result: dict) -> dict:
    """Run Wav2Vec2 alignment to get word-level timestamps."""
    detected_language = result.get("language", "en")
    logger.info("Aligning timestamps...")
    try:
        model_a, metadata = load_align_model(detected_language)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )
        logger.info("Timestamp alignment complete")
        clear_gpu_memory()
    except Exception as e:
        logger.warning(f"Timestamp alignment failed: {e}, continuing without word-level timestamps")
    return result


# ---------------------------------------------------------------------------
# Stage 3 -- Diarization
# ---------------------------------------------------------------------------
def diarize(
    audio: np.ndarray,
    result: dict,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_speaker_embeddings: bool = False,
) -> Tuple[dict, Optional[dict]]:
    """
    Run pyannote speaker diarization and assign speakers to segments.

    Returns (result_with_speakers, speaker_embeddings_or_None).

    Serialized via _diarize_lock as a precaution — pyannote internals are
    not verified thread-safe.
    """
    if not HF_TOKEN:
        logger.warning("Speaker diarization requested but HF_TOKEN not set")
        return result, None

    with _diarize_lock:
        logger.info("Starting speaker diarization...")
        speaker_embeddings = None
        try:
            diarize_model = load_diarize_pipeline()

            diarize_params: Dict[str, Any] = {}
            if num_speakers is not None:
                diarize_params["num_speakers"] = num_speakers
                logger.info(f"Diarization with exact speaker count: {num_speakers}")
            else:
                if min_speakers is not None:
                    diarize_params["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarize_params["max_speakers"] = max_speakers
                logger.info(f"Diarization with speaker range: {min_speakers}-{max_speakers}")

            if return_speaker_embeddings:
                diarize_params["return_embeddings"] = True
                logger.info("Speaker embeddings will be returned")

            diarize_output = diarize_model(audio, **diarize_params)

            if return_speaker_embeddings and isinstance(diarize_output, tuple):
                diarize_segments, speaker_embeddings = diarize_output
                logger.info(f"Received speaker embeddings for {len(speaker_embeddings)} speakers")
            else:
                diarize_segments = diarize_output

            if hasattr(diarize_segments, "exclusive_speaker_diarization"):
                diarize_segments = diarize_segments.exclusive_speaker_diarization
                logger.info("Using exclusive speaker diarization for better timestamp reconciliation")

            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info("Speaker diarization complete")
            clear_gpu_memory()
        except Exception as e:
            logger.warning(f"Speaker diarization failed: {e}, continuing without diarization")

    return result, speaker_embeddings


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------
def sanitize_float_values(obj):
    """Recursively sanitize float values for JSON compliance (NaN/Inf -> None)."""
    if isinstance(obj, dict):
        return {key: sanitize_float_values(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_float_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_float_values(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return obj


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------
def run_pipeline(
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
    """
    Run the full 3-stage pipeline: transcribe -> align -> diarize.

    Returns (result, speaker_embeddings_or_None).

    Tracks in-flight GPU operations so clear_gpu_memory() only calls
    torch.cuda.empty_cache() when no other request is using the GPU.
    """
    gpu_in_flight_enter()
    try:
        result = transcribe(
            audio,
            model_name=model_name,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )

        if word_timestamps:
            result = align(audio, result)

        speaker_embeddings = None
        if should_diarize:
            result, speaker_embeddings = diarize(
                audio,
                result,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_speaker_embeddings=return_speaker_embeddings,
            )

        return result, speaker_embeddings
    finally:
        gpu_in_flight_exit()
