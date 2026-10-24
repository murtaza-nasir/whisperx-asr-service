"""
Shared ASR pipeline stage functions.

Extracts the 3-stage WhisperX pipeline (transcribe -> align -> diarize) into
reusable functions consumed by both the legacy FastAPI endpoints and the
Ray Serve deployments.
"""

import os
import gc
import copy
import json
import math
import time
import logging
import threading
import warnings
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple, List

# Suppress pyannote's torchcodec warning -- we decode audio via whisperx.load_audio (ffmpeg),
# not pyannote's built-in decoder, so the missing torchcodec is irrelevant.
warnings.filterwarnings("ignore", message=".*torchcodec.*")

import numpy as np
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from app.alignment_preprocessing import prepare_alignment_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at import time, same as before)
# ---------------------------------------------------------------------------
def _env_or_none(name: str) -> Optional[str]:
    """Read an env var, treating unset OR empty/whitespace as None.

    Compose forwards optional vars as `${VAR:-}`, which sets them to an empty
    string rather than leaving them unset, so a plain os.getenv() would return
    "" and downstream float() would crash. Normalize that to None here.
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _env_int(name: str, default: int) -> int:
    """Integer env var; unset, empty or unparseable values give the default."""
    value = _env_or_none(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"{name}={value!r} is not an integer; using {default}")
        return default


DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
BATCH_SIZE = _env_int("BATCH_SIZE", 16 if DEVICE == "cuda" else 2)
# Device for the Wav2Vec2 alignment stage. Defaults to DEVICE; set
# ALIGN_DEVICE=cpu to keep alignment off the GPU and reduce VRAM at the cost
# of slower word timestamps (issue #32).
ALIGN_DEVICE = os.getenv("ALIGN_DEVICE", "").strip().lower() or DEVICE
HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("CACHE_DIR", "/.cache")


# Model selection.
#   PRELOAD_MODEL: Whisper model loaded at startup. Unset or empty means no
#       preload.
#   DEFAULT_MODEL: model used when a request names none. Falls back to
#       PRELOAD_MODEL, then to large-v3, so an empty PRELOAD_MODEL disables
#       preloading without leaving requests without a model (issue: Speakr
#       #409). Both are read through _env_or_none because compose forwards
#       unset variables as empty strings.
#   ALLOWED_MODELS: optional comma-separated list; when set, requests may only
#       name these models (the default model is always allowed).
#   MAX_LOADED_MODELS: optional cap on Whisper models kept in memory at once;
#       the least recently used idle model is unloaded to make room. 0 = no cap.
BUILTIN_DEFAULT_MODEL = "large-v3"
PRELOAD_MODEL = _env_or_none("PRELOAD_MODEL")
_CONFIGURED_DEFAULT_MODEL = _env_or_none("DEFAULT_MODEL") or PRELOAD_MODEL or BUILTIN_DEFAULT_MODEL
DEFAULT_MODEL = _CONFIGURED_DEFAULT_MODEL  # resolved to a canonical name below
MAX_LOADED_MODELS = max(0, _env_int("MAX_LOADED_MODELS", 0))

# Idle model eviction. Set MODEL_KEEP_ALIVE_SECONDS > 0 to unload Whisper,
# alignment and diarization models that have not been used in that many
# seconds. Floor of 30s on the sweep interval to avoid pegging a thread on
# tight loops.
MODEL_KEEP_ALIVE_SECONDS = _env_int("MODEL_KEEP_ALIVE_SECONDS", 0)
MODEL_EVICTION_INTERVAL_SECONDS = max(
    30, _env_int("MODEL_EVICTION_INTERVAL_SECONDS", 60)
)

# Diarization hyperparameter tuning (pyannote community-1).
# All unset by default -> the pipeline runs with the model's published defaults,
# so behaviour is unchanged unless you opt in.
#
#   DIARIZE_CLUSTERING_THRESHOLD: the main lever for merged/missed speakers.
#       community-1 default is 0.6. Lower it (e.g. 0.5) to split voices more
#       aggressively when distinct speakers share one label; raise it to merge
#       more (fewer phantom speakers). Useful range ~0.4-0.8.
#   DIARIZE_MIN_DURATION_OFF: non-speech gaps shorter than this (seconds) are
#       filled, MERGING the turns on either side. community-1 default is 0.0.
#       RAISE it (e.g. 0.1-0.5) to suppress over-segmentation; lowering below
#       0.0 is not possible, so it does not help recover rapid turns -- use the
#       clustering threshold for that.
#   DIARIZE_PARAM_OVERRIDES: escape hatch -- a JSON object deep-merged into the
#       pipeline's instantiated parameters, for any key the two vars above don't
#       cover. The exact schema is logged at pipeline load (see logs).
DIARIZE_CLUSTERING_THRESHOLD = _env_or_none("DIARIZE_CLUSTERING_THRESHOLD")
DIARIZE_MIN_DURATION_OFF = _env_or_none("DIARIZE_MIN_DURATION_OFF")
DIARIZE_PARAM_OVERRIDES = _env_or_none("DIARIZE_PARAM_OVERRIDES")

# Voice activity detection (VAD) for the whisper backend. Whisper only sees the
# stretches the VAD marks as speech, so these settings decide whether silence
# and background noise reach the model, where Whisper tends to invent text
# ("Thank you.") in it. All unset by default -> WhisperX's own defaults, and
# the model is loaded exactly as before.
#
#   VAD_METHOD: "pyannote" (WhisperX default) or "silero".
#   VAD_ONSET: speech probability above which a stretch counts as speech.
#       WhisperX default 0.5. In tests with a noise floor raised by a
#       recorder's gain control, the invented text was removed at 0.8 but
#       not at 0.6-0.7; at 0.8 a few percent of quiet real speech was lost.
#   VAD_OFFSET: probability below which speech is considered to have ended.
#       WhisperX default 0.363.
# The chunk length is deliberately not configurable: with chunks shorter
# than WhisperX's 30 s, far more invented text appeared in the same tests.
# The batched WhisperX pipeline does not apply Whisper's no_speech,
# log_prob or compression-ratio thresholds, so the VAD is the setting that
# acts on noise. The settings are read at model load; restart to change them.
def _env_float(name: str, low: float, high: float) -> Optional[float]:
    """Float env var strictly between low and high; otherwise None, with a warning."""
    value = _env_or_none(name)
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        logger.warning(f"{name}={value!r} is not a number; using the WhisperX default")
        return None
    if not low < number < high:
        logger.warning(f"{name}={value!r} is outside {low}-{high}; using the WhisperX default")
        return None
    return number


VAD_METHOD = (_env_or_none("VAD_METHOD") or "").lower() or None
if VAD_METHOD not in (None, "pyannote", "silero"):
    logger.warning(f"VAD_METHOD={VAD_METHOD!r} is not pyannote or silero; using the WhisperX default")
    VAD_METHOD = None
VAD_ONSET = _env_float("VAD_ONSET", 0.0, 1.0)
VAD_OFFSET = _env_float("VAD_OFFSET", 0.0, 1.0)


def vad_load_kwargs() -> Dict[str, Any]:
    """Arguments for whisperx.load_model; empty when nothing is set."""
    options = {key: value for key, value in (
        ("vad_onset", VAD_ONSET), ("vad_offset", VAD_OFFSET),
    ) if value is not None}
    kwargs: Dict[str, Any] = {}
    if VAD_METHOD:
        kwargs["vad_method"] = VAD_METHOD
    if options:
        kwargs["vad_options"] = options
    return kwargs


# When True, words/segments that fall outside every diarization turn are assigned
# the *nearest* speaker instead of being left unlabeled. Fixes "orphan" segments
# (e.g. a closing line with no speaker tag) at the cost of occasionally labeling
# a long silence. Default False preserves the prior behaviour.
DIARIZE_FILL_NEAREST = os.getenv("DIARIZE_FILL_NEAREST", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

# ASR backend selection. "whisper" (default) is the faster-whisper/CTranslate2
# path. "qwen3" transcribes with Qwen3-ASR and aligns with the Qwen3 forced
# aligner (language-agnostic, useful for code-switched audio); it ignores the
# requested Whisper model name and does not support task=translate.
ASR_BACKEND = (_env_or_none("ASR_BACKEND") or "whisper").lower()

# When True, segments are rebuilt at speaker-change boundaries after
# diarization, so rapid turns are not merged into one speaker's segment.
# Always applied on the qwen3 backend (its pre-diarization segments come from
# silence gaps alone); opt-in for the whisper backend because it changes the
# segment shape existing users are accustomed to.
RESEGMENT_BY_SPEAKER = (
    os.getenv("RESEGMENT_BY_SPEAKER", "false").strip().lower()
    in ("1", "true", "yes", "on")
)


def get_canonical_models() -> list:
    """
    Canonical model names accepted by the underlying faster-whisper engine.

    Sourced from faster_whisper.available_models() so this list stays in sync
    with whatever version of faster-whisper is installed, instead of being
    hardcoded here.
    """
    try:
        from faster_whisper import available_models
        return list(available_models())
    except Exception:
        # Defensive fallback if the import surface ever changes upstream.
        return [
            "tiny.en", "tiny", "base.en", "base", "small.en", "small",
            "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large",
            "distil-large-v2", "distil-medium.en", "distil-small.en",
            "distil-large-v3", "distil-large-v3.5", "large-v3-turbo", "turbo",
        ]


class InvalidModelError(ValueError):
    """A requested model name the service cannot or may not load (HTTP 400)."""


# OpenAI-style aliases → canonical faster-whisper names. These are kept for
# backwards compatibility on the request path; new clients should use the
# canonical names returned by /v1/models. `whisper-1` is filled in below,
# once the default model is known.
_STATIC_ALIASES = {
    "whisper-large-v3": "large-v3",
    "whisper-large-v2": "large-v2",
    "whisper-medium": "medium",
    "whisper-small": "small",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}
_HF_REPO_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _is_hf_repo_id(name: str) -> bool:
    """`org/repo` Hugging Face ids, which faster-whisper downloads."""
    parts = name.split("/")
    return (
        len(parts) == 2
        and all(parts)
        and all(set(part) <= _HF_REPO_CHARS for part in parts)
    )


def _map_alias(name: str) -> str:
    """Apply aliases and the `whisper-<canonical>` prefix; no validation."""
    if name == "whisper-1":
        return _MODEL_ALIASES["whisper-1"]
    if name in _STATIC_ALIASES:
        return _STATIC_ALIASES[name]
    if name.startswith("whisper-"):
        stripped = name[len("whisper-"):]
        if stripped in set(get_canonical_models()):
            return stripped
    return name


def is_loadable_model(name: str) -> bool:
    """True for names faster-whisper can load: canonical sizes, `org/repo`
    Hugging Face ids and existing local model directories."""
    if not name:
        return False
    return (
        name in set(get_canonical_models())
        or _is_hf_repo_id(name)
        or os.path.isdir(name)
    )


def _resolve_default(raw: str) -> str:
    """Canonical form of a configured default; an unusable value falls back
    to the built-in default with a warning, so a typo cannot fail every
    request that names no model."""
    name = raw.strip()
    if name == "whisper-1":
        name = _OPENAI_WHISPER1_MODEL or BUILTIN_DEFAULT_MODEL
    name = _STATIC_ALIASES.get(name, name)
    if name.startswith("whisper-") and name[len("whisper-"):] in set(get_canonical_models()):
        name = name[len("whisper-"):]
    if not is_loadable_model(name):
        logger.warning(
            f"Default model {raw!r} is not a model this service can load; "
            f"using {BUILTIN_DEFAULT_MODEL}. Accepted: "
            f"{', '.join(get_canonical_models())}, a Hugging Face id (org/repo) "
            "or a local model directory."
        )
        return BUILTIN_DEFAULT_MODEL
    return name


_OPENAI_WHISPER1_MODEL = _env_or_none("OPENAI_WHISPER1_MODEL")
DEFAULT_MODEL = _resolve_default(_CONFIGURED_DEFAULT_MODEL)
_MODEL_ALIASES = dict(_STATIC_ALIASES)
# whisper-1 maps to OPENAI_WHISPER1_MODEL, else the default. A value that is
# itself "whisper-1" would point the alias at itself, so it is resolved as a
# default (which never returns "whisper-1").
_MODEL_ALIASES["whisper-1"] = (
    _resolve_default(_OPENAI_WHISPER1_MODEL) if _OPENAI_WHISPER1_MODEL else DEFAULT_MODEL
)

ALLOWED_MODELS: List[str] = []
for _entry in (_env_or_none("ALLOWED_MODELS") or "").split(","):
    _entry = _entry.strip()
    if _entry:
        _mapped = _map_alias(_entry)
        if _mapped not in ALLOWED_MODELS:
            ALLOWED_MODELS.append(_mapped)


def resolve_model_name(model: Optional[str]) -> str:
    """
    Resolve a user-supplied model identifier to the name the engine loads.

    Empty or whitespace falls back to DEFAULT_MODEL. Canonical names
    (tiny, large-v3, distil-medium.en, ...) pass through; OpenAI-style aliases
    (whisper-1, whisper-large-v3, ...) map to canonical names. Hugging Face
    ids (org/repo) and local model directories are accepted. Anything else,
    or a model outside ALLOWED_MODELS, raises InvalidModelError, which the
    endpoints return as HTTP 400.
    """
    name = (model or "").strip()
    if not name:
        return DEFAULT_MODEL
    resolved = _map_alias(name)
    if not is_loadable_model(resolved):
        raise InvalidModelError(
            f"Unknown model {name!r}. Accepted: {', '.join(list_available_models())}, "
            "a Hugging Face id (org/repo) or a local model directory."
        )
    if ALLOWED_MODELS and resolved not in ALLOWED_MODELS and resolved != DEFAULT_MODEL:
        raise InvalidModelError(
            f"Model {name!r} is not allowed on this server. Allowed: "
            f"{', '.join(list_available_models())}."
        )
    return resolved


def list_available_models() -> List[str]:
    """Canonical model names clients may request (ALLOWED_MODELS applied)."""
    if ALLOWED_MODELS:
        names = list(ALLOWED_MODELS)
        if DEFAULT_MODEL not in names:
            names.insert(0, DEFAULT_MODEL)
        return names
    return get_canonical_models()


def describe_model_settings() -> str:
    """One line for the startup log."""
    parts = [
        f"default model: {DEFAULT_MODEL}",
        f"preload: {PRELOAD_MODEL or 'none'}",
    ]
    if ALLOWED_MODELS:
        parts.append(f"allowed models: {', '.join(ALLOWED_MODELS)}")
    if MAX_LOADED_MODELS:
        parts.append(f"max loaded models: {MAX_LOADED_MODELS}")
    return ", ".join(parts)


if _CONFIGURED_DEFAULT_MODEL != DEFAULT_MODEL:
    logger.info(f"Default model {_CONFIGURED_DEFAULT_MODEL!r} resolves to {DEFAULT_MODEL!r}")
for _entry in ALLOWED_MODELS:
    if not is_loadable_model(_entry):
        logger.warning(f"ALLOWED_MODELS entry {_entry!r} is not a model this service can load")


_model_load_lock = threading.Lock()
# Short-held lock for the Whisper model cache, its last-used times and in-use
# counts. Loading a model takes _model_load_lock (which serialises the slow
# loads); requests for a model that is already loaded only take this one, so
# they never wait behind another model's load.
_whisper_state_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model caches
# ---------------------------------------------------------------------------
_whisper_models: Dict[str, Any] = {}
_whisper_models_last_used: Dict[str, float] = {}
# Requests currently using each Whisper model; such a model is never unloaded
# by the MAX_LOADED_MODELS cap or the idle sweep.
_whisper_models_in_use: Dict[str, int] = {}
_align_models: Dict[str, Tuple[Any, Any]] = {}
_align_models_last_used: Dict[str, float] = {}
_diarize_pipeline: Optional[DiarizationPipeline] = None
_diarize_last_used: Optional[float] = None

_eviction_thread_lock = threading.Lock()
_eviction_thread_started = False


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def clear_gpu_memory():
    """Clear GPU memory cache to prevent VRAM buildup."""
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")


# ---------------------------------------------------------------------------
# Stage 0 -- model loading
# ---------------------------------------------------------------------------
def _count_eviction(model_name: str) -> None:
    try:
        from app import metrics as prom_metrics
        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=model_name).inc()
    except Exception:
        pass


def _make_room_for(model_name: str) -> bool:
    """With MAX_LOADED_MODELS set, unload least recently used idle Whisper
    models until one more fits. Caller holds _whisper_state_lock. Returns True
    if anything was unloaded."""
    if MAX_LOADED_MODELS <= 0:
        return False
    evicted = False
    while len(_whisper_models) >= MAX_LOADED_MODELS:
        idle = [
            name for name in _whisper_models
            if name != model_name and not _whisper_models_in_use.get(name)
        ]
        if not idle:
            logger.warning(
                f"MAX_LOADED_MODELS={MAX_LOADED_MODELS} reached and every loaded "
                f"model is in use; loading {model_name} above the cap"
            )
            break
        victim = min(idle, key=lambda name: _whisper_models_last_used.get(name, 0.0))
        logger.info(f"Unloading model {victim} to stay within MAX_LOADED_MODELS={MAX_LOADED_MODELS}")
        del _whisper_models[victim]
        _whisper_models_last_used.pop(victim, None)
        _count_eviction(victim)
        evicted = True
    return evicted


def _take_cached(model_name: str, acquire: bool):
    """The cached model (marked used, and in use if acquire), or None.
    Caller holds _whisper_state_lock."""
    model = _whisper_models.get(model_name)
    if model is None:
        return None
    _whisper_models_last_used[model_name] = time.time()
    if acquire:
        _whisper_models_in_use[model_name] = _whisper_models_in_use.get(model_name, 0) + 1
    return model


def load_whisper_model(model_name: str, _acquire: bool = False):
    """Load WhisperX model with caching (thread-safe).

    With _acquire=True the model is also marked in use (see use_whisper_model)
    in the same step, so it cannot be unloaded between loading and use.
    """
    with _whisper_state_lock:
        model = _take_cached(model_name, _acquire)
    if model is None:
        with _model_load_lock:
            with _whisper_state_lock:
                model = _take_cached(model_name, _acquire)
                freed = False if model is not None else _make_room_for(model_name)
            if model is None:
                if freed:
                    clear_gpu_memory()
                vad_kwargs = vad_load_kwargs()
                logger.info(f"Loading WhisperX model: {model_name}"
                            + (f" (VAD: {vad_kwargs})" if vad_kwargs else ""))
                loaded = whisperx.load_model(
                    model_name,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    download_root=CACHE_DIR,
                    **vad_kwargs,
                )
                logger.info(f"Model {model_name} loaded successfully")
                # Pre-register the eviction counter time series for this model
                # so the row appears in /metrics with value 0 from the moment
                # the model is loaded, instead of only after the first eviction.
                try:
                    from app import metrics as prom_metrics
                    prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=model_name)
                except Exception:
                    pass
                with _whisper_state_lock:
                    _whisper_models[model_name] = loaded
                    model = _take_cached(model_name, _acquire)
    _ensure_eviction_thread()
    return model


@contextmanager
def use_whisper_model(model_name: str):
    """Load (or reuse) a Whisper model and keep it loaded while in use."""
    model = load_whisper_model(model_name, _acquire=True)
    try:
        yield model
    finally:
        with _whisper_state_lock:
            remaining = _whisper_models_in_use.get(model_name, 0) - 1
            if remaining > 0:
                _whisper_models_in_use[model_name] = remaining
            else:
                _whisper_models_in_use.pop(model_name, None)
            _whisper_models_last_used[model_name] = time.time()


def preload_whisper_model(where: str = "startup") -> Optional[str]:
    """Preload PRELOAD_MODEL, if set. Never raises: a failed preload is logged
    and the model loads on the first request that needs it.

    Skipped on the qwen3 and external backends, whose transcription does not
    use a Whisper model (only task=translate does, loaded on demand).
    """
    if not PRELOAD_MODEL:
        logger.info(f"{where}: no PRELOAD_MODEL set; models load on first use (default: {DEFAULT_MODEL})")
        return None
    if ASR_BACKEND in ("qwen3", "external"):
        logger.info(
            f"{where}: ASR_BACKEND={ASR_BACKEND} does not transcribe with a Whisper model; "
            f"skipping the preload of {PRELOAD_MODEL}"
        )
        return None
    try:
        name = _resolve_default(PRELOAD_MODEL)
        logger.info(f"{where}: preloading model {name}")
        load_whisper_model(name)
        logger.info(f"{where}: preloaded model {name}")
        return name
    except Exception as e:
        logger.error(f"{where}: failed to preload model {PRELOAD_MODEL!r}: {e}")
        return None


def _ensure_eviction_thread():
    """Lazily start the idle-model eviction daemon (no-op if disabled)."""
    global _eviction_thread_started
    if MODEL_KEEP_ALIVE_SECONDS <= 0 or _eviction_thread_started:
        return
    with _eviction_thread_lock:
        if _eviction_thread_started:
            return
        t = threading.Thread(
            target=_eviction_loop, daemon=True, name="model-evictor"
        )
        t.start()
        _eviction_thread_started = True
        logger.info(
            f"Idle model eviction enabled: unload after "
            f"{MODEL_KEEP_ALIVE_SECONDS}s idle, sweep every "
            f"{MODEL_EVICTION_INTERVAL_SECONDS}s"
        )


def _evict_from_cache(
    cache: dict,
    last_used: dict,
    label: str,
    now: float,
    with_metrics: bool = False,
    in_use: Optional[dict] = None,
    lock: Optional[threading.Lock] = None,
) -> bool:
    """Evict entries idle longer than MODEL_KEEP_ALIVE_SECONDS from a model cache.

    Snapshots last_used under the lock, then re-checks inside the lock before
    deleting to avoid racing against concurrent loaders.
    Returns True if at least one entry was evicted.
    """
    lock = lock or _model_load_lock
    with lock:
        snapshot = list(last_used.items())
    candidates = [k for k, last in snapshot
                  if now - last > MODEL_KEEP_ALIVE_SECONDS and k in cache]
    evicted_any = False
    for key in candidates:
        with lock:
            last = last_used.get(key, 0)
            if in_use and in_use.get(key):
                continue
            if key in cache and now - last > MODEL_KEEP_ALIVE_SECONDS:
                logger.info(f"Evicting idle {label} {key}")
                del cache[key]
                last_used.pop(key, None)
                evicted_any = True
                if with_metrics:
                    try:
                        from app import metrics as prom_metrics
                        prom_metrics.MODEL_EVICTIONS_TOTAL.labels(model=key).inc()
                    except Exception:
                        pass
    return evicted_any


def _eviction_loop():
    global _diarize_pipeline, _diarize_last_used
    while True:
        time.sleep(MODEL_EVICTION_INTERVAL_SECONDS)
        if MODEL_KEEP_ALIVE_SECONDS <= 0:
            continue
        now = time.time()
        evicted_any = _evict_from_cache(
            _whisper_models, _whisper_models_last_used, "model", now, with_metrics=True,
            in_use=_whisper_models_in_use, lock=_whisper_state_lock,
        )
        evicted_any |= _evict_from_cache(
            _align_models, _align_models_last_used, "alignment model for language", now
        )

        # Sweep idle diarization pipeline (singleton).
        if (
            _diarize_last_used is not None
            and now - _diarize_last_used > MODEL_KEEP_ALIVE_SECONDS
        ):
            with _model_load_lock:
                if (
                    _diarize_last_used is not None
                    and now - _diarize_last_used > MODEL_KEEP_ALIVE_SECONDS
                    and _diarize_pipeline is not None
                ):
                    logger.info("Evicting idle diarization pipeline")
                    _diarize_pipeline = None
                    _diarize_last_used = None
                    evicted_any = True

        if evicted_any:
            clear_gpu_memory()


def load_align_model(language_code: str):
    """Load alignment model with per-language caching (thread-safe)."""
    if language_code not in _align_models:
        with _model_load_lock:
            if language_code not in _align_models:
                logger.info(
                    f"Loading alignment model for language: {language_code} "
                    f"on {ALIGN_DEVICE}"
                )
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=ALIGN_DEVICE,
                    model_dir=CACHE_DIR,
                )
                model_a = prepare_alignment_model(model_a, metadata, CACHE_DIR)
                _align_models[language_code] = (model_a, metadata)
                logger.info(f"Alignment model for {language_code} loaded")
    with _model_load_lock:
        _align_models_last_used[language_code] = time.time()
    _ensure_eviction_thread()
    return _align_models[language_code]


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into `base` in place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _set_scoped(params: dict, section: str, key: str, value: float) -> bool:
    """Set params[section][key]=value only if that exact path already exists."""
    sec = params.get(section)
    if isinstance(sec, dict) and key in sec:
        sec[key] = value
        return True
    return False


def _apply_diarize_tuning(pipeline_wrapper: DiarizationPipeline) -> None:
    """
    Apply env-configured hyperparameter overrides to the underlying pyannote
    pipeline. No-op unless at least one DIARIZE_* tuning var is set.

    Reads the pipeline's *actual* instantiated parameters and merges overrides
    into them, so this stays correct regardless of community-1's internal
    parameter schema. Any failure is logged and swallowed -- diarization then
    runs with published defaults rather than breaking.
    """
    if not any([DIARIZE_CLUSTERING_THRESHOLD, DIARIZE_MIN_DURATION_OFF, DIARIZE_PARAM_OVERRIDES]):
        return

    pyannote_pipeline = getattr(pipeline_wrapper, "model", None)
    if pyannote_pipeline is None:
        logger.warning("Diarization tuning requested but underlying pyannote pipeline not accessible; using defaults")
        return

    try:
        current = pyannote_pipeline.parameters(instantiated=True)
        params = copy.deepcopy(dict(current))
    except Exception as e:
        logger.warning(f"Could not read diarization pipeline parameters for tuning ({e}); using defaults")
        return

    logger.info(f"Diarization default hyperparameters: {params}")

    applied = []
    if DIARIZE_CLUSTERING_THRESHOLD is not None:
        try:
            val = float(DIARIZE_CLUSTERING_THRESHOLD)
        except ValueError:
            logger.warning(f"DIARIZE_CLUSTERING_THRESHOLD={DIARIZE_CLUSTERING_THRESHOLD!r} is not a number; ignoring")
        else:
            if _set_scoped(params, "clustering", "threshold", val):
                applied.append(f"clustering.threshold={val}")
            else:
                logger.warning(
                    "DIARIZE_CLUSTERING_THRESHOLD set but no clustering.threshold in pipeline "
                    "params (see logged schema above); ignoring"
                )
    if DIARIZE_MIN_DURATION_OFF is not None:
        try:
            val = float(DIARIZE_MIN_DURATION_OFF)
        except ValueError:
            logger.warning(f"DIARIZE_MIN_DURATION_OFF={DIARIZE_MIN_DURATION_OFF!r} is not a number; ignoring")
        else:
            if _set_scoped(params, "segmentation", "min_duration_off", val):
                applied.append(f"segmentation.min_duration_off={val}")
            else:
                logger.warning(
                    "DIARIZE_MIN_DURATION_OFF set but no segmentation.min_duration_off in pipeline "
                    "params (see logged schema above); ignoring"
                )
    if DIARIZE_PARAM_OVERRIDES:
        try:
            overrides = json.loads(DIARIZE_PARAM_OVERRIDES)
            if not isinstance(overrides, dict):
                raise ValueError("DIARIZE_PARAM_OVERRIDES must be a JSON object")
            _deep_merge(params, overrides)
            applied.append(f"json_overrides={overrides}")
        except Exception as e:
            logger.warning(f"DIARIZE_PARAM_OVERRIDES could not be applied ({e}); ignoring")

    if not applied:
        return

    try:
        pyannote_pipeline.instantiate(params)
        logger.info(f"Applied diarization hyperparameter overrides: {', '.join(applied)}")
    except Exception as e:
        logger.warning(f"Failed to apply diarization hyperparameter overrides ({e}); using defaults")


def load_diarize_pipeline() -> DiarizationPipeline:
    """Load diarization pipeline (singleton, thread-safe)."""
    global _diarize_pipeline, _diarize_last_used
    if _diarize_pipeline is None:
        with _model_load_lock:
            if _diarize_pipeline is None:
                logger.info("Loading diarization pipeline: pyannote/speaker-diarization-community-1")
                pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-community-1",
                    use_auth_token=HF_TOKEN,
                    device=torch.device(DEVICE),
                )
                _apply_diarize_tuning(pipeline)
                _diarize_pipeline = pipeline
                logger.info("Diarization pipeline loaded")
    with _model_load_lock:
        _diarize_last_used = time.time()
    _ensure_eviction_thread()
    return _diarize_pipeline


# ---------------------------------------------------------------------------
# Stage 1 -- Transcription
# ---------------------------------------------------------------------------
def transcribe(
    audio: np.ndarray,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
) -> dict:
    """Run WhisperX transcription and return raw result dict."""
    if ASR_BACKEND in ("qwen3", "external"):
        if task == "transcribe":
            if ASR_BACKEND == "qwen3":
                from app import qwen3_backend as backend_mod
            else:
                from app import external_backend as backend_mod

            context = " ".join(p for p in (initial_prompt, hotwords) if p) or None
            logger.info(f"Starting transcription ({ASR_BACKEND} backend)...")
            result = backend_mod.transcribe(audio, language=language, context=context)
            logger.info(
                f"Transcription complete. Detected language: {result.get('language')}"
            )
            clear_gpu_memory()
            return result
        logger.warning(
            f"ASR_BACKEND={ASR_BACKEND} does not support task=translate; "
            "using whisper backend for this request"
        )

    model_name = resolve_model_name(model_name)
    transcribe_options: Dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "language": language,
        "task": task,
    }

    with use_whisper_model(model_name) as whisper_model:
        # Set per-request options on the model's transcription options.
        # The model is cached/shared, so we must reset after transcription.
        if hotwords is not None:
            whisper_model.options.hotwords = hotwords
        if initial_prompt is not None:
            whisper_model.options.initial_prompt = initial_prompt

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
    """Run alignment to get word-level timestamps (Wav2Vec2, or the Qwen3
    forced aligner for qwen3-backend results and external results without
    provider timestamps)."""
    if result.get("_asr_backend") == "qwen3" or result.get("_qwen_align"):
        from app import qwen3_backend

        logger.info("Aligning timestamps (qwen3 forced aligner)...")
        try:
            result = qwen3_backend.align(audio, result)
            logger.info("Timestamp alignment complete")
            clear_gpu_memory()
        except Exception as e:
            logger.warning(
                f"Timestamp alignment failed: {e}, continuing without word-level timestamps"
            )
        return result

    detected_language = result.get("language", "en")
    logger.info("Aligning timestamps...")
    try:
        model_a, metadata = load_align_model(detected_language)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            ALIGN_DEVICE,
            return_char_alignments=False,
        )
        with _model_load_lock:
            _align_models_last_used[detected_language] = time.time()
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
    """
    global _diarize_last_used

    if not HF_TOKEN:
        logger.warning("Speaker diarization requested but HF_TOKEN not set")
        return result, None

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

        result = whisperx.assign_word_speakers(
            diarize_segments, result, fill_nearest=DIARIZE_FILL_NEAREST
        )
        # qwen3 and text-only external segments are built from silence gaps
        # or chunk spans alone, so a segment can span several speakers'
        # turns. Now that words carry speaker labels, rebuild the segments
        # so each one holds a single speaker's turn. Opt-in for the whisper
        # backend via RESEGMENT_BY_SPEAKER.
        if (
            result.get("_asr_backend") == "qwen3"
            or result.get("_qwen_align")
            or RESEGMENT_BY_SPEAKER
        ):
            from app import qwen3_backend

            result = qwen3_backend.resegment_by_speaker(result)
        with _model_load_lock:
            _diarize_last_used = time.time()
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
    """
    Run the full 3-stage pipeline: transcribe -> align -> diarize.

    Returns (result, speaker_embeddings_or_None).
    """
    result = transcribe(
        audio,
        model_name=model_name,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )

    # The qwen3 backend and text-only external results produce coarse
    # chunk-level segments, so speaker assignment needs word-level
    # timestamps even if the caller did not ask for them; the whisper
    # backend keeps its original behaviour.
    internally_aligned = (
        result.get("_asr_backend") == "qwen3" or result.get("_qwen_align")
    )
    needs_align = word_timestamps or (should_diarize and internally_aligned)
    if needs_align:
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

    # Non-whisper backends may align even when the caller did not ask for
    # word timestamps (speaker assignment needs them); strip the word-level
    # data from the response in that case, matching the whisper backend's
    # response shape.
    if not word_timestamps and internally_aligned:
        result.pop("word_segments", None)
        for seg in result.get("segments", []):
            seg.pop("words", None)

    result.pop("_asr_backend", None)
    result.pop("_language_name", None)
    result.pop("_qwen_align", None)
    return result, speaker_embeddings
