"""Optional external ASR backend (ASR_BACKEND=external).

Outsources the transcription stage to an OpenAI-compatible API while
alignment, diarization, and speaker embeddings keep running locally.
Two provider shapes are supported, selected by EXTERNAL_ASR_MODE:

- "transcriptions" (default): POST {base}/audio/transcriptions
  (whisper-1, Groq, self-hosted Whisper servers). When the provider
  returns verbose_json with timestamped segments, the existing Wav2Vec2
  alignment stage consumes them directly. Text-only responses fall back
  to the Qwen forced aligner for word timestamps.
- "chat": POST {base}/chat/completions with base64 audio input
  (OpenRouter audio models such as Voxtral, gpt-audio, Gemini; vLLM
  multimodal servers). These return text only, so word timestamps come
  from the Qwen forced aligner, which accepts transcripts from any ASR.

Note: audio is uploaded to the configured provider. This backend is
opt-in and off unless ASR_BACKEND=external is set.
"""
import base64
import io
import logging
import time
import wave
from typing import List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

from app.env import env_str as _env

SAMPLE_RATE = 16000


BASE_URL = _env("EXTERNAL_ASR_BASE_URL").rstrip("/")
API_KEY = _env("EXTERNAL_ASR_API_KEY")
MODEL = _env("EXTERNAL_ASR_MODEL")
MODE = _env("EXTERNAL_ASR_MODE", "transcriptions").lower()
CHUNK_SECONDS = int(_env("EXTERNAL_ASR_CHUNK_SECONDS", "300") or "300")
TIMEOUT_SECONDS = int(_env("EXTERNAL_ASR_TIMEOUT_SECONDS", "600") or "600")
# Aligner selection. "auto" (default) uses Wav2Vec2 when the provider
# returns timestamped segments and the Qwen forced aligner otherwise.
# "qwen" forces the Qwen aligner even for timestamped responses, which is
# language-agnostic across its supported set and handles code-switched
# audio; the reverse (Wav2Vec2 for text-only responses) is not possible
# because Wav2Vec2 needs segment timestamps as anchors.
ALIGNER = _env("EXTERNAL_ASR_ALIGNER", "auto").lower()

CHAT_INSTRUCTION = (
    "Transcribe this audio verbatim in its original spoken language, with "
    "punctuation. Output only the transcription."
)


def _headers() -> dict:
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _wav_bytes(audio: np.ndarray) -> bytes:
    """Encode float32 mono 16 kHz samples as a 16-bit PCM WAV."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _language_code(value: Optional[str], fallback: Optional[str]) -> Optional[str]:
    """Normalize a provider language ('english', 'en') to an ISO code."""
    if not value:
        return fallback
    value = value.strip().lower()
    if len(value) <= 3:
        return value
    try:
        from whisperx.utils import LANGUAGES  # code -> name

        for code, name in LANGUAGES.items():
            if name.lower() == value:
                return code
    except Exception:
        pass
    return fallback


def _chunks(audio: np.ndarray) -> List[Tuple[float, np.ndarray]]:
    size = CHUNK_SECONDS * SAMPLE_RATE
    return [
        (i / SAMPLE_RATE, audio[i : i + size]) for i in range(0, len(audio), size)
    ]


def _require_config():
    missing = [
        name
        for name, val in (
            ("EXTERNAL_ASR_BASE_URL", BASE_URL),
            ("EXTERNAL_ASR_MODEL", MODEL),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"ASR_BACKEND=external requires {', '.join(missing)} to be set"
        )


def transcribe(
    audio: np.ndarray,
    language: Optional[str] = None,
    context: Optional[str] = None,
) -> dict:
    """Transcribe via the configured external API. Returns a whisperx-shaped
    result dict; sets _qwen_align when the provider gave no timestamps."""
    _require_config()
    if MODE == "chat":
        return _transcribe_chat(audio, language, context)
    return _transcribe_endpoint(audio, language, context)


def _transcribe_endpoint(
    audio: np.ndarray, language: Optional[str], context: Optional[str]
) -> dict:
    url = f"{BASE_URL}/audio/transcriptions"
    data = {"model": MODEL, "response_format": "verbose_json"}
    if language:
        data["language"] = language
    if context:
        data["prompt"] = context

    t0 = time.time()
    resp = requests.post(
        url,
        headers=_headers(),
        data=data,
        files={"file": ("audio.wav", _wav_bytes(audio), "audio/wav")},
        timeout=TIMEOUT_SECONDS,
    )
    if resp.status_code == 400 and "response_format" in resp.text:
        # Provider does not support verbose_json; retry plain json.
        data["response_format"] = "json"
        resp = requests.post(
            url,
            headers=_headers(),
            data=data,
            files={"file": ("audio.wav", _wav_bytes(audio), "audio/wav")},
            timeout=TIMEOUT_SECONDS,
        )
    resp.raise_for_status()
    payload = resp.json()

    detected = _language_code(payload.get("language"), language)
    duration = len(audio) / SAMPLE_RATE
    raw_segments = payload.get("segments") or []
    timestamped = [
        s for s in raw_segments if s.get("start") is not None and s.get("end") is not None
    ]
    logger.info(
        f"External ASR ({MODEL}): {len(timestamped)} timestamped segments in "
        f"{time.time()-t0:.1f}s for {duration:.0f}s audio"
    )

    if timestamped:
        result = {
            "segments": [
                {
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "text": (s.get("text") or "").strip(),
                }
                for s in timestamped
                if (s.get("text") or "").strip()
            ],
            "language": detected or "en",
            "_asr_backend": "external",
        }
        if ALIGNER == "qwen":
            result["_qwen_align"] = True
            _stash_language_name(result, detected or language)
        return result

    # Text-only response: one chunk-span segment, timestamps come from the
    # Qwen forced aligner downstream.
    text = (payload.get("text") or "").strip()
    result = {
        "segments": [{"start": 0.0, "end": duration, "text": text}] if text else [],
        "language": detected or "en",
        "_asr_backend": "external",
        "_qwen_align": True,
    }
    _stash_language_name(result, detected or language)
    return result


def _transcribe_chat(
    audio: np.ndarray, language: Optional[str], context: Optional[str]
) -> dict:
    url = f"{BASE_URL}/chat/completions"
    instruction = CHAT_INSTRUCTION
    if language:
        instruction += f" The audio language is {language}."
    if context:
        instruction += f" Context: {context}"

    segments = []
    for start_s, chunk in _chunks(audio):
        end_s = start_s + len(chunk) / SAMPLE_RATE
        body = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(_wav_bytes(chunk)).decode(),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        }
        t0 = time.time()
        resp = requests.post(
            url, headers=_headers(), json=body, timeout=TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        text = (
            resp.json()["choices"][0]["message"].get("content") or ""
        ).strip()
        logger.info(
            f"External ASR chat ({MODEL}) chunk {start_s:.0f}-{end_s:.0f}s: "
            f"{len(text)} chars in {time.time()-t0:.1f}s"
        )
        if text:
            segments.append({"start": start_s, "end": end_s, "text": text})

    result = {
        "segments": segments,
        "language": language or "en",
        "_asr_backend": "external",
        "_qwen_align": True,
    }
    _stash_language_name(result, language)
    return result


def _stash_language_name(result: dict, language: Optional[str]) -> None:
    """Store the full language name the Qwen aligner expects, when known."""
    try:
        from app.qwen3_backend import _language_name

        result["_language_name"] = _language_name(language)
    except Exception:
        result["_language_name"] = None
