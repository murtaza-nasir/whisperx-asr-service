"""Optional Qwen3-ASR backend (ASR_BACKEND=qwen3).

Transcribes with Qwen/Qwen3-ASR-*-hf and produces word timestamps with
Qwen/Qwen3-ForcedAligner-0.6B-hf, both via stock transformers (>= 5.13,
where the qwen3_asr architecture is natively supported). No extra
packages are required and model weights download lazily into the HF
cache like every other model in the service.

Differences from the default whisper backend:
- The Whisper model name is ignored; QWEN3_ASR_MODEL selects the model.
- task=translate is not supported (the caller falls back to whisper).
- The forced aligner is language-agnostic across its supported set, so
  code-switched audio (e.g. zh/en) aligns without a per-language model.
- hotwords/initial_prompt are passed as free-text context in the system
  message (Qwen3-ASR context biasing).
"""
import logging
import os
import threading
import time
from typing import List, Optional, Tuple

from app.env import env_str as _env_or_default

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# The encoder handles up to ~5 min per pass, but shorter chunks are far less
# prone to early-EOS truncation and repetition loops under greedy decoding.
CHUNK_SECONDS = int(os.getenv("QWEN3_CHUNK_SECONDS", "90") or "90")
MAX_NEW_TOKENS = 4096
# Cap generation relative to chunk duration so a repetition loop cannot blow
# up a chunk (real speech stays well under 20 tokens/second).
TOKENS_PER_SECOND_CAP = 20

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


ASR_MODEL_ID = _env_or_default("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B-hf")
ALIGNER_MODEL_ID = _env_or_default(
    "QWEN3_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B-hf"
)

# Standing context prepended to every request's system message. Important for
# code-switched audio: with no language hint and no context, Qwen3-ASR picks
# one language per chunk and TRANSLATES the other language into it. Any hint
# (a language, or a note that the audio is mixed) makes it transcribe
# verbatim. Empty by default to leave single-language behaviour untouched.
DEFAULT_CONTEXT = _env_or_default("QWEN3_DEFAULT_CONTEXT", "")

_load_lock = threading.Lock()
_asr = None  # (processor, model)
_aligner = None  # (processor, model)


def _load_asr():
    global _asr
    if _asr is None:
        with _load_lock:
            if _asr is None:
                from transformers import AutoProcessor, AutoModelForMultimodalLM

                logger.info(f"Loading Qwen3-ASR model: {ASR_MODEL_ID}")
                t0 = time.time()
                processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
                dtype = torch.float16 if DEVICE == "cuda" else torch.float32
                model = AutoModelForMultimodalLM.from_pretrained(
                    ASR_MODEL_ID, dtype=dtype
                ).to(DEVICE)
                model.eval()
                _asr = (processor, model)
                logger.info(f"Qwen3-ASR loaded in {time.time()-t0:.1f}s")
    return _asr


def _load_aligner():
    global _aligner
    if _aligner is None:
        with _load_lock:
            if _aligner is None:
                from transformers import AutoProcessor, AutoModelForTokenClassification

                logger.info(f"Loading Qwen3 forced aligner: {ALIGNER_MODEL_ID}")
                t0 = time.time()
                processor = AutoProcessor.from_pretrained(ALIGNER_MODEL_ID)
                dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
                model = AutoModelForTokenClassification.from_pretrained(
                    ALIGNER_MODEL_ID, dtype=dtype
                ).to(DEVICE)
                model.eval()
                _aligner = (processor, model)
                logger.info(f"Qwen3 forced aligner loaded in {time.time()-t0:.1f}s")
    return _aligner


def _language_name(language: Optional[str]) -> Optional[str]:
    """Resolve a code or name to the canonical full name, None if unknown."""
    if not language:
        return None
    try:
        from transformers.models.qwen3_asr.processing_qwen3_asr import resolve_language

        return resolve_language(language)
    except Exception:
        return None


def _language_code(name: Optional[str]) -> Optional[str]:
    """Map a canonical full language name back to its ISO code."""
    if not name:
        return None
    try:
        from transformers.models.qwen3_asr.processing_qwen3_asr import (
            LANGUAGE_CODE_TO_NAME,
        )

        for code, full in LANGUAGE_CODE_TO_NAME.items():
            if full.lower() == name.lower():
                return code
    except Exception:
        pass
    return None


def _chunks(audio: np.ndarray) -> List[Tuple[float, np.ndarray]]:
    """Split audio into (start_seconds, samples) chunks of CHUNK_SECONDS."""
    size = CHUNK_SECONDS * SAMPLE_RATE
    return [
        (i / SAMPLE_RATE, audio[i : i + size]) for i in range(0, len(audio), size)
    ]


def transcribe(
    audio: np.ndarray,
    language: Optional[str] = None,
    context: Optional[str] = None,
) -> dict:
    """Transcribe audio in chunks. Returns a whisperx-shaped result dict
    with one segment per chunk (word timestamps come from align())."""
    processor, model = _load_asr()

    system_parts = []
    if DEFAULT_CONTEXT:
        system_parts.append(DEFAULT_CONTEXT)
    if context:
        system_parts.append(context)
    lang_name = _language_name(language)
    if lang_name:
        system_parts.append(lang_name)
    system_text = "\n".join(system_parts) if system_parts else None

    def _generate(
        chunk: np.ndarray, system: Optional[str], seed_lang: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        conversation = []
        if system is not None:
            conversation.append(
                {"role": "system", "content": [{"type": "text", "text": system}]}
            )
        conversation.append(
            {"role": "user", "content": [{"type": "audio", "audio": chunk}]}
        )
        inputs = processor.apply_chat_template(
            [conversation], tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, model.dtype)
        if seed_lang is not None:
            # Seed the assistant turn with the output format and language.
            # This is what makes code-switched audio transcribe verbatim: a
            # language hint in the system message alone makes the model
            # translate everything into the dominant language instead.
            seed = processor.tokenizer(
                f"language {seed_lang}<asr_text>",
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(model.device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], seed], dim=1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(seed)], dim=1
            )
        max_new = min(
            MAX_NEW_TOKENS,
            int(len(chunk) / SAMPLE_RATE * TOKENS_PER_SECOND_CAP) + 128,
        )
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        if seed_lang is not None:
            text = processor.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            ).strip()
            return text, seed_lang
        parsed = processor.decode(generated_ids, return_format="parsed")[0]
        return (parsed.get("transcription") or "").strip(), parsed.get("language")

    segments = []
    detected_name = lang_name
    for start_s, chunk in _chunks(audio):
        chunk_s = len(chunk) / SAMPLE_RATE
        end_s = start_s + chunk_s
        text, chunk_lang = _generate(chunk, system_text, seed_lang=lang_name)
        # Degenerate-output guard: a long chunk yielding almost no text
        # usually means the model went off-format (e.g. echoed the context
        # instead of transcribing). Retry once without the system message.
        if system_text is not None and chunk_s > 60 and len(text) < chunk_s * 0.5:
            logger.warning(
                f"Qwen3-ASR chunk {start_s:.0f}-{end_s:.0f}s produced only "
                f"{len(text)} chars; retrying without context"
            )
            retry_text, retry_lang = _generate(chunk, None, seed_lang=lang_name)
            if len(retry_text) > len(text):
                text, chunk_lang = retry_text, retry_lang
        if detected_name is None:
            detected_name = chunk_lang
        if text:
            segments.append({"start": start_s, "end": end_s, "text": text})
        logger.info(
            f"Qwen3-ASR chunk {start_s:.0f}-{end_s:.0f}s: "
            f"{len(text)} chars, language={chunk_lang}"
        )

    return {
        "segments": segments,
        "language": _language_code(detected_name) or (language or "en"),
        "_asr_backend": "qwen3",
        "_language_name": detected_name,
    }


_PUNCT = ".,!?;:)\"'。！？，、；：”’"


def _is_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _join_words(words: List[dict]) -> str:
    out = ""
    for w in words:
        token = w["word"]
        if out and not _is_cjk(token) and not _is_cjk(out[-1]):
            out += " "
        out += token
    return out


def _words_to_segments(
    words: List[dict], max_gap: float = 1.0, max_len: float = 30.0
) -> List[dict]:
    """Group aligned words into segments, splitting on speaker changes
    (when words carry a speaker label), silence gaps, sentence-ending
    punctuation, or excessive segment length."""
    segments: List[dict] = []
    current: List[dict] = []
    current_speaker = None
    last_timed = None
    first_timed = None
    for w in words:
        speaker = w.get("speaker")
        timed = "start" in w and "end" in w
        # Words without timestamps (wav2vec2 skips some tokens) ride along
        # in the current segment; they cannot trigger boundary decisions.
        if current and timed and last_timed is not None:
            gap = w["start"] - last_timed["end"]
            duration = w["end"] - first_timed["start"]
            ended = current[-1]["word"].rstrip()[-1:] in ".。!?！？"
            turn = (
                speaker is not None
                and current_speaker is not None
                and speaker != current_speaker
            )
            if turn or gap > max_gap or duration > max_len or (ended and gap > 0.2):
                segments.append(current)
                current = []
                current_speaker = None
                first_timed = None
        current.append(w)
        if timed:
            last_timed = w
            if first_timed is None:
                first_timed = w
        if speaker is not None:
            current_speaker = speaker
    if current:
        segments.append(current)

    out = []
    for seg in segments:
        timed = [w for w in seg if "start" in w and "end" in w]
        if not timed:
            # No usable timestamps at all: append the text to the previous
            # segment rather than inventing a zero-length one.
            if out:
                out[-1]["text"] = _join_words([{"word": out[-1]["text"]}] + seg)
                out[-1]["words"] = out[-1]["words"] + seg
            continue
        entry = {
            "start": timed[0]["start"],
            "end": timed[-1]["end"],
            "text": _join_words(seg),
            "words": seg,
        }
        speakers = [w["speaker"] for w in seg if w.get("speaker")]
        if speakers:
            entry["speaker"] = max(set(speakers), key=speakers.count)
        out.append(entry)
    return out


def resegment_by_speaker(result: dict) -> dict:
    """Rebuild segments after diarization so each segment holds a single
    speaker's turn. The pre-diarization segments may span several speakers
    because they were built from silence gaps alone.

    Words are collected from the segments, NOT from result["word_segments"]:
    assign_word_speakers labels the segment word dicts in place, and in Ray
    Serve's split pipeline the result crosses process boundaries, so
    word_segments is a separate unlabeled copy rather than the same objects.
    """
    words = [
        w for seg in result.get("segments", []) for w in seg.get("words", [])
    ]
    if not any("start" in w and "end" in w for w in words):
        return result
    result["segments"] = _words_to_segments(words)
    result["word_segments"] = [w for w in words if "start" in w and "end" in w]
    return result


def align(audio: np.ndarray, result: dict) -> dict:
    """Run the Qwen3 forced aligner over each transcribed chunk and
    rebuild segments from word timestamps."""
    processor, model = _load_aligner()

    lang_name = _language_name(result.get("_language_name")) or "English"
    words: List[dict] = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        chunk = audio[
            int(seg["start"] * SAMPLE_RATE) : int(seg["end"] * SAMPLE_RATE)
        ]
        try:
            aligner_inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio=chunk, transcript=text, language=lang_name
            )
            aligner_inputs = aligner_inputs.to(model.device, model.dtype)
            with torch.inference_mode():
                outputs = model(**aligner_inputs)
            timestamps = processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=aligner_inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=model.config.timestamp_token_id,
            )[0]
        except Exception as e:
            # Keep the chunk's text as a single un-timed span so content is
            # never silently dropped from the transcript.
            logger.warning(
                f"Qwen3 alignment failed for chunk "
                f"{seg['start']:.0f}-{seg['end']:.0f}s: {e}; keeping chunk text unaligned"
            )
            words.append(
                {"word": text, "start": seg["start"], "end": seg["end"]}
            )
            continue
        pos = 0
        for item in timestamps:
            token = item["text"]
            # Re-attach the punctuation the aligner's tokenizer stripped, by
            # walking the original transcript in order.
            idx = text.find(token, pos)
            if idx != -1:
                end_idx = idx + len(token)
                while end_idx < len(text) and text[end_idx] in _PUNCT:
                    token += text[end_idx]
                    end_idx += 1
                pos = end_idx
            words.append(
                {
                    "word": token,
                    "start": round(seg["start"] + item["start_time"], 3),
                    "end": round(seg["start"] + item["end_time"], 3),
                }
            )

    if words:
        result["segments"] = _words_to_segments(words)
        result["word_segments"] = words
    return result
