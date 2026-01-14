"""
OpenAI-compatible Whisper API endpoints
POST /v1/audio/transcriptions
POST /v1/audio/translations
GET /v1/models
"""

import os
import tempfile
import logging
import time
from typing import Optional, List, Union
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import whisperx

from app.schemas import (
    ResponseFormat,
    TranscriptionWord,
    TranscriptionSegment,
    TranscriptionVerboseJsonResponse,
    OpenAIErrorDetail,
    OpenAIErrorResponse,
)

# Import shared resources from main module
from app.main import (
    load_whisper_model,
    clear_gpu_memory,
    format_timestamp,
    DEVICE,
    BATCH_SIZE,
    CACHE_DIR,
    DEFAULT_MODEL,
    MAX_FILE_SIZE_MB,
    loaded_models,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["OpenAI Compatible"])
models_router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])

# Model mapping: OpenAI model names to WhisperX model names
MODEL_MAPPING = {
    "whisper-1": os.getenv("OPENAI_WHISPER1_MODEL", DEFAULT_MODEL),
    "whisper-large-v3": "large-v3",
    "whisper-large-v2": "large-v2",
    "whisper-medium": "medium",
    "whisper-small": "small",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}


def create_openai_error(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None
) -> JSONResponse:
    """Create OpenAI-compatible error response"""
    error_response = OpenAIErrorResponse(
        error=OpenAIErrorDetail(
            message=message,
            type=error_type,
            param=param,
            code=code
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump()
    )


def format_verbose_json_response(
    result: dict,
    task: str,
    language: str,
    duration: float,
    include_words: bool,
    include_segments: bool
) -> TranscriptionVerboseJsonResponse:
    """Format WhisperX result as OpenAI verbose_json response"""

    # Build full text from segments
    full_text = " ".join([
        seg.get("text", "").strip()
        for seg in result.get("segments", [])
    ]).strip()

    # Build segments list
    segments = []
    if include_segments:
        for idx, seg in enumerate(result.get("segments", [])):
            segments.append(TranscriptionSegment(
                id=idx,
                seek=int(seg.get("start", 0) * 100),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
                tokens=[],
                temperature=0.0,
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=0.0
            ))

    # Build words list from word_segments
    words = None
    if include_words:
        words = []
        word_segments = result.get("word_segments", [])
        # Also check segments for word-level data
        if not word_segments:
            for seg in result.get("segments", []):
                word_segments.extend(seg.get("words", []))

        for word_data in word_segments:
            if "word" in word_data and "start" in word_data and "end" in word_data:
                words.append(TranscriptionWord(
                    word=word_data["word"].strip(),
                    start=word_data.get("start", 0.0),
                    end=word_data.get("end", 0.0)
                ))

    return TranscriptionVerboseJsonResponse(
        task=task,
        language=language,
        duration=duration,
        text=full_text,
        segments=segments,
        words=words
    )


def format_srt_response(result: dict) -> str:
    """Format WhisperX result as SRT subtitle format"""
    srt_content = []
    for i, segment in enumerate(result.get("segments", []), 1):
        start_time = format_timestamp(segment.get("start", 0))
        end_time = format_timestamp(segment.get("end", 0))
        text = segment.get("text", "").strip()
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_content)


def format_vtt_response(result: dict) -> str:
    """Format WhisperX result as WebVTT subtitle format"""
    vtt_content = ["WEBVTT\n"]
    for segment in result.get("segments", []):
        # VTT uses period for milliseconds, not comma
        start_time = format_timestamp(segment.get("start", 0)).replace(',', '.')
        end_time = format_timestamp(segment.get("end", 0)).replace(',', '.')
        text = segment.get("text", "").strip()
        vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
    return "\n".join(vtt_content)


async def process_audio(
    file: UploadFile,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    response_format: ResponseFormat,
    temperature: float,
    timestamp_granularities: List[str],
    task: str = "transcribe"
) -> Union[JSONResponse, PlainTextResponse]:
    """
    Core audio processing function shared by transcriptions and translations endpoints
    """
    temp_audio_path = None

    try:
        # Validate model
        whisperx_model = MODEL_MAPPING.get(model)
        if not whisperx_model:
            # If not in mapping, try using directly (allows large-v3, etc.)
            if model in ["tiny", "base", "small", "medium", "large-v2", "large-v3"]:
                whisperx_model = model
            else:
                return create_openai_error(
                    400,
                    f"Invalid model: {model}. Supported: whisper-1, or whisperx models (tiny, base, small, medium, large-v2, large-v3)",
                    param="model"
                )

        # Validate timestamp_granularities requires verbose_json
        if timestamp_granularities and response_format != ResponseFormat.VERBOSE_JSON:
            return create_openai_error(
                400,
                "timestamp_granularities requires response_format='verbose_json'",
                param="timestamp_granularities"
            )

        # Validate temperature range
        if temperature < 0 or temperature > 1:
            return create_openai_error(
                400,
                "temperature must be between 0 and 1",
                param="temperature"
            )

        # Save uploaded file to temporary location
        suffix = Path(file.filename).suffix if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_audio_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return create_openai_error(
                413,
                f"File too large ({file_size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB",
                code="file_too_large"
            )

        logger.info(f"OpenAI-compat: Processing {file.filename} ({file_size_mb:.1f}MB), model: {whisperx_model}, task: {task}")

        # Load model (uses cached if available)
        whisper_model = load_whisper_model(whisperx_model)

        # Load audio
        audio = whisperx.load_audio(temp_audio_path)
        duration = len(audio) / 16000  # WhisperX loads at 16kHz

        # Transcription options
        transcribe_options = {
            "batch_size": BATCH_SIZE,
            "language": language,
            "task": task
        }
        if prompt:
            transcribe_options["initial_prompt"] = prompt

        # Run transcription
        result = whisper_model.transcribe(audio, **transcribe_options)
        detected_language = result.get("language", language or "en")

        clear_gpu_memory()

        # Determine if we need word-level alignment
        need_word_timestamps = (
            response_format == ResponseFormat.VERBOSE_JSON and
            "word" in timestamp_granularities
        )

        # Run alignment if needed for word timestamps
        if need_word_timestamps:
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=DEVICE,
                    model_dir=CACHE_DIR
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    DEVICE,
                    return_char_alignments=False
                )
                del model_a
                clear_gpu_memory()
            except Exception as e:
                logger.warning(f"Word alignment failed: {e}")
                # Continue without word timestamps

        # Format response based on requested format
        if response_format == ResponseFormat.JSON:
            full_text = " ".join([
                seg.get("text", "").strip()
                for seg in result.get("segments", [])
            ]).strip()
            return JSONResponse(content={"text": full_text})

        elif response_format == ResponseFormat.TEXT:
            full_text = " ".join([
                seg.get("text", "").strip()
                for seg in result.get("segments", [])
            ]).strip()
            return PlainTextResponse(content=full_text)

        elif response_format == ResponseFormat.SRT:
            srt_content = format_srt_response(result)
            return PlainTextResponse(content=srt_content, media_type="text/plain")

        elif response_format == ResponseFormat.VTT:
            vtt_content = format_vtt_response(result)
            return PlainTextResponse(content=vtt_content, media_type="text/vtt")

        elif response_format == ResponseFormat.VERBOSE_JSON:
            include_words = "word" in timestamp_granularities
            include_segments = "segment" in timestamp_granularities or not timestamp_granularities

            response = format_verbose_json_response(
                result=result,
                task=task,
                language=detected_language,
                duration=duration,
                include_words=include_words,
                include_segments=include_segments
            )
            return JSONResponse(content=response.model_dump(exclude_none=True))

        # Should not reach here
        return create_openai_error(400, f"Unsupported response format: {response_format}")

    except HTTPException as e:
        return create_openai_error(e.status_code, e.detail)

    except Exception as e:
        logger.error(f"OpenAI-compat error: {str(e)}", exc_info=True)
        return create_openai_error(
            500,
            f"Internal server error: {str(e)}",
            error_type="server_error"
        )

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@router.post("/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(..., description="Model ID (whisper-1, large-v3, etc.)"),
    language: Optional[str] = Form(None, description="ISO-639-1 language code"),
    prompt: Optional[str] = Form(None, description="Guidance text/context"),
    response_format: ResponseFormat = Form(
        ResponseFormat.JSON,
        description="Output format: json, text, srt, verbose_json, vtt"
    ),
    temperature: float = Form(0.0, ge=0.0, le=1.0, description="Sampling temperature"),
):
    """
    Transcribes audio into the input language.

    OpenAI-compatible endpoint: POST /v1/audio/transcriptions

    Returns transcription in the requested format. When using verbose_json,
    you can request word-level and/or segment-level timestamps via
    timestamp_granularities[].
    """
    # Parse timestamp_granularities from form data (handles timestamp_granularities[]=word format)
    form_data = await request.form()
    timestamp_granularities = form_data.getlist("timestamp_granularities[]")

    # Default timestamp_granularities to segment if verbose_json but not specified
    if not timestamp_granularities:
        timestamp_granularities = []
    if response_format == ResponseFormat.VERBOSE_JSON and not timestamp_granularities:
        timestamp_granularities = ["segment"]

    return await process_audio(
        file=file,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        task="transcribe"
    )


@router.post("/translations")
async def create_translation(
    request: Request,
    file: UploadFile = File(..., description="Audio file to translate"),
    model: str = Form(..., description="Model ID (whisper-1, large-v3, etc.)"),
    prompt: Optional[str] = Form(None, description="Guidance text/context"),
    response_format: ResponseFormat = Form(
        ResponseFormat.JSON,
        description="Output format: json, text, srt, verbose_json, vtt"
    ),
    temperature: float = Form(0.0, ge=0.0, le=1.0, description="Sampling temperature"),
):
    """
    Translates audio into English.

    OpenAI-compatible endpoint: POST /v1/audio/translations

    Similar to transcriptions but always outputs English text regardless
    of the source language.
    """
    # Parse timestamp_granularities from form data (handles timestamp_granularities[]=word format)
    form_data = await request.form()
    timestamp_granularities = form_data.getlist("timestamp_granularities[]")

    if not timestamp_granularities:
        timestamp_granularities = []
    if response_format == ResponseFormat.VERBOSE_JSON and not timestamp_granularities:
        timestamp_granularities = ["segment"]

    return await process_audio(
        file=file,
        model=model,
        language=None,  # Translation doesn't take language param
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        timestamp_granularities=timestamp_granularities,
        task="translate"
    )


# Available whisper models
AVAILABLE_MODELS = [
    {"id": "whisper-1", "object": "model", "owned_by": "openai"},
    {"id": "whisper-large-v3", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-large-v2", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-medium", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-small", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-base", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-tiny", "object": "model", "owned_by": "whisperx"},
]


@models_router.get("/models")
async def list_models():
    """
    List available models.

    OpenAI-compatible endpoint: GET /v1/models

    Returns a list of available whisper models that can be used for transcription.
    """
    return {
        "object": "list",
        "data": AVAILABLE_MODELS
    }


@models_router.get("/models/{model_id}")
async def get_model(model_id: str):
    """
    Get details about a specific model.

    OpenAI-compatible endpoint: GET /v1/models/{model_id}
    """
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return model

    return create_openai_error(
        404,
        f"Model '{model_id}' not found",
        error_type="invalid_request_error",
        code="model_not_found"
    )
