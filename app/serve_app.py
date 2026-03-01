"""
Ray Serve application entry point.

Wraps the existing FastAPI app with @serve.ingress so all current endpoints
(/asr, /v1/audio/transcriptions, /health, etc.) are preserved.  The ASR
pipeline stages run as independent Ray Serve deployments with cross-request
batching.

Start with:
    serve run app.serve_app:app
"""

import os
import logging
import tempfile
import warnings
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import whisperx
from ray import serve

from app.version import __version__
from app.pipeline import (
    DEVICE,
    COMPUTE_TYPE,
    BATCH_SIZE,
    DEFAULT_MODEL,
    format_timestamp,
    sanitize_float_values,
    _whisper_models as loaded_models,
)
from app.schemas import (
    ResponseFormat,
    TranscriptionWord,
    TranscriptionSegment,
    TranscriptionVerboseJsonResponse,
    OpenAIErrorDetail,
    OpenAIErrorResponse,
)
from app.serve_deployments import (
    PIPELINE_STRATEGY,
    FullPipelineDeployment,
    WhisperDeployment,
    AlignDeployment,
    DiarizeDeployment,
)

warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1000"))

MODEL_MAPPING = {
    "whisper-1": os.getenv("OPENAI_WHISPER1_MODEL", DEFAULT_MODEL),
    "whisper-large-v3": "large-v3",
    "whisper-large-v2": "large-v2",
    "whisper-medium": "medium",
    "whisper-small": "small",
    "whisper-base": "base",
    "whisper-tiny": "tiny",
}

AVAILABLE_MODELS = [
    {"id": "whisper-1", "object": "model", "owned_by": "openai"},
    {"id": "whisper-large-v3", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-large-v2", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-medium", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-small", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-base", "object": "model", "owned_by": "whisperx"},
    {"id": "whisper-tiny", "object": "model", "owned_by": "whisperx"},
]


def create_openai_error(status_code, message, error_type="invalid_request_error",
                        param=None, code=None):
    error_response = OpenAIErrorResponse(
        error=OpenAIErrorDetail(message=message, type=error_type, param=param, code=code)
    )
    return JSONResponse(status_code=status_code, content=error_response.model_dump())


fastapi_app = FastAPI(
    title="WhisperX ASR API (Ray Serve)",
    description="Automatic Speech Recognition API with Speaker Diarization using WhisperX",
    version=__version__,
)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
    # The ingress is a lightweight async FastAPI router -- it reads the file,
    # then awaits a GPU replica.  It must accept many concurrent requests so
    # the proxy never blocks while GPU replicas are busy.
    max_ongoing_requests=100,
)
@serve.ingress(fastapi_app)
class ASRIngress:

    def __init__(self, pipeline_handle=None, whisper_handle=None,
                 align_handle=None, diarize_handle=None):
        self._pipeline = pipeline_handle
        self._whisper = whisper_handle
        self._align = align_handle
        self._diarize = diarize_handle

    # ------------------------------------------------------------------
    # Basic endpoints
    # ------------------------------------------------------------------
    @fastapi_app.get("/")
    async def root(self):
        return {
            "status": "running",
            "service": "WhisperX ASR API",
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "serve_mode": "ray",
        }

    @fastapi_app.get("/health")
    async def health_check(self):
        return {
            "status": "healthy",
            "device": DEVICE,
            "loaded_models": list(loaded_models.keys()),
            "serve_mode": "ray",
        }

    @fastapi_app.get("/metrics")
    async def metrics(self):
        return {
            "serve_mode": "ray",
            "device": DEVICE,
            "loaded_models": list(loaded_models.keys()),
        }

    # ------------------------------------------------------------------
    # /asr endpoint
    # ------------------------------------------------------------------
    @fastapi_app.post("/asr")
    async def transcribe_audio(
        self,
        audio_file: UploadFile = File(...),
        task: str = Query("transcribe"),
        language: Optional[str] = Query(None),
        initial_prompt: Optional[str] = Query(None),
        hotwords: Optional[str] = Query(None),
        word_timestamps: bool = Query(True),
        output_format: str = Query("json"),
        output: Optional[str] = Query(None),
        model: str = Query(DEFAULT_MODEL),
        num_speakers: Optional[int] = Query(None),
        min_speakers: Optional[int] = Query(None),
        max_speakers: Optional[int] = Query(None),
        diarize: Optional[bool] = Query(None),
        enable_diarization: Optional[bool] = Query(None),
        return_speaker_embeddings: Optional[bool] = Query(None),
    ):
        temp_audio_path = None
        try:
            if output is not None:
                output_format = output

            if diarize is not None or enable_diarization is not None:
                should_diarize = (diarize is True) or (enable_diarization is True)
            else:
                should_diarize = True
            if return_speaker_embeddings is None:
                return_speaker_embeddings = False

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as temp_file:
                temp_audio_path = temp_file.name
                content = await audio_file.read()
                temp_file.write(content)

            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB.",
                )

            logger.info(f"Processing {audio_file.filename} ({file_size_mb:.1f}MB), model: {model}")
            audio = whisperx.load_audio(temp_audio_path)

            if self._pipeline:
                result, speaker_embeddings = await self._pipeline.run.remote(
                    audio, model_name=model, language=language,
                    task=task, initial_prompt=initial_prompt,
                    hotwords=hotwords,
                    word_timestamps=word_timestamps,
                    should_diarize=should_diarize,
                    num_speakers=num_speakers, min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    return_speaker_embeddings=return_speaker_embeddings,
                )
            else:
                result = await self._whisper.transcribe.remote(
                    audio, model_name=model, language=language,
                    task=task, initial_prompt=initial_prompt,
                    hotwords=hotwords,
                )
                if word_timestamps:
                    result = await self._align.align.remote(audio, result)
                speaker_embeddings = None
                if should_diarize:
                    result, speaker_embeddings = await self._diarize.diarize.remote(
                        audio, result,
                        num_speakers=num_speakers, min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        return_speaker_embeddings=return_speaker_embeddings,
                    )

            detected_language = result.get("language", language or "en")
            return self._format_asr_response(
                result, detected_language, output_format,
                return_speaker_embeddings, speaker_embeddings,
            )

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # OpenAI-compat: /v1/audio/transcriptions
    # ------------------------------------------------------------------
    @fastapi_app.post("/v1/audio/transcriptions")
    async def create_transcription(
        self,
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        hotwords: Optional[str] = Form(None),
        response_format: ResponseFormat = Form(ResponseFormat.JSON),
        temperature: float = Form(0.0, ge=0.0, le=1.0),
    ):
        form_data = await request.form()
        timestamp_granularities = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            timestamp_granularities = []
        if response_format == ResponseFormat.VERBOSE_JSON and not timestamp_granularities:
            timestamp_granularities = ["segment"]

        return await self._process_openai_audio(
            file=file, model=model, language=language, prompt=prompt,
            response_format=response_format, temperature=temperature,
            timestamp_granularities=timestamp_granularities, task="transcribe",
            hotwords=hotwords,
        )

    # ------------------------------------------------------------------
    # OpenAI-compat: /v1/audio/translations
    # ------------------------------------------------------------------
    @fastapi_app.post("/v1/audio/translations")
    async def create_translation(
        self,
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        hotwords: Optional[str] = Form(None),
        response_format: ResponseFormat = Form(ResponseFormat.JSON),
        temperature: float = Form(0.0, ge=0.0, le=1.0),
    ):
        form_data = await request.form()
        timestamp_granularities = form_data.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            timestamp_granularities = []
        if response_format == ResponseFormat.VERBOSE_JSON and not timestamp_granularities:
            timestamp_granularities = ["segment"]

        return await self._process_openai_audio(
            file=file, model=model, language=None, prompt=prompt,
            response_format=response_format, temperature=temperature,
            timestamp_granularities=timestamp_granularities, task="translate",
            hotwords=hotwords,
        )

    # ------------------------------------------------------------------
    # OpenAI-compat: /v1/models
    # ------------------------------------------------------------------
    @fastapi_app.get("/v1/models")
    async def list_models(self):
        return {"object": "list", "data": AVAILABLE_MODELS}

    @fastapi_app.get("/v1/models/{model_id}")
    async def get_model(self, model_id: str):
        for m in AVAILABLE_MODELS:
            if m["id"] == model_id:
                return m
        return create_openai_error(404, f"Model '{model_id}' not found",
                                   code="model_not_found")

    # ------------------------------------------------------------------
    # Shared OpenAI-compat processing
    # ------------------------------------------------------------------
    async def _process_openai_audio(
        self, file, model, language, prompt, response_format,
        temperature, timestamp_granularities, task, hotwords=None,
    ):
        temp_audio_path = None
        try:
            whisperx_model = MODEL_MAPPING.get(model)
            if not whisperx_model:
                if model in ["tiny", "base", "small", "medium", "large-v2", "large-v3"]:
                    whisperx_model = model
                else:
                    return create_openai_error(
                        400,
                        f"Invalid model: {model}. Supported: whisper-1, or whisperx models",
                        param="model",
                    )

            if timestamp_granularities and response_format != ResponseFormat.VERBOSE_JSON:
                return create_openai_error(
                    400,
                    "timestamp_granularities requires response_format='verbose_json'",
                    param="timestamp_granularities",
                )

            if temperature < 0 or temperature > 1:
                return create_openai_error(400, "temperature must be between 0 and 1",
                                           param="temperature")

            suffix = Path(file.filename).suffix if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_audio_path = temp_file.name
                content = await file.read()
                temp_file.write(content)

            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return create_openai_error(
                    413, f"File too large ({file_size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB",
                    code="file_too_large",
                )

            effective_hotwords = hotwords or prompt

            logger.info(f"OpenAI-compat: Processing {file.filename} ({file_size_mb:.1f}MB), model: {whisperx_model}, task: {task}")

            audio = whisperx.load_audio(temp_audio_path)
            duration = len(audio) / 16000

            need_word_timestamps = (
                response_format == ResponseFormat.VERBOSE_JSON
                and "word" in timestamp_granularities
            )

            if self._pipeline:
                result, _ = await self._pipeline.run.remote(
                    audio, model_name=whisperx_model, language=language, task=task,
                    hotwords=effective_hotwords,
                    word_timestamps=need_word_timestamps,
                    should_diarize=False,
                )
            else:
                result = await self._whisper.transcribe.remote(
                    audio, model_name=whisperx_model, language=language, task=task,
                    hotwords=effective_hotwords,
                )
                if need_word_timestamps:
                    result = await self._align.align.remote(audio, result)

            detected_language = result.get("language", language or "en")

            # Format response
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
                srt_content = []
                for i, segment in enumerate(result.get("segments", []), 1):
                    start_time = format_timestamp(segment.get("start", 0))
                    end_time = format_timestamp(segment.get("end", 0))
                    text = segment.get("text", "").strip()
                    srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
                return PlainTextResponse(content="\n".join(srt_content), media_type="text/plain")

            elif response_format == ResponseFormat.VTT:
                vtt_content = ["WEBVTT\n"]
                for segment in result.get("segments", []):
                    start_time = format_timestamp(segment.get("start", 0)).replace(",", ".")
                    end_time = format_timestamp(segment.get("end", 0)).replace(",", ".")
                    text = segment.get("text", "").strip()
                    vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
                return PlainTextResponse(content="\n".join(vtt_content), media_type="text/vtt")

            elif response_format == ResponseFormat.VERBOSE_JSON:
                include_words = "word" in timestamp_granularities
                include_segments = "segment" in timestamp_granularities or not timestamp_granularities

                full_text = " ".join([
                    seg.get("text", "").strip()
                    for seg in result.get("segments", [])
                ]).strip()

                segments = []
                if include_segments:
                    for idx, seg in enumerate(result.get("segments", [])):
                        segments.append(TranscriptionSegment(
                            id=idx,
                            seek=int(seg.get("start", 0) * 100),
                            start=seg.get("start", 0.0),
                            end=seg.get("end", 0.0),
                            text=seg.get("text", "").strip(),
                        ))

                words = None
                if include_words:
                    words = []
                    word_segments = result.get("word_segments", [])
                    if not word_segments:
                        for seg in result.get("segments", []):
                            word_segments.extend(seg.get("words", []))
                    for wd in word_segments:
                        if "word" in wd and "start" in wd and "end" in wd:
                            words.append(TranscriptionWord(
                                word=wd["word"].strip(),
                                start=wd.get("start", 0.0),
                                end=wd.get("end", 0.0),
                            ))

                resp = TranscriptionVerboseJsonResponse(
                    task=task, language=detected_language, duration=duration,
                    text=full_text, segments=segments, words=words,
                )
                return JSONResponse(content=resp.model_dump(exclude_none=True))

            return create_openai_error(400, f"Unsupported response format: {response_format}")

        except HTTPException as e:
            return create_openai_error(e.status_code, e.detail)
        except Exception as e:
            logger.error(f"OpenAI-compat error: {e}", exc_info=True)
            return create_openai_error(500, f"Internal server error: {e}",
                                       error_type="server_error")
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # /asr response formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_asr_response(result, detected_language, output_format,
                             return_speaker_embeddings, speaker_embeddings):
        if output_format == "json":
            response_data = {
                "text": result.get("segments", []),
                "language": detected_language,
                "segments": result.get("segments", []),
                "word_segments": result.get("word_segments", []),
            }
            if return_speaker_embeddings and speaker_embeddings:
                response_data["speaker_embeddings"] = sanitize_float_values(speaker_embeddings)
            return JSONResponse(content=response_data)

        elif output_format == "text":
            text = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
            return {"text": text}

        elif output_format == "srt":
            srt_content = []
            for i, segment in enumerate(result.get("segments", []), 1):
                start_time = format_timestamp(segment.get("start", 0))
                end_time = format_timestamp(segment.get("end", 0))
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")
                if speaker:
                    text = f"[{speaker}] {text}"
                srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
            return {"srt": "\n".join(srt_content)}

        elif output_format == "vtt":
            vtt_content = ["WEBVTT\n"]
            for segment in result.get("segments", []):
                start_time = format_timestamp(segment.get("start", 0)).replace(",", ".")
                end_time = format_timestamp(segment.get("end", 0)).replace(",", ".")
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")
                if speaker:
                    text = f"[{speaker}] {text}"
                vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
            return {"vtt": "\n".join(vtt_content)}

        elif output_format == "tsv":
            tsv_content = ["start\tend\ttext\tspeaker"]
            for segment in result.get("segments", []):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")
                tsv_content.append(f"{start}\t{end}\t{text}\t{speaker}")
            return {"tsv": "\n".join(tsv_content)}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported output format: {output_format}")


# ------------------------------------------------------------------
# Bind deployments into the application graph
# ------------------------------------------------------------------
if PIPELINE_STRATEGY == "split":
    logger.info("Using split strategy: separate deployments per stage")
    app = ASRIngress.bind(
        pipeline_handle=None,
        whisper_handle=WhisperDeployment.bind(),
        align_handle=AlignDeployment.bind(),
        diarize_handle=DiarizeDeployment.bind(),
    )
else:
    logger.info("Using replicate strategy: full pipeline per GPU")
    app = ASRIngress.bind(
        pipeline_handle=FullPipelineDeployment.bind(),
    )
