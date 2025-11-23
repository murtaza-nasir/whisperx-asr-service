"""
WhisperX ASR API Service
Compatible with openai-whisper-asr-webservice API endpoints
"""

import os
import tempfile
import logging
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import whisperx
import torch
from pyannote.audio import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WhisperX ASR API",
    description="Automatic Speech Recognition API with Speaker Diarization using WhisperX",
    version="1.0.0"
)

# Configuration from environment variables
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("CACHE_DIR", "/.cache")

# Model cache
loaded_models = {}

logger.info(f"WhisperX ASR Service initialized on device: {DEVICE}")
logger.info(f"Compute type: {COMPUTE_TYPE}, Batch size: {BATCH_SIZE}")


@app.on_event("startup")
async def startup_event():
    """Preload models on startup"""
    preload_model = os.getenv("PRELOAD_MODEL", None)
    if preload_model:
        logger.info(f"Preloading model on startup: {preload_model}")
        try:
            load_whisper_model(preload_model)
            logger.info(f"Successfully preloaded model: {preload_model}")
        except Exception as e:
            logger.error(f"Failed to preload model {preload_model}: {str(e)}")


def load_whisper_model(model_name: str):
    """Load WhisperX model with caching"""
    if model_name not in loaded_models:
        logger.info(f"Loading WhisperX model: {model_name}")
        try:
            model = whisperx.load_model(
                model_name,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                download_root=CACHE_DIR
            )
            loaded_models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    return loaded_models[model_name]


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "WhisperX ASR API",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE
    }


@app.post("/asr")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    task: str = Form("transcribe"),
    language: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
    word_timestamps: bool = Form(True),
    output_format: str = Form("json"),
    model: str = Form("large-v3"),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    enable_diarization: bool = Form(True)
):
    """
    Main ASR endpoint compatible with openai-whisper-asr-webservice

    Args:
        audio_file: Audio file to transcribe
        task: transcribe or translate
        language: Language code (e.g., 'en', 'es', 'fr')
        initial_prompt: Optional prompt to guide the model
        word_timestamps: Return word-level timestamps
        output_format: json, text, srt, vtt, or tsv
        model: WhisperX model name (tiny, base, small, medium, large-v2, large-v3)
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        enable_diarization: Enable speaker diarization
    """
    temp_audio_path = None

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as temp_file:
            temp_audio_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)

        logger.info(f"Processing audio file: {audio_file.filename}, model: {model}, language: {language}")

        # Load model
        whisper_model = load_whisper_model(model)

        # Step 1: Transcribe with WhisperX
        logger.info("Starting transcription...")
        audio = whisperx.load_audio(temp_audio_path)

        transcribe_options = {
            "batch_size": BATCH_SIZE,
            "language": language,
            "task": task
        }

        if initial_prompt:
            transcribe_options["initial_prompt"] = initial_prompt

        result = whisper_model.transcribe(audio, **transcribe_options)

        detected_language = result.get("language", language or "en")
        logger.info(f"Transcription complete. Detected language: {detected_language}")

        # Step 2: Align whisper output with word-level timestamps
        if word_timestamps:
            logger.info("Aligning timestamps...")
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
                logger.info("Timestamp alignment complete")
            except Exception as e:
                logger.warning(f"Timestamp alignment failed: {str(e)}, continuing without word-level timestamps")

        # Step 3: Speaker diarization (if enabled and HF token available)
        if enable_diarization and HF_TOKEN:
            logger.info("Starting speaker diarization with pyannote community-1...")
            try:
                # Load latest diarization pipeline (pyannote.audio 4.0)
                diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    token=HF_TOKEN
                )
                diarize_model.to(torch.device(DEVICE))

                diarize_options = {}
                if min_speakers:
                    diarize_options["min_speakers"] = min_speakers
                if max_speakers:
                    diarize_options["max_speakers"] = max_speakers

                diarize_segments = diarize_model(audio, **diarize_options)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info("Speaker diarization complete")
            except Exception as e:
                logger.warning(f"Speaker diarization failed: {str(e)}, continuing without diarization")
        elif enable_diarization and not HF_TOKEN:
            logger.warning("Speaker diarization requested but HF_TOKEN not set")

        # Format output based on requested format
        if output_format == "json":
            response_data = {
                "text": result.get("segments", []),
                "language": detected_language,
                "segments": result.get("segments", []),
                "word_segments": result.get("word_segments", [])
            }
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
                start_time = format_timestamp(segment.get("start", 0)).replace(',', '.')
                end_time = format_timestamp(segment.get("end", 0)).replace(',', '.')
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

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "loaded_models": list(loaded_models.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
