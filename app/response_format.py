"""
Formatting a pipeline result as an /asr response.

The synchronous endpoints in app/main.py and app/serve_app.py already had
this code; it is shared here so an asynchronous job's stored result is
formatted exactly the same way when it is fetched, in either serve mode.

The two helpers from app.pipeline are imported inside the function, so this
module can be imported without pulling in the ML stack.
"""

import logging
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

OUTPUT_FORMATS = ("json", "text", "srt", "vtt", "tsv")


def format_asr_response(result: dict, detected_language: str, output_format: str,
                        return_speaker_embeddings: bool = False,
                        speaker_embeddings=None, diarization=None,
                        headers: Optional[dict] = None) -> JSONResponse:
    from app.pipeline import format_timestamp, sanitize_float_values

    segments = result.get("segments", [])

    if output_format == "json":
        response_data = {
            "text": segments,
            "language": detected_language,
            "segments": segments,
            "word_segments": result.get("word_segments", []),
        }
        if return_speaker_embeddings and speaker_embeddings:
            response_data["speaker_embeddings"] = sanitize_float_values(speaker_embeddings)
        if diarization:
            response_data["diarization"] = diarization
        return JSONResponse(content=response_data, headers=headers)

    elif output_format == "text":
        text = " ".join([seg.get("text", "") for seg in segments])
        return JSONResponse(content={"text": text}, headers=headers)

    elif output_format == "srt":
        srt_content = []
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp(segment.get("start", 0))
            end_time = format_timestamp(segment.get("end", 0))
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "")
            if speaker:
                text = f"[{speaker}] {text}"
            srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
        return JSONResponse(content={"srt": "\n".join(srt_content)}, headers=headers)

    elif output_format == "vtt":
        vtt_content = ["WEBVTT\n"]
        for segment in segments:
            start_time = format_timestamp(segment.get("start", 0)).replace(",", ".")
            end_time = format_timestamp(segment.get("end", 0)).replace(",", ".")
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "")
            if speaker:
                text = f"[{speaker}] {text}"
            vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
        return JSONResponse(content={"vtt": "\n".join(vtt_content)}, headers=headers)

    elif output_format == "tsv":
        tsv_content = ["start\tend\ttext\tspeaker"]
        for segment in segments:
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "")
            tsv_content.append(f"{start}\t{end}\t{text}\t{speaker}")
        return JSONResponse(content={"tsv": "\n".join(tsv_content)}, headers=headers)

    raise HTTPException(status_code=400,
                        detail=f"Unsupported output format: {output_format}")


__all__ = ["format_asr_response", "OUTPUT_FORMATS"]
