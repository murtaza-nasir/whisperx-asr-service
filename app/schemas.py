"""
OpenAI-compatible Pydantic models for Whisper API
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ResponseFormat(str, Enum):
    """Supported response formats for transcription"""
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    VERBOSE_JSON = "verbose_json"


class TimestampGranularity(str, Enum):
    """Timestamp granularity options"""
    WORD = "word"
    SEGMENT = "segment"


# --- Response Models ---

class TranscriptionWord(BaseModel):
    """Word-level timestamp object for verbose_json"""
    word: str
    start: float
    end: float


class TranscriptionSegment(BaseModel):
    """Segment-level object for verbose_json"""
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionJsonResponse(BaseModel):
    """Simple JSON response format"""
    text: str


class TranscriptionVerboseJsonResponse(BaseModel):
    """Verbose JSON response with segments and optional words"""
    task: Literal["transcribe", "translate"]
    language: str
    duration: float
    text: str
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    words: Optional[List[TranscriptionWord]] = None


# --- Error Models (OpenAI-compatible) ---

class OpenAIErrorDetail(BaseModel):
    """Error detail matching OpenAI format"""
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    """Error response matching OpenAI format"""
    error: OpenAIErrorDetail
