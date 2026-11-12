"""A failed diarization is reported, not dropped silently; an out-of-memory
error is retried once."""

import numpy as np
import pytest

from test_model_selection import fresh  # noqa: F401  (fixture)

RESULT = {"language": "en", "segments": [{"text": "hi", "start": 0.0, "end": 1.0}]}


class _Turns(list):
    pass


def _pipeline_with(monkeypatch, p, behaviours):
    calls = []

    def model(audio, **kwargs):
        calls.append(kwargs)
        action = behaviours[min(len(calls), len(behaviours)) - 1]
        if isinstance(action, BaseException):
            raise action
        return _Turns([{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}])

    monkeypatch.setattr(p, "load_diarize_pipeline", lambda: model)
    return calls


def test_success_is_marked_ok(fresh, monkeypatch):
    p = fresh(HF_TOKEN="hf_test")
    _pipeline_with(monkeypatch, p, [None])
    result, _ = p.diarize(np.zeros(16000, np.float32), dict(RESULT))
    assert result["_diarization"] == {"status": "ok"}
    assert result["segments"][0]["speaker"] == "SPEAKER_00"


def test_a_failure_is_flagged_by_default(fresh, monkeypatch):
    p = fresh(HF_TOKEN="hf_test")
    calls = _pipeline_with(monkeypatch, p, [ValueError("bad input")])
    result, embeddings = p.diarize(np.zeros(16000, np.float32), dict(RESULT))
    assert len(calls) == 1                       # not an OOM: no retry
    assert result["_diarization"] == {"status": "failed", "error": "bad input"}
    assert embeddings is None
    assert "speaker" not in result["segments"][0]


def test_out_of_memory_is_retried_once(fresh, monkeypatch):
    p = fresh(HF_TOKEN="hf_test")
    calls = _pipeline_with(monkeypatch, p, [RuntimeError("CUDA out of memory. Tried to allocate"), None])
    result, _ = p.diarize(np.zeros(16000, np.float32), dict(RESULT))
    assert len(calls) == 2
    assert result["_diarization"]["status"] == "ok"


def test_fail_mode_raises(fresh, monkeypatch):
    p = fresh(HF_TOKEN="hf_test", DIARIZE_ON_ERROR="fail")
    _pipeline_with(monkeypatch, p, [RuntimeError("CUDA out of memory")] * 2)
    with pytest.raises(p.DiarizationError):
        p.diarize(np.zeros(16000, np.float32), dict(RESULT))


def test_without_a_token_diarization_is_unavailable(fresh, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    p = fresh()
    result, _ = p.diarize(np.zeros(16000, np.float32), dict(RESULT))
    assert result["_diarization"]["status"] == "unavailable"
