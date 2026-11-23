"""CLEAR_GPU_AFTER_TRANSCRIBE (issue #35).

The GPU cache is cleared after every Whisper transcription unless the
setting turns it off. Unset or empty must keep that behaviour, because
compose forwards unset variables as empty strings.
"""

import numpy as np
import pytest

from test_model_selection import FakeModel, fresh  # noqa: F401  (fixture)


@pytest.fixture
def pipeline(fresh, monkeypatch):  # noqa: F811
    """Load app.pipeline with the given env and count clear_gpu_memory() calls."""
    def _load(**env):
        monkeypatch.delenv("CLEAR_GPU_AFTER_TRANSCRIBE", raising=False)
        p = fresh(**env)
        import whisperx
        monkeypatch.setattr(whisperx, "load_model", lambda name, **kwargs: FakeModel(name))
        calls = []
        monkeypatch.setattr(p, "clear_gpu_memory", lambda: calls.append(1))
        return p, calls
    return _load


def _transcribe(p):
    p.transcribe(np.zeros(16000, dtype=np.float32), model_name="tiny")


@pytest.mark.parametrize("value", [None, "", "  ", "true", "1", "yes", "TRUE"])
def test_cleanup_runs_by_default(pipeline, value):
    p, calls = pipeline() if value is None else pipeline(CLEAR_GPU_AFTER_TRANSCRIBE=value)
    assert p.CLEAR_GPU_AFTER_TRANSCRIBE is True
    _transcribe(p)
    assert len(calls) == 1


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " False "])
def test_cleanup_can_be_turned_off(pipeline, value):
    p, calls = pipeline(CLEAR_GPU_AFTER_TRANSCRIBE=value)
    assert p.CLEAR_GPU_AFTER_TRANSCRIBE is False
    _transcribe(p)
    assert calls == []
