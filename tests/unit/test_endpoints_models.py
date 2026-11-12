"""The /asr and OpenAI endpoints resolve models the same way.

Audio decoding and the pipeline are faked, so no model is loaded.
"""

import io
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from test_model_selection import MODEL_ENV, _drop_app_modules


@pytest.fixture
def client_for(monkeypatch):
    def _make(**env):
        for key in MODEL_ENV:
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        _drop_app_modules()
        import app.main as main
        import app.openai_compat as oc
        import app.audio_io as audio_io

        async def fake_decode(src, dst):
            np.zeros(16000, dtype=np.int16).tofile(dst)
            return 1.0

        monkeypatch.setattr(audio_io, "decode_to_pcm", fake_decode)
        seen = {}

        def fake_pipeline(audio, model_name=None, **kwargs):
            seen["asr"] = model_name
            return {"language": "en", "segments": [], "word_segments": []}, None

        def fake_transcribe_and_align(audio, model, *args, **kwargs):
            seen["openai"] = model
            return {"language": "en", "segments": [{"text": "hi", "start": 0.0, "end": 1.0}]}

        monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
        monkeypatch.setattr(oc, "_run_transcribe_and_align", fake_transcribe_and_align)
        return TestClient(main.app), seen   # no `with`: the startup preload does not run
    yield _make
    _drop_app_modules()


def _audio():
    return {"audio_file": ("a.wav", io.BytesIO(b"RIFF0000WAVE"), "audio/wav")}


def test_asr_without_a_model_uses_the_default_even_with_an_empty_preload(client_for):
    client, seen = client_for(PRELOAD_MODEL="")
    r = client.post("/asr", files=_audio())
    assert r.status_code == 200, r.text
    assert seen["asr"] == "large-v3"

    r = client.post("/asr?model=", files=_audio())
    assert r.status_code == 200 and seen["asr"] == "large-v3"


def test_asr_resolves_aliases_and_rejects_unknown_models(client_for):
    client, seen = client_for(PRELOAD_MODEL="large-v3")
    assert client.post("/asr?model=whisper-medium", files=_audio()).status_code == 200
    assert seen["asr"] == "medium"
    r = client.post("/asr?model=large-v4", files=_audio())
    assert r.status_code == 400
    assert "Unknown model" in r.json()["detail"]


@pytest.mark.parametrize("model,expected", [
    ("turbo", "turbo"),
    ("distil-large-v3.5", "distil-large-v3.5"),
    ("medium.en", "medium.en"),
    ("whisper-1", "large-v3"),
])
def test_openai_endpoint_accepts_every_listed_model(client_for, model, expected):
    client, seen = client_for(PRELOAD_MODEL="")
    r = client.post("/v1/audio/transcriptions", data={"model": model},
                    files={"file": ("a.wav", io.BytesIO(b"RIFF0000WAVE"), "audio/wav")})
    assert r.status_code == 200, r.text
    assert seen["openai"] == expected


def test_openai_endpoint_rejects_unknown_models(client_for):
    client, _ = client_for()
    r = client.post("/v1/audio/transcriptions", data={"model": "large-v4"},
                    files={"file": ("a.wav", io.BytesIO(b"RIFF0000WAVE"), "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "model"


def test_models_list_matches_what_is_accepted(client_for):
    client, _ = client_for(ALLOWED_MODELS="turbo")
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["whisper-1", "large-v3", "turbo"]
