"""HTTP security layer: API keys, rate limiting, docs, prompts, models, CORS, headers.

app.security reads its environment at import time, so every test imports a
fresh copy of the app with the environment it needs. Audio decoding and the
pipeline are faked, so no model is loaded and no GPU is used.
"""

import importlib
import io

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_model_selection import MODEL_ENV, _drop_app_modules

SECURITY_ENV = (
    "API_KEY",
    "API_KEYS",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "ALLOWED_ORIGINS",
    "MAX_PROMPT_CHARS",
    "ALLOW_LOCAL_MODEL_PATHS",
    "ENABLE_DOCS",
    "TRUST_PROXY",
    "ASYNC_JOBS_ENABLED",
    "JOB_DIR",
)
KEY = "test-secret-key"


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    def _make(**env):
        for key in MODEL_ENV + SECURITY_ENV:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("JOB_DIR", str(tmp_path / "jobs"))
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        _drop_app_modules()
        main = importlib.import_module("app.main")
        audio_io = importlib.import_module("app.audio_io")

        async def fake_decode(src, dst):
            np.zeros(16000, dtype=np.int16).tofile(dst)
            return 1.0

        def fake_pipeline(audio, model_name=None, **kwargs):
            return {"language": "en", "segments": [], "word_segments": []}, None

        monkeypatch.setattr(audio_io, "decode_to_pcm", fake_decode)
        monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
        return TestClient(main.app)  # no `with`: the startup preload does not run

    yield _make
    _drop_app_modules()


def _audio():
    return {"audio_file": ("a.wav", io.BytesIO(b"RIFF0000WAVE"), "audio/wav")}


def _openai_file():
    return {"file": ("a.wav", io.BytesIO(b"RIFF0000WAVE"), "audio/wav")}


def test_default_is_open_with_only_the_always_on_headers(make_client):
    client = make_client()
    assert client.post("/asr", files=_audio()).status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/queue-metrics").status_code == 200
    assert client.post("/jobs", files=_audio()).status_code == 202
    r = client.get("/health", headers={"Origin": "https://anywhere.example"})
    assert "WWW-Authenticate" not in r.headers
    assert "Access-Control-Allow-Origin" not in r.headers
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_asr_and_openai_routes_reject_missing_or_wrong_key(make_client):
    client = make_client(API_KEY=KEY)
    for response in (
        client.post("/asr", files=_audio()),
        client.post("/asr", files=_audio(), headers={"X-API-Key": "wrong"}),
        client.post(
            "/v1/audio/transcriptions",
            files=_openai_file(),
            data={"model": "whisper-1"},
        ),
        client.post(
            "/v1/audio/transcriptions",
            files=_openai_file(),
            data={"model": "whisper-1"},
            headers={"Authorization": "Bearer wrong"},
        ),
    ):
        assert response.status_code == 401, response.text
        assert response.json() == {"detail": "Not authenticated"}
        assert response.headers["WWW-Authenticate"] == "Bearer"


def test_bearer_and_x_api_key_are_both_accepted(make_client):
    client = make_client(API_KEY=KEY)
    r = client.post("/asr", files=_audio(), headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 200, r.text
    r = client.post("/asr", files=_audio(), headers={"X-API-Key": KEY})
    assert r.status_code == 200, r.text
    r = client.post(
        "/v1/audio/transcriptions",
        files=_openai_file(),
        data={"model": "whisper-1"},
        headers={"Authorization": f"Bearer {KEY}"},
    )
    assert r.status_code == 200, r.text


def test_health_and_model_list_stay_open_when_a_key_is_set(make_client):
    client = make_client(API_KEY=KEY)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 200


def test_metrics_routes_require_the_key_when_it_is_set(make_client):
    client = make_client(API_KEY=KEY)
    for path in ("/metrics", "/queue-metrics"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"X-API-Key": KEY}).status_code == 200


def test_job_routes_require_the_key_when_it_is_set(make_client):
    client = make_client(API_KEY=KEY)
    assert client.post("/jobs", files=_audio()).status_code == 401
    assert client.get("/jobs/missing-job").status_code == 401
    assert client.get("/jobs/missing-job/result").status_code == 401
    assert client.delete("/jobs/missing-job").status_code == 401

    headers = {"Authorization": f"Bearer {KEY}"}
    r = client.post("/jobs", files=_audio(), headers=headers)
    assert r.status_code == 202, r.text
    assert client.get("/jobs/missing-job", headers=headers).status_code == 404


def test_post_jobs_is_rate_limited_like_asr(make_client):
    client = make_client(RATE_LIMIT_REQUESTS="2")
    for index in range(2):
        payload = {
            "audio_file": (
                f"{index}.wav",
                io.BytesIO(b"RIFF%04dWAVE" % index),
                "audio/wav",
            )
        }
        assert client.post("/jobs", files=payload).status_code == 202
    r = client.post(
        "/jobs",
        files={"audio_file": ("2.wav", io.BytesIO(b"RIFF0002WAVE"), "audio/wav")},
    )
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_rate_limit_returns_429_with_retry_after_on_the_third_request(make_client):
    client = make_client(RATE_LIMIT_REQUESTS="2")
    assert client.post("/asr", files=_audio()).status_code == 200
    assert client.post("/asr", files=_audio()).status_code == 200
    r = client.post("/asr", files=_audio())
    assert r.status_code == 429
    assert r.json() == {"detail": "Rate limit exceeded"}
    assert int(r.headers["Retry-After"]) >= 1


def test_docs_are_hidden_when_disabled(make_client):
    client = make_client(ENABLE_DOCS="false")
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_docs_are_served_by_default(make_client):
    client = make_client()
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_oversized_initial_prompt_is_rejected(make_client):
    client = make_client(MAX_PROMPT_CHARS="10")
    r = client.post("/asr?initial_prompt=" + "x" * 11, files=_audio())
    assert r.status_code == 400
    assert r.json()["detail"] == "initial_prompt exceeds 10 characters"


def test_oversized_hotwords_are_rejected(make_client):
    client = make_client(MAX_PROMPT_CHARS="10")
    r = client.post("/asr?hotwords=" + "x" * 11, files=_audio())
    assert r.status_code == 400
    assert r.json()["detail"] == "hotwords exceeds 10 characters"


def test_jobs_reject_oversized_prompt_and_hotwords(make_client):
    client = make_client(MAX_PROMPT_CHARS="10")
    r = client.post("/jobs?initial_prompt=" + "x" * 11, files=_audio())
    assert r.status_code == 400
    assert r.json()["detail"] == "initial_prompt exceeds 10 characters"
    r = client.post("/jobs?hotwords=" + "x" * 11, files=_audio())
    assert r.status_code == 400
    assert r.json()["detail"] == "hotwords exceeds 10 characters"


def test_prompt_length_helper_accepts_the_limit_and_rejects_one_more(monkeypatch):
    security = importlib.import_module("app.security")

    monkeypatch.setattr(security, "MAX_PROMPT_CHARS", 10)
    security.check_prompt_lengths("x" * 10, "y" * 10)
    security.check_prompt_lengths(None, None)
    with pytest.raises(HTTPException) as exc:
        security.check_prompt_lengths("x" * 11, None)
    assert exc.value.status_code == 400
    assert exc.value.detail == "initial_prompt exceeds 10 characters"
    with pytest.raises(HTTPException) as exc:
        security.check_prompt_lengths(None, "y" * 11)
    assert exc.value.detail == "hotwords exceeds 10 characters"


def test_local_model_paths_helper_follows_the_module_flag(monkeypatch):
    security = importlib.import_module("app.security")

    monkeypatch.setattr(security, "ALLOW_LOCAL_MODEL_PATHS", False)
    assert security.local_model_paths_allowed() is False
    monkeypatch.setattr(security, "ALLOW_LOCAL_MODEL_PATHS", True)
    assert security.local_model_paths_allowed() is True


def test_local_model_path_is_rejected_by_default(make_client):
    client = make_client()
    r = client.post("/asr?model=/srv/models/private-model", files=_audio())
    assert r.status_code == 400
    assert "Unknown model" in r.json()["detail"]


def test_jobs_reject_local_model_path_by_default(make_client):
    client = make_client()
    r = client.post("/jobs?model=/srv/models/private-model", files=_audio())
    assert r.status_code == 400
    assert "Unknown model" in r.json()["detail"]


def test_local_model_path_is_accepted_when_enabled(make_client, tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    client = make_client(ALLOW_LOCAL_MODEL_PATHS="true")
    r = client.post(f"/asr?model={model_dir}", files=_audio())
    assert r.status_code == 200, r.text


def test_generic_500_does_not_leak_internals(make_client, monkeypatch):
    client = make_client()
    main = importlib.import_module("app.main")

    def failing_pipeline(audio, model_name=None, **kwargs):
        raise RuntimeError("secret internals /srv/keys/model.bin")

    monkeypatch.setattr(main, "run_pipeline", failing_pipeline)
    r = client.post("/asr", files=_audio())
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}
    assert "secret internals" not in r.text
    assert "/srv/keys" not in r.text


def test_security_headers_are_present_on_health(make_client):
    client = make_client()
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert "camera=()" in r.headers["Permissions-Policy"]


def test_cors_unlisted_origin_gets_no_allow_origin_header(make_client):
    client = make_client(ALLOWED_ORIGINS="https://app.example.com")
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200
    assert "Access-Control-Allow-Origin" not in r.headers


def test_cors_listed_origin_is_reflected(make_client):
    client = make_client(ALLOWED_ORIGINS="https://app.example.com")
    r = client.get("/health", headers={"Origin": "https://app.example.com"})
    assert r.headers["Access-Control-Allow-Origin"] == "https://app.example.com"
    assert r.headers["Vary"] == "Origin"
