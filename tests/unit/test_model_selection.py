"""Model defaults, name resolution, preloading and the loaded-model cap.

Regression for Speakr #409: with PRELOAD_MODEL set but empty (the shipped
compose files forward an unset variable as an empty string), every request
that named no model failed with "Invalid model size ''".

app.pipeline reads its settings at import time, so each test imports a fresh
copy with the environment it needs.
"""

import sys
import time

import numpy as np
import pytest

MODEL_ENV = (
    "PRELOAD_MODEL", "DEFAULT_MODEL", "OPENAI_WHISPER1_MODEL",
    "ALLOWED_MODELS", "MAX_LOADED_MODELS", "ASR_BACKEND",
    "MODEL_KEEP_ALIVE_SECONDS",
)


# Prometheus metrics register in a process-wide registry, so app.metrics is
# kept across re-imports; every other app module is imported afresh.
_KEEP = {"app", "app.metrics"}


def _drop_app_modules():
    for name in [m for m in sys.modules if (m == "app" or m.startswith("app.")) and m not in _KEEP]:
        sys.modules.pop(name)


@pytest.fixture
def fresh(monkeypatch):
    """Import app.pipeline (or another app module) with the given env."""
    def _load(module="app.pipeline", **env):
        for key in MODEL_ENV:
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        _drop_app_modules()
        __import__(module)
        return sys.modules[module]
    yield _load
    _drop_app_modules()


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.options = type("Options", (), {"hotwords": None, "initial_prompt": None})()

    def transcribe(self, audio, **kwargs):
        return {"language": "en", "segments": [], "model": self.name}


@pytest.fixture
def fake_loader(monkeypatch):
    """Replace whisperx.load_model; returns the list of names loaded."""
    import whisperx
    loaded = []

    def _load_model(name, **kwargs):
        loaded.append(name)
        return FakeModel(name)

    monkeypatch.setattr(whisperx, "load_model", _load_model)
    return loaded


# ---------------------------------------------------------------- defaults

@pytest.mark.parametrize("env", [{}, {"PRELOAD_MODEL": ""}, {"PRELOAD_MODEL": "   "}])
def test_unset_or_empty_preload_model_keeps_a_usable_default(fresh, env):
    p = fresh(**env)
    assert p.PRELOAD_MODEL is None
    assert p.DEFAULT_MODEL == "large-v3"
    assert p.resolve_model_name(None) == "large-v3"
    assert p.resolve_model_name("") == "large-v3"


def test_preload_model_is_still_the_default_when_set(fresh):
    p = fresh(PRELOAD_MODEL=" medium ")
    assert p.PRELOAD_MODEL == "medium"
    assert p.DEFAULT_MODEL == "medium"


def test_default_model_takes_precedence_over_preload(fresh):
    p = fresh(PRELOAD_MODEL="large-v3", DEFAULT_MODEL="turbo")
    assert p.DEFAULT_MODEL == "turbo"
    assert p.resolve_model_name("  ") == "turbo"


def test_default_given_as_an_alias_is_resolved(fresh):
    assert fresh(PRELOAD_MODEL="whisper-large-v3").DEFAULT_MODEL == "large-v3"
    assert fresh(DEFAULT_MODEL="whisper-turbo").DEFAULT_MODEL == "turbo"


def test_an_unusable_default_falls_back_with_a_warning(fresh, caplog):
    caplog.set_level("WARNING")
    p = fresh(DEFAULT_MODEL="large-v4")
    assert p.DEFAULT_MODEL == "large-v3"
    assert "large-v4" in caplog.text


# ---------------------------------------------------------------- resolution

def test_resolve_model_name(fresh, tmp_path):
    p = fresh(PRELOAD_MODEL="large-v3")
    assert p.resolve_model_name("large-v3") == "large-v3"
    assert p.resolve_model_name(" turbo ") == "turbo"
    assert p.resolve_model_name("distil-large-v3.5") == "distil-large-v3.5"
    assert p.resolve_model_name("whisper-large-v2") == "large-v2"
    assert p.resolve_model_name("whisper-medium.en") == "medium.en"
    assert p.resolve_model_name("whisper-1") == "large-v3"
    assert p.resolve_model_name("Systran/faster-whisper-small") == "Systran/faster-whisper-small"
    assert p.resolve_model_name(str(tmp_path), allow_local_paths=True) == str(tmp_path)
    for bad in ("large-v4", "whisper-2", "a/b/c", "not a model"):
        with pytest.raises(p.InvalidModelError) as err:
            p.resolve_model_name(bad)
        assert "large-v3" in str(err.value)
    assert issubclass(p.InvalidModelError, ValueError)


def test_whisper_1_follows_its_setting_or_the_default(fresh):
    assert fresh(PRELOAD_MODEL="medium").resolve_model_name("whisper-1") == "medium"
    assert fresh(PRELOAD_MODEL="medium", OPENAI_WHISPER1_MODEL="small").resolve_model_name("whisper-1") == "small"


@pytest.mark.parametrize("env", [
    {"OPENAI_WHISPER1_MODEL": "whisper-1"},
    {"DEFAULT_MODEL": "whisper-1"},
    {"PRELOAD_MODEL": "whisper-1"},
])
def test_whisper_1_never_points_at_itself(fresh, env):
    p = fresh(**env)
    assert p.resolve_model_name("whisper-1") == "large-v3"
    assert p.DEFAULT_MODEL == "large-v3"


def test_allowed_models(fresh):
    p = fresh(PRELOAD_MODEL="large-v3", ALLOWED_MODELS="turbo, whisper-large-v2 ,")
    assert p.ALLOWED_MODELS == ["turbo", "large-v2"]
    assert p.resolve_model_name("turbo") == "turbo"
    assert p.resolve_model_name("whisper-large-v2") == "large-v2"
    assert p.resolve_model_name(None) == "large-v3"  # the default is always allowed
    with pytest.raises(p.InvalidModelError, match="not allowed"):
        p.resolve_model_name("medium")
    assert p.list_available_models() == ["large-v3", "turbo", "large-v2"]


def test_all_canonical_models_are_listed_without_a_restriction(fresh):
    p = fresh()
    assert p.list_available_models() == p.get_canonical_models()
    assert "turbo" in p.list_available_models()


# ---------------------------------------------------------------- preload

def test_preload(fresh, monkeypatch):
    p = fresh()
    calls = []
    monkeypatch.setattr(p, "load_whisper_model", lambda name: calls.append(name))
    assert p.preload_whisper_model() is None and calls == []

    p = fresh(PRELOAD_MODEL="whisper-large-v3")
    calls = []
    monkeypatch.setattr(p, "load_whisper_model", lambda name: calls.append(name))
    assert p.preload_whisper_model() == "large-v3"
    assert calls == ["large-v3"]  # same cache key a request uses


@pytest.mark.parametrize("backend", ["qwen3", "external"])
def test_preload_is_skipped_on_backends_without_whisper(fresh, monkeypatch, backend):
    p = fresh(PRELOAD_MODEL="large-v3", ASR_BACKEND=backend)
    calls = []
    monkeypatch.setattr(p, "load_whisper_model", lambda name: calls.append(name))
    assert p.preload_whisper_model() is None
    assert calls == []


def test_a_failed_preload_is_logged_not_raised(fresh, monkeypatch, caplog):
    p = fresh(PRELOAD_MODEL="large-v3")

    def _boom(name):
        raise RuntimeError("no disk")

    monkeypatch.setattr(p, "load_whisper_model", _boom)
    caplog.set_level("ERROR")
    assert p.preload_whisper_model("WhisperDeployment") is None
    assert "no disk" in caplog.text


# ---------------------------------------------------------------- loading and the cap

def test_transcribe_without_a_model_uses_the_default(fresh, fake_loader):
    p = fresh(PRELOAD_MODEL="")
    result = p.transcribe(np.zeros(16000, dtype=np.float32), model_name=None)
    assert fake_loader == ["large-v3"]
    assert result["model"] == "large-v3"
    assert p._whisper_models_in_use == {}


def test_a_loaded_model_is_reused(fresh, fake_loader):
    p = fresh()
    first = p.load_whisper_model("small")
    assert p.load_whisper_model("small") is first
    assert fake_loader == ["small"]


def test_max_loaded_models_unloads_the_least_recently_used(fresh, fake_loader):
    p = fresh(MAX_LOADED_MODELS="2")
    p.load_whisper_model("tiny")
    time.sleep(0.01)
    p.load_whisper_model("base")
    time.sleep(0.01)
    p.load_whisper_model("tiny")  # tiny is now the most recently used
    p.load_whisper_model("small")
    assert sorted(p._whisper_models) == ["small", "tiny"]


def test_a_model_in_use_is_never_unloaded_by_the_cap(fresh, fake_loader, caplog):
    p = fresh(MAX_LOADED_MODELS="1")
    with p.use_whisper_model("tiny"):
        caplog.set_level("WARNING")
        p.load_whisper_model("base")
        assert "tiny" in p._whisper_models
        assert "every loaded model is in use" in caplog.text
    p.load_whisper_model("small")  # tiny is idle again: now both it and base may go
    assert list(p._whisper_models) == ["small"]


def test_the_idle_sweep_skips_models_in_use(fresh, fake_loader):
    p = fresh(MODEL_KEEP_ALIVE_SECONDS="1")
    p.load_whisper_model("tiny")
    p.load_whisper_model("base")
    old = time.time() - 60
    p._whisper_models_last_used["tiny"] = old
    p._whisper_models_last_used["base"] = old
    p._whisper_models_in_use["tiny"] = 1
    p._evict_from_cache(
        p._whisper_models, p._whisper_models_last_used, "model", time.time(),
        in_use=p._whisper_models_in_use, lock=p._whisper_state_lock,
    )
    assert list(p._whisper_models) == ["tiny"]


@pytest.mark.parametrize("value", ["", "abc", "-3"])
def test_bad_integer_settings_fall_back(fresh, value):
    p = fresh(MAX_LOADED_MODELS=value, MODEL_KEEP_ALIVE_SECONDS=value)
    assert p.MAX_LOADED_MODELS == 0
    assert p.MODEL_KEEP_ALIVE_SECONDS in (0, -3)
