"""The service works with the configuration its documentation tells users to use.

Each case builds the container environment exactly as a user's setup does
(docker-compose.yml plus a .env file, see tests/e2e/compose_env.py), imports
the service in a fresh interpreter on the stand-in ML packages, and checks
that a request naming no model, an empty or blank model, or `whisper-1`
resolves to a model the engine can load, and that preloading does what the
docs say. Speakr #409 was a documented setup (PRELOAD_MODEL missing from .env)
in which every request without a model failed.
"""

import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "e2e"))
from compose_env import compose_environment, parse_env_file  # noqa: E402

MINIMAL_ENV = os.path.join(HERE, "minimal.env")
STUBS = os.path.join(REPO_ROOT, "tests", "e2e", "stubs")
# The regression check points this at an older checkout of app/.
APP_ROOT = os.environ.get("WX_APP_ROOT", REPO_ROOT)


def _env_file(tmp_path, drop=(), extra=None):
    """minimal.env with some keys removed and others added, as a user's .env."""
    lines = []
    with open(MINIMAL_ENV, encoding="utf-8") as handle:
        for line in handle:
            key = line.split("=", 1)[0].strip()
            if key in drop:
                continue
            lines.append(line.rstrip("\n"))
    for key, value in (extra or {}).items():
        lines.append(f"{key}={value}")
    path = tmp_path / ".env"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _probe(env_file, overrides=None):
    container = compose_environment(env_file=env_file)
    container.update(overrides or {})
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR") if k in os.environ}
    env.update(container)
    env["PYTHONPATH"] = os.pathsep.join([STUBS, APP_ROOT])
    proc = subprocess.run([sys.executable, os.path.join(HERE, "probe_settings.py")],
                          capture_output=True, text=True, env=env, cwd=APP_ROOT, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1]), container


def _assert_requests_without_a_model_work(report):
    assert report["import_error"] is None, report["import_error"]
    for label in ("none", "empty", "whitespace", "whisper-1"):
        entry = report["resolved"][label]
        assert entry["loadable"], f"a request with model={label} resolves to {entry}"


# ---------------------------------------------------------------------------
# The documented setup and the documented variations of it
# ---------------------------------------------------------------------------
def test_the_setup_guide_env_block_and_minimal_env_agree():
    with open(os.path.join(REPO_ROOT, "SETUP_GUIDE.md"), encoding="utf-8") as handle:
        guide = handle.read()
    match = re.search(r"Update `\.env`:\s*```bash\n(.*?)```", guide, re.S)
    assert match, "SETUP_GUIDE.md no longer has the 'Update `.env`:' block"
    block = tmp = os.path.join(HERE, ".guide-block.env")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(match.group(1))
        assert parse_env_file(block) == parse_env_file(MINIMAL_ENV), (
            "tests/defaults/minimal.env must match the .env block in SETUP_GUIDE.md")
    finally:
        os.remove(tmp)


def test_documented_setup(tmp_path):
    report, _ = _probe(MINIMAL_ENV)
    _assert_requests_without_a_model_work(report)
    assert report["resolved"]["none"]["name"] == "large-v3"
    assert report["preload"]["loaded"] == ["large-v3"]


@pytest.mark.parametrize("case", ["missing", "empty", "blank"])
def test_no_preload_model_still_leaves_a_default(tmp_path, case):
    """#409: the key left out of .env (compose forwards ''), set empty as the
    docs allow, or set to whitespace."""
    if case == "missing":
        env_file = _env_file(tmp_path, drop={"PRELOAD_MODEL"})
    elif case == "empty":
        env_file = _env_file(tmp_path, drop={"PRELOAD_MODEL"}, extra={"PRELOAD_MODEL": ""})
    else:
        env_file = _env_file(tmp_path, drop={"PRELOAD_MODEL"}, extra={"PRELOAD_MODEL": '"   "'})
    report, container = _probe(env_file)
    assert container["PRELOAD_MODEL"].strip() == ""
    _assert_requests_without_a_model_work(report)
    assert report["resolved"]["none"]["name"] == "large-v3"
    assert report["preload"]["loaded"] == []


def test_an_alias_as_preload_model(tmp_path):
    report, _ = _probe(_env_file(tmp_path, drop={"PRELOAD_MODEL"}, extra={"PRELOAD_MODEL": "whisper-large-v3"}))
    _assert_requests_without_a_model_work(report)
    assert report["preload"]["loaded"] == ["large-v3"]
    assert report["resolved"]["none"]["name"] == "large-v3"


def test_another_preload_model_is_also_the_default(tmp_path):
    report, _ = _probe(_env_file(tmp_path, drop={"PRELOAD_MODEL"}, extra={"PRELOAD_MODEL": "medium"}))
    _assert_requests_without_a_model_work(report)
    assert report["resolved"]["none"]["name"] == "medium"
    assert report["preload"]["loaded"] == ["medium"]


def test_default_model_without_preloading(tmp_path):
    report, _ = _probe(_env_file(tmp_path, drop={"PRELOAD_MODEL"}, extra={"DEFAULT_MODEL": "small"}))
    _assert_requests_without_a_model_work(report)
    assert report["resolved"]["none"]["name"] == "small"
    assert report["preload"]["loaded"] == []


def test_openai_whisper1_model(tmp_path):
    report, _ = _probe(_env_file(tmp_path, extra={"OPENAI_WHISPER1_MODEL": "turbo"}))
    _assert_requests_without_a_model_work(report)
    assert report["resolved"]["whisper-1"]["name"] == "turbo"


def test_a_misspelt_preload_model_does_not_break_requests(tmp_path):
    report, _ = _probe(_env_file(tmp_path, drop={"PRELOAD_MODEL"}, extra={"PRELOAD_MODEL": "large-v4"}))
    _assert_requests_without_a_model_work(report)
    assert report["resolved"]["none"]["name"] == "large-v3"


@pytest.mark.parametrize("backend", ["qwen3", "external"])
def test_backends_without_whisper_skip_the_preload(tmp_path, backend):
    report, _ = _probe(_env_file(tmp_path, extra={"ASR_BACKEND": backend}))
    _assert_requests_without_a_model_work(report)
    assert report["preload"]["loaded"] == []


def test_bad_integer_settings_do_not_crash_the_import(tmp_path):
    """`docker run --env-file` keeps inline comments and quotes in the value."""
    report, _ = _probe(MINIMAL_ENV, overrides={
        "BATCH_SIZE": "16  # per GPU",
        "MAX_FILE_SIZE_MB": "abc",
        "MAX_LOADED_MODELS": "two",
        "MODEL_KEEP_ALIVE_SECONDS": '"300"',
        "GPU_CONCURRENCY": "",
        "MAX_QUEUE_SIZE": "lots",
    })
    _assert_requests_without_a_model_work(report)


def test_long_file_and_health_settings_default_to_off_or_generous(tmp_path):
    """Admission limits are off, health thresholds and the watchdog use their
    documented defaults, with the documented setup and with every new key
    left out of the compose environment."""
    expected = {
        "admission": {"max_audio_seconds": 0, "max_inflight_audio_seconds": 0,
                      "max_queued_audio_seconds": 0, "max_queued_requests": 0,
                      "queue_wait_timeout_seconds": 0},
        "job_dir": "/tmp/whisperx-jobs",
        "job_ttl_seconds": 86400,
        "health_min_shm_free_pct": 5.0,
        "health_min_disk_free_mb": 500.0,
        "health_stage_timeout_seconds": 5.0,
        "watchdog": {"enabled": True, "interval": 30.0, "failures": 6,
                     "shm_max_used": 98.0, "shm_failures": 4},
        "diarize_on_error": "flag",
    }
    report, container = _probe(MINIMAL_ENV)
    assert report["robustness"] == expected
    for key in ("MAX_AUDIO_SECONDS", "MAX_INFLIGHT_AUDIO_SECONDS", "JOB_DIR",
                "WATCHDOG_ENABLED", "HEALTH_MIN_SHM_FREE_PCT", "RAY_TMPDIR"):
        assert key in container, f"docker-compose.yml does not forward {key}"

    blank = {key: "" for key in container if key.startswith(
        ("MAX_AUDIO", "MAX_INFLIGHT", "MAX_QUEUED", "QUEUE_WAIT", "JOB_", "HEALTH_", "WATCHDOG_",
         "DIARIZE_ON_ERROR"))}
    report, _ = _probe(MINIMAL_ENV, overrides=blank)
    assert report["robustness"] == expected
