"""Print, as JSON, how the service resolves models in the current environment.

Run in a fresh interpreter (tests/defaults/test_documented_defaults.py does
this) with tests/e2e/stubs first on PYTHONPATH and the app root after it.
Written against the public names of app.pipeline, with fallbacks, so it can
also describe older versions of the service (the regression check in
tests/defaults/README.md).
"""

import json
import os
import sys

result = {"import_error": None}
try:
    from app import pipeline as p
    import app.main  # noqa: F401  (module-level settings of the web app)
except Exception as exc:  # the defaults must never crash the import
    result["import_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result))
    sys.exit(0)

from faster_whisper import available_models

canonical = list(available_models())


def loadable(name):
    if not isinstance(name, str) or not name.strip():
        return False
    return name in canonical or "/" in name or os.path.isdir(name)


resolved = {}
for label, value in (("none", None), ("empty", ""), ("whitespace", "   "),
                     ("whisper-1", "whisper-1"), ("large-v3", "large-v3"),
                     ("alias", "whisper-large-v3")):
    try:
        name = p.resolve_model_name(value)
        resolved[label] = {"name": name, "loadable": loadable(name)}
    except Exception as exc:
        resolved[label] = {"error": f"{type(exc).__name__}: {exc}", "loadable": False}

preloaded = "n/a"
if hasattr(p, "preload_whisper_model"):
    calls = []
    p.load_whisper_model = lambda name, *a, **k: calls.append(name)
    returned = p.preload_whisper_model("probe")
    preloaded = {"loaded": calls, "returned": returned}

robustness = "n/a"
try:
    from app import admission, audio_io, health
    from app.watchdog import Watchdog

    dog = Watchdog(["true"])
    robustness = {
        "admission": admission.AdmissionController.from_env().settings(),
        "job_dir": audio_io.JOB_DIR,
        "job_ttl_seconds": audio_io.JOB_TTL_SECONDS,
        "health_min_shm_free_pct": health.HEALTH_MIN_SHM_FREE_PCT,
        "health_min_disk_free_mb": health.HEALTH_MIN_DISK_FREE_MB,
        "health_stage_timeout_seconds": health.HEALTH_STAGE_TIMEOUT_SECONDS,
        "watchdog": {"enabled": dog.enabled, "interval": dog.interval, "failures": dog.max_failures,
                     "shm_max_used": dog.shm_max_used, "shm_failures": dog.shm_max_failures},
        "diarize_on_error": getattr(p, "DIARIZE_ON_ERROR", None),
    }
except ImportError:
    pass

result.update({
    "robustness": robustness,
    "default_model": getattr(p, "DEFAULT_MODEL", None),
    "preload_model": getattr(p, "PRELOAD_MODEL", os.getenv("PRELOAD_MODEL")),
    "asr_backend": getattr(p, "ASR_BACKEND", None),
    "resolved": resolved,
    "preload": preloaded,
})
print(json.dumps(result))
