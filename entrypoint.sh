#!/bin/bash
set -e

VERSION=$(python3 -c "from app.version import __version__; print(__version__)")
echo "WhisperX ASR Service v${VERSION}"

# Nothing can be running at container start, so every job dir and upload
# copy left in the temp dir by an earlier run is removed.
python3 -m app.audio_io --all || true

# The service runs under app.watchdog, which forwards signals and exits when
# /health keeps failing so the restart policy starts a clean container
# (WATCHDOG_ENABLED=false turns the health polling off).
if [ "$SERVE_MODE" = "ray" ]; then
    echo "Starting in Ray Serve mode..."

    # Only the Ray object store lives in /dev/shm. Session files and logs
    # go to RAY_TMPDIR (a small tmpfs in the compose files) and objects
    # spill to RAY_SPILL_DIR on disk, so neither can fill shared memory.
    export RAY_TMPDIR="${RAY_TMPDIR:-/ray-tmp}"
    export RAY_SPILL_DIR="${RAY_SPILL_DIR:-/.cache/ray-spill}"
    export RAY_PLASMA_DIR="${RAY_PLASMA_DIR:-/dev/shm}"
    mkdir -p "$RAY_TMPDIR" "$RAY_SPILL_DIR"

    # Cap Ray's object store so it fits within the container's /dev/shm.
    # Default 4 GB leaves headroom in the 8g shm.
    export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-4000000000}"

    # Rotate Ray's log files (default 512 MB x 5 per file) at 50 MB x 1.
    export RAY_ROTATION_MAX_BYTES="${RAY_ROTATION_MAX_BYTES:-52428800}"
    export RAY_ROTATION_BACKUP_COUNT="${RAY_ROTATION_BACKUP_COUNT:-1}"

    # Silence the FutureWarning about accelerator env var override
    export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

    # The Serve HTTP address is passed to serve.start() before serve.run().
    # RAY_SERVE_DEFAULT_HTTP_PORT is ignored by current Ray releases (checked
    # on 2.59), and the port then falls back to 8000.
    #
    # The dashboard listens on the container's loopback interface unless
    # RAY_DASHBOARD_HOST is set (0.0.0.0 makes the published port reachable).
    # It stays off the network by default, as in earlier releases, because
    # jobs submitted through it run as code in the container (see README
    # "Ray Dashboard Access").
    export RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-127.0.0.1}"
    exec python3 -m app.watchdog -- python3 -c "
import os
import ray
from ray import serve

ray.init(
    dashboard_host=os.environ['RAY_DASHBOARD_HOST'],
    object_store_memory=int(float(os.environ['RAY_OBJECT_STORE_MEMORY'])),
    _plasma_directory=os.environ['RAY_PLASMA_DIR'],
    object_spilling_directory=os.environ['RAY_SPILL_DIR'],
)
serve.start(http_options={'host': '0.0.0.0', 'port': 9000})

from app.serve_app import app
serve.run(app, blocking=True)
"
else
    echo "Starting in simple mode (uvicorn)..."
    exec python3 -m app.watchdog -- python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9000
fi
