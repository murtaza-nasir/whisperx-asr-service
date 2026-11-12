"""
In-container watchdog: run the service as a child process and exit when it
stays unhealthy, so the container's restart policy brings up a clean one.

Docker marks a container "unhealthy" but never restarts it for that. This
wrapper polls /health; after WATCHDOG_FAILURES consecutive failed checks
(non-200, timeout or no connection), or when shared memory stays nearly full
for WATCHDOG_SHM_FAILURES checks, it stops the whole process group and exits
with status 1. Failures only count after the service has answered healthy
once, so a slow first start (model downloads) never triggers it.

SIGTERM and SIGINT are forwarded to the child, so `docker stop` works as
before. If the child exits by itself, the watchdog exits with its status.

    python3 -m app.watchdog -- python3 -m uvicorn app.main:app ...

Settings (env): WATCHDOG_ENABLED (default true), WATCHDOG_URL,
WATCHDOG_INTERVAL_SECONDS (30), WATCHDOG_TIMEOUT_SECONDS (10),
WATCHDOG_FAILURES (6), WATCHDOG_SHM_PATH (/dev/shm),
WATCHDOG_SHM_MAX_USED_PCT (98), WATCHDOG_SHM_FAILURES (4),
WATCHDOG_STOP_TIMEOUT_SECONDS (30).
"""

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional


def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        _log(f"{name}={value!r} is not a number; using {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


def _log(message: str) -> None:
    print(f"[watchdog] {message}", file=sys.stderr, flush=True)


def check_health(url: str, timeout: float) -> Optional[str]:
    """None when /health answers 200, otherwise a short reason."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return None
            return f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:  # connection refused, timeout, reset
        return f"{type(e).__name__}: {e}"


def shm_used_pct(path: str) -> Optional[float]:
    try:
        st = os.statvfs(path)
    except OSError:
        return None
    if st.f_blocks <= 0:
        return None
    return 100.0 * (st.f_blocks - st.f_bavail) / st.f_blocks


class Watchdog:
    def __init__(self, cmd: List[str]):
        self.cmd = cmd
        self.enabled = _env_bool("WATCHDOG_ENABLED", True)
        self.url = _env("WATCHDOG_URL") or "http://127.0.0.1:9000/health"
        self.interval = max(0.1, _env_float("WATCHDOG_INTERVAL_SECONDS", 30))
        self.timeout = max(0.1, _env_float("WATCHDOG_TIMEOUT_SECONDS", 10))
        self.max_failures = max(1, int(_env_float("WATCHDOG_FAILURES", 6)))
        self.shm_path = _env("WATCHDOG_SHM_PATH") or "/dev/shm"
        self.shm_max_used = _env_float("WATCHDOG_SHM_MAX_USED_PCT", 98)
        self.shm_max_failures = max(1, int(_env_float("WATCHDOG_SHM_FAILURES", 4)))
        self.stop_timeout = max(1.0, _env_float("WATCHDOG_STOP_TIMEOUT_SECONDS", 30))
        self.child: Optional[subprocess.Popen] = None
        self._stopping = False

    # ------------------------------------------------------------------
    def _forward(self, signum, _frame):
        self._stopping = True
        if self.child and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signum)
            except ProcessLookupError:
                pass

    def _poll(self) -> Optional[int]:
        """Child exit status, or None while it runs. As PID 1 the watchdog
        also reaps orphaned grandchildren (Ray helpers) so no zombies pile up."""
        if os.getpid() == 1 and self.child.returncode is None:
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                if pid == self.child.pid:
                    self.child.returncode = os.waitstatus_to_exitcode(status)
        return self.child.poll()

    def stop_child(self) -> None:
        if not self.child or self._poll() is not None:
            return
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self.stop_timeout
        while time.monotonic() < deadline:
            if self._poll() is not None:
                break
            time.sleep(0.2)
        # Ray starts helper processes in the same group; kill whatever is left.
        try:
            os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while self._poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._forward)
        signal.signal(signal.SIGINT, self._forward)
        self.child = subprocess.Popen(self.cmd, start_new_session=True)
        if not self.enabled:
            _log("disabled (WATCHDOG_ENABLED=false); only forwarding signals")
        else:
            _log(f"watching {self.url} every {self.interval:g}s; exit after "
                 f"{self.max_failures} failed checks or {self.shm_max_failures} checks "
                 f"with {self.shm_path} over {self.shm_max_used:g}% used")

        armed = False
        failures = 0
        shm_failures = 0
        while True:
            deadline = time.monotonic() + self.interval
            while time.monotonic() < deadline:
                code = self._poll()
                if code is not None:
                    if not self._stopping:
                        _log(f"service exited with status {code}")
                    return code if code >= 0 else 128 - code
                time.sleep(min(0.5, self.interval))
            if not self.enabled or self._stopping:
                continue

            reason = check_health(self.url, self.timeout)
            if reason is None:
                if not armed:
                    _log("service healthy; watchdog armed")
                armed = True
                failures = 0
            elif armed:
                failures += 1
                _log(f"health check failed ({failures}/{self.max_failures}): {reason}")

            used = shm_used_pct(self.shm_path)
            if used is not None and self.shm_max_used > 0 and used >= self.shm_max_used:
                shm_failures += 1
                _log(f"{self.shm_path} {used:.1f}% used "
                     f"({shm_failures}/{self.shm_max_failures})")
            else:
                shm_failures = 0

            if failures >= self.max_failures or shm_failures >= self.shm_max_failures:
                why = (f"{failures} consecutive failed health checks"
                       if failures >= self.max_failures
                       else f"{self.shm_path} over {self.shm_max_used:g}% used")
                _log(f"{why}; stopping the service so the container restarts")
                self.stop_child()
                return 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("usage: python3 -m app.watchdog -- COMMAND [ARGS...]", file=sys.stderr)
        return 2
    return Watchdog(argv).run()


if __name__ == "__main__":
    sys.exit(main())
