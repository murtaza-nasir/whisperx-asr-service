"""The in-container watchdog: exits after sustained health failures, stops
the whole process group, forwards signals, and passes the child's status on.

A small HTTP server stands in for the service; a file next to it sets the
status code /health returns.
"""

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAKE_SERVICE = textwrap.dedent('''
    import http.server, os, subprocess, sys
    port, status_file, pid_file = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    # A grandchild in the same process group, like Ray's helper processes.
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(pid_file, "w") as f:
        f.write(f"{os.getpid()} {helper.pid}")

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                code = int(open(status_file).read().strip())
            except Exception:
                code = 200
            self.send_response(code)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); treat it as gone.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


@pytest.fixture
def service(tmp_path):
    port = _free_port()
    script = tmp_path / "fake_service.py"
    script.write_text(FAKE_SERVICE)
    status = tmp_path / "status"
    status.write_text("200")
    pids = tmp_path / "pids"

    def start(**env):
        child_env = dict(os.environ)
        child_env.update({
            "PYTHONPATH": REPO_ROOT,
            "WATCHDOG_URL": f"http://127.0.0.1:{port}/health",
            "WATCHDOG_INTERVAL_SECONDS": "0.2",
            "WATCHDOG_TIMEOUT_SECONDS": "1",
            "WATCHDOG_FAILURES": "3",
            "WATCHDOG_SHM_PATH": str(tmp_path),
            "WATCHDOG_SHM_MAX_USED_PCT": "100.1",
            "WATCHDOG_STOP_TIMEOUT_SECONDS": "3",
        })
        child_env.update(env)
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.watchdog", "--", sys.executable, str(script),
             str(port), str(status), str(pids)],
            env=child_env, stderr=subprocess.PIPE, text=True,
        )
        deadline = time.time() + 10
        while not pids.exists() or not pids.read_text().strip():
            assert time.time() < deadline, "fake service did not start"
            time.sleep(0.05)
        service_pid, helper_pid = map(int, pids.read_text().split())
        return proc, service_pid, helper_pid

    yield start, status


def _wait(proc, timeout):
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("watchdog did not exit in time")


def test_sustained_503_stops_the_service_and_exits_1(service):
    start, status = service
    proc, service_pid, helper_pid = start()
    time.sleep(1.0)                       # armed after a healthy answer
    assert proc.poll() is None
    status.write_text("503")
    code = _wait(proc, 15)
    err = proc.stderr.read()
    assert code == 1, err
    assert "consecutive failed health checks" in err
    assert not _alive(service_pid) and not _alive(helper_pid)


def test_a_single_failure_does_not_trigger(service):
    start, status = service
    proc, service_pid, _ = start()
    time.sleep(0.8)
    status.write_text("503")
    time.sleep(0.3)                       # one or two failed checks, fewer than 3
    status.write_text("200")
    time.sleep(1.5)
    assert proc.poll() is None
    proc.send_signal(signal.SIGTERM)
    _wait(proc, 10)


def test_failures_before_the_first_healthy_answer_do_not_count(service):
    start, status = service
    status.write_text("503")
    proc, service_pid, _ = start()
    time.sleep(2.0)                       # ten failed checks, but never armed
    assert proc.poll() is None and _alive(service_pid)
    proc.send_signal(signal.SIGTERM)
    _wait(proc, 10)


def test_full_shared_memory_stops_the_service(service):
    start, _ = service
    proc, service_pid, helper_pid = start(WATCHDOG_SHM_MAX_USED_PCT="0.0001",
                                          WATCHDOG_SHM_FAILURES="2")
    code = _wait(proc, 15)
    assert code == 1
    assert "used" in proc.stderr.read()
    assert not _alive(service_pid) and not _alive(helper_pid)


def test_sigterm_is_forwarded_and_the_status_passed_on(service):
    start, _ = service
    proc, service_pid, _ = start()
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    code = _wait(proc, 10)
    assert code == 128 + signal.SIGTERM
    assert not _alive(service_pid)


def test_disabled_watchdog_only_runs_the_child(service):
    start, status = service
    proc, service_pid, _ = start(WATCHDOG_ENABLED="false")
    time.sleep(0.5)
    status.write_text("503")
    time.sleep(1.5)
    assert proc.poll() is None
    proc.send_signal(signal.SIGTERM)
    _wait(proc, 10)


def test_the_childs_exit_status_is_returned(tmp_path):
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, WATCHDOG_INTERVAL_SECONDS="0.2")
    code = subprocess.run(
        [sys.executable, "-m", "app.watchdog", "--", sys.executable, "-c", "raise SystemExit(7)"],
        env=env, timeout=20,
    ).returncode
    assert code == 7
