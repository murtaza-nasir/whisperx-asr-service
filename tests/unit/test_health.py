"""Health evaluation: low shared memory or disk, and unresponsive stages."""

from app import health


def _usage(table):
    def usage(path):
        return table.get(path)
    return usage


def _info(path, total_mb, free_mb):
    return {"path": path, "total_mb": total_mb, "free_mb": free_mb,
            "free_pct": round(100.0 * free_mb / total_mb, 2)}


def test_healthy_when_there_is_room_and_every_stage_answers():
    usage = _usage({"/dev/shm": _info("/dev/shm", 8000, 4000),
                    "/jobs": _info("/jobs", 100000, 50000)})
    reasons, details = health.check_resources(["/dev/shm"], ["/jobs"], 5, 500, usage=usage)
    assert reasons == [] and set(details) == {"/dev/shm", "/jobs"}
    code, status, reasons = health.evaluate(reasons, {"whisper": {"ok": True},
                                                       "align": {"ok": True, "busy": True}})
    assert (code, status, reasons) == (200, "healthy", [])


def test_low_shared_memory_is_degraded():
    usage = _usage({"/dev/shm": _info("/dev/shm", 8000, 100)})
    reasons, _ = health.check_resources(["/dev/shm"], [], 5, 500, usage=usage)
    assert len(reasons) == 1 and "/dev/shm" in reasons[0]
    code, status, _ = health.evaluate(reasons)
    assert (code, status) == (503, "degraded")


def test_low_disk_is_degraded():
    usage = _usage({"/jobs": _info("/jobs", 100000, 200)})
    reasons, _ = health.check_resources([], ["/jobs"], 5, 500, usage=usage)
    assert reasons and "200" in reasons[0]


def test_thresholds_of_zero_turn_the_checks_off():
    usage = _usage({"/dev/shm": _info("/dev/shm", 8000, 0),
                    "/jobs": _info("/jobs", 1000, 0)})
    reasons, _ = health.check_resources(["/dev/shm"], ["/jobs"], 0, 0, usage=usage)
    assert reasons == []


def test_unreadable_paths_are_skipped():
    reasons, details = health.check_resources(["/nope"], ["/nope2"], 5, 500, usage=_usage({}))
    assert reasons == [] and details == {}


def test_a_stage_that_does_not_answer_is_degraded():
    code, status, reasons = health.evaluate([], {
        "whisper": {"ok": True},
        "diarize": {"ok": False, "error": "no answer within 5s"},
    })
    assert code == 503 and status == "degraded"
    assert reasons == ["stage diarize not responding: no answer within 5s"]


def test_disk_usage_reads_a_real_filesystem(tmp_path):
    info = health.disk_usage(str(tmp_path))
    assert info["total_mb"] > 0 and 0 <= info["free_pct"] <= 100
    assert health.disk_usage(str(tmp_path / "missing")) is None
