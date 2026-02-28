#!/usr/bin/env python3
"""
Stress test for the WhisperX ASR service.

Sends concurrent requests using audio files from the 'test files/' directory
and reports throughput, latency, and error rates.

Usage:
    # Default: 4 concurrent workers, all files, /asr endpoint
    python tests/stress_test.py

    # Custom concurrency and rounds
    python tests/stress_test.py --workers 8 --rounds 3

    # Test OpenAI-compat endpoint
    python tests/stress_test.py --endpoint openai

    # Quick smoke test (1 file, 1 worker)
    python tests/stress_test.py --workers 1 --rounds 1

    # Custom base URL
    python tests/stress_test.py --url http://remote-host:9000
"""

import argparse
import os
import sys
import time
import json
import statistics
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_URL = "http://localhost:9000"
TEST_FILES_DIR = Path(__file__).parent.parent / "testfiles"


@dataclass
class RequestResult:
    file: str
    status_code: int
    latency: float  # seconds
    response_size: int  # bytes
    error: Optional[str] = None
    segments: int = 0
    detected_language: Optional[str] = None
    # Pipeline validation
    has_text: bool = False
    has_word_timestamps: bool = False
    has_speakers: bool = False
    words_with_timestamps: int = 0
    speakers_found: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class TestReport:
    endpoint: str
    workers: int
    rounds: int
    total_files: int
    results: List[RequestResult] = field(default_factory=list)
    wall_time: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline output validation
# ---------------------------------------------------------------------------
def validate_asr_response(data: dict, result: RequestResult, diarize: bool):
    """Validate that the full pipeline produced expected output."""
    segments = data.get("segments", [])
    result.segments = len(segments)
    result.detected_language = data.get("language")

    # Check transcription produced text
    all_text = " ".join(seg.get("text", "") for seg in segments).strip()
    result.has_text = len(all_text) > 0
    if not result.has_text:
        result.validation_errors.append("No transcription text produced")

    # Check word-level timestamps (alignment stage)
    word_segments = data.get("word_segments", [])
    if not word_segments:
        for seg in segments:
            word_segments.extend(seg.get("words", []))
    result.words_with_timestamps = sum(
        1 for w in word_segments
        if "start" in w and "end" in w and "word" in w
    )
    result.has_word_timestamps = result.words_with_timestamps > 0
    if not result.has_word_timestamps:
        result.validation_errors.append("No word-level timestamps (alignment may have failed)")

    # Check speaker labels (diarization stage)
    if diarize:
        speakers = set()
        for seg in segments:
            spk = seg.get("speaker")
            if spk:
                speakers.add(spk)
        result.speakers_found = sorted(speakers)
        result.has_speakers = len(speakers) > 0
        if not result.has_speakers:
            result.validation_errors.append("No speaker labels (diarization may have failed)")


def validate_openai_response(data: dict, result: RequestResult):
    """Validate OpenAI-compat verbose_json response."""
    result.detected_language = data.get("language")

    text = data.get("text", "")
    result.has_text = len(text.strip()) > 0
    if not result.has_text:
        result.validation_errors.append("No transcription text produced")

    segments = data.get("segments", [])
    result.segments = len(segments)

    words = data.get("words", [])
    if words:
        result.words_with_timestamps = sum(
            1 for w in words if "start" in w and "end" in w
        )
        result.has_word_timestamps = result.words_with_timestamps > 0


# ---------------------------------------------------------------------------
# Request senders
# ---------------------------------------------------------------------------
def send_asr_request(
    base_url: str,
    file_path: Path,
    model: str = "large-v3",
    diarize: bool = True,
) -> RequestResult:
    """Send a request to the /asr endpoint."""
    url = f"{base_url}/asr"
    params = {
        "task": "transcribe",
        "word_timestamps": "true",
        "output_format": "json",
        "model": model,
        "diarize": str(diarize).lower(),
    }

    start = time.perf_counter()
    try:
        with open(file_path, "rb") as f:
            files = {"audio_file": (file_path.name, f, "audio/mpeg")}
            resp = requests.post(url, params=params, files=files, timeout=600)
        elapsed = time.perf_counter() - start

        result = RequestResult(
            file=file_path.name,
            status_code=resp.status_code,
            latency=elapsed,
            response_size=len(resp.content),
        )

        if resp.status_code == 200:
            try:
                data = resp.json()
                validate_asr_response(data, result, diarize)
            except (json.JSONDecodeError, KeyError) as e:
                result.validation_errors.append(f"Failed to parse response: {e}")
        else:
            result.error = resp.text[:200]

        return result

    except Exception as e:
        elapsed = time.perf_counter() - start
        return RequestResult(
            file=file_path.name,
            status_code=0,
            latency=elapsed,
            response_size=0,
            error=str(e),
        )


def send_openai_request(
    base_url: str,
    file_path: Path,
    model: str = "whisper-1",
    response_format: str = "verbose_json",
) -> RequestResult:
    """Send a request to the /v1/audio/transcriptions endpoint."""
    url = f"{base_url}/v1/audio/transcriptions"

    start = time.perf_counter()
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "audio/mpeg")}
            data = {
                "model": model,
                "response_format": response_format,
            }
            resp = requests.post(url, data=data, files=files, timeout=600)
        elapsed = time.perf_counter() - start

        result = RequestResult(
            file=file_path.name,
            status_code=resp.status_code,
            latency=elapsed,
            response_size=len(resp.content),
        )

        if resp.status_code == 200:
            try:
                body = resp.json()
                validate_openai_response(body, result)
            except (json.JSONDecodeError, KeyError) as e:
                result.validation_errors.append(f"Failed to parse response: {e}")
        else:
            result.error = resp.text[:200]

        return result

    except Exception as e:
        elapsed = time.perf_counter() - start
        return RequestResult(
            file=file_path.name,
            status_code=0,
            latency=elapsed,
            response_size=0,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_stress_test(
    base_url: str,
    endpoint: str,
    workers: int,
    rounds: int,
    model: str,
    diarize: bool,
) -> TestReport:
    audio_files = sorted(TEST_FILES_DIR.glob("*.mp3"))
    if not audio_files:
        print(f"No .mp3 files found in {TEST_FILES_DIR}")
        sys.exit(1)

    # Build the work queue: each round sends all files
    work_items = []
    for _ in range(rounds):
        work_items.extend(audio_files)

    report = TestReport(
        endpoint=endpoint,
        workers=workers,
        rounds=rounds,
        total_files=len(work_items),
    )

    print(f"Endpoint:    /{endpoint}")
    print(f"Base URL:    {base_url}")
    print(f"Workers:     {workers}")
    print(f"Rounds:      {rounds}")
    print(f"Audio files: {len(audio_files)} ({', '.join(f.name for f in audio_files)})")
    print(f"Total reqs:  {len(work_items)}")
    print(f"Model:       {model}")
    if endpoint == "asr":
        print(f"Diarize:     {diarize}")
    print("-" * 70)
    print("  T=Transcription  A=Alignment  D=Diarization  (Y=pass, N=missing)")
    print("-" * 70)

    sender = send_asr_request if endpoint == "asr" else send_openai_request

    wall_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, file_path in enumerate(work_items):
            if endpoint == "asr":
                fut = pool.submit(send_asr_request, base_url, file_path, model, diarize)
            else:
                fut = pool.submit(send_openai_request, base_url, file_path, model)
            futures[fut] = i

        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            result = fut.result()
            report.results.append(result)

            if result.status_code != 200:
                status = f"ERR {result.status_code}"
            elif result.validation_errors:
                status = "WARN"
            else:
                status = "OK"

            # Build pipeline check string
            checks = []
            if result.status_code == 200:
                checks.append(f"T:{'Y' if result.has_text else 'N'}")
                checks.append(f"A:{'Y' if result.has_word_timestamps else 'N'}")
                if diarize and endpoint == "asr":
                    checks.append(f"D:{'Y' if result.has_speakers else 'N'}")
            check_str = " ".join(checks)

            print(
                f"  [{len(report.results):>3}/{len(work_items)}] "
                f"{result.file:<25} {status:<4}  "
                f"{result.latency:>7.1f}s  "
                f"{result.segments:>4} segs  "
                f"{check_str}"
            )
            if result.error:
                print(f"         ERROR: {result.error[:100]}")
            for ve in result.validation_errors:
                print(f"         WARN: {ve}")

    report.wall_time = time.perf_counter() - wall_start
    return report


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(report: TestReport):
    print()
    print("=" * 70)
    print("STRESS TEST REPORT")
    print("=" * 70)

    successes = [r for r in report.results if r.status_code == 200]
    failures = [r for r in report.results if r.status_code != 200]

    print(f"Endpoint:        /{report.endpoint}")
    print(f"Workers:         {report.workers}")
    print(f"Rounds:          {report.rounds}")
    print(f"Total requests:  {report.total_files}")
    print(f"Successes:       {len(successes)}")
    print(f"Failures:        {len(failures)}")
    print(f"Wall time:       {report.wall_time:.1f}s")

    if successes:
        latencies = [r.latency for r in successes]
        print()
        print("Latency (successful requests):")
        print(f"  Min:           {min(latencies):.1f}s")
        print(f"  Max:           {max(latencies):.1f}s")
        print(f"  Mean:          {statistics.mean(latencies):.1f}s")
        print(f"  Median:        {statistics.median(latencies):.1f}s")
        if len(latencies) >= 2:
            print(f"  Stdev:         {statistics.stdev(latencies):.1f}s")
        print(f"  p95:           {sorted(latencies)[int(len(latencies) * 0.95)]:.1f}s")

        total_sequential = sum(latencies)
        print()
        print("Throughput:")
        print(f"  Sequential:    {total_sequential:.1f}s (sum of all latencies)")
        print(f"  Wall clock:    {report.wall_time:.1f}s")
        print(f"  Speedup:       {total_sequential / report.wall_time:.2f}x")
        print(f"  Reqs/min:      {len(successes) / report.wall_time * 60:.1f}")

    # Pipeline validation summary
    if successes:
        with_text = sum(1 for r in successes if r.has_text)
        with_alignment = sum(1 for r in successes if r.has_word_timestamps)
        with_speakers = sum(1 for r in successes if r.has_speakers)
        total_words = sum(r.words_with_timestamps for r in successes)
        all_speakers = set()
        for r in successes:
            all_speakers.update(r.speakers_found)

        print()
        print("Pipeline validation:")
        print(f"  Transcription:   {with_text}/{len(successes)} have text")
        print(f"  Alignment:       {with_alignment}/{len(successes)} have word timestamps ({total_words} total words)")
        if report.endpoint == "asr":
            print(f"  Diarization:     {with_speakers}/{len(successes)} have speaker labels ({len(all_speakers)} unique speakers)")

        warned = [r for r in successes if r.validation_errors]
        if warned:
            print()
            print("Validation warnings:")
            for r in warned:
                for ve in r.validation_errors:
                    print(f"  {r.file}: {ve}")

    if failures:
        print()
        print("Failures:")
        for r in failures:
            print(f"  {r.file}: HTTP {r.status_code} - {r.error[:80] if r.error else 'unknown'}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stress test the WhisperX ASR service")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL (default: %(default)s)")
    parser.add_argument("--endpoint", choices=["asr", "openai"], default="asr",
                        help="Endpoint to test (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers (default: %(default)s)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of rounds (each round sends all files) (default: %(default)s)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: large-v3 for /asr, whisper-1 for openai)")
    parser.add_argument("--no-diarize", action="store_true",
                        help="Disable diarization (only for /asr endpoint)")
    args = parser.parse_args()

    if args.model is None:
        args.model = "large-v3" if args.endpoint == "asr" else "whisper-1"

    # Health check
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        health = resp.json()
        print(f"Service healthy: {health.get('serve_mode', 'unknown')} mode")
        print()
    except Exception as e:
        print(f"Service not reachable at {args.url}: {e}")
        sys.exit(1)

    report = run_stress_test(
        base_url=args.url,
        endpoint=args.endpoint,
        workers=args.workers,
        rounds=args.rounds,
        model=args.model,
        diarize=not args.no_diarize,
    )

    print_report(report)


if __name__ == "__main__":
    main()
