# Tests

## Stress Test

`stress_test.py` sends concurrent audio transcription requests to measure throughput, latency, and error rates across the service's GPU pipeline.

### Prerequisites

- Service running at `http://localhost:9000` (or specify `--url`)
- Audio files (`.mp3`) in the `testfiles/` directory at the project root
- Python `requests` library installed

### Usage

```bash
# Default: 4 concurrent workers, 1 round of all files, /asr endpoint
python tests/stress_test.py

# Quick smoke test (single file, single worker)
python tests/stress_test.py --workers 1 --rounds 1

# Heavy load: 8 concurrent workers, 3 rounds (sends all files 3 times)
python tests/stress_test.py --workers 8 --rounds 3

# Test OpenAI-compatible endpoint
python tests/stress_test.py --endpoint openai

# Without diarization (faster, transcription + alignment only)
python tests/stress_test.py --no-diarize

# Against a remote host
python tests/stress_test.py --url http://gpu-server:9000
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:9000` | Service base URL |
| `--endpoint` | `asr` | `asr` or `openai` (`/v1/audio/transcriptions`) |
| `--workers` | `4` | Number of concurrent request threads |
| `--rounds` | `1` | Number of times to send all files |
| `--model` | `large-v3` / `whisper-1` | Model name (auto-selected per endpoint) |
| `--no-diarize` | off | Skip speaker diarization (`/asr` only) |

### Output

Per-request progress lines followed by a summary report:

```
Service healthy: ray mode

Endpoint:    /asr
Workers:     4
Rounds:      1
Audio files: 4 (250218_0013.mp3, 250321_2144.mp3, ...)
Total reqs:  4
Model:       large-v3
Diarize:     True
----------------------------------------------------------------------
  [  1/  4] 250218_0013.mp3           OK    42.3s    87 segs    12.4 KB
  [  2/  4] 250321_2144.mp3           OK    38.7s    72 segs    10.1 KB
  ...

======================================================================
STRESS TEST REPORT
======================================================================
Total requests:  4
Successes:       4
Failures:        0
Wall time:       45.2s

Latency (successful requests):
  Min:           38.7s
  Max:           45.1s
  Mean:          41.2s
  Median:        40.5s

Throughput:
  Sequential:    164.8s (sum of all latencies)
  Wall clock:    45.2s
  Speedup:       3.65x
  Reqs/min:      5.3
======================================================================
```

The **Speedup** metric shows how much faster concurrent processing is compared to sequential -- a value close to `NUM_GPU_REPLICAS` indicates good GPU utilization.
