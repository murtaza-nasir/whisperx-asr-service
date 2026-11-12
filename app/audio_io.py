"""
Per-request job directories, streamed uploads and audio decoding.

Each request gets its own directory under JOB_DIR. The upload is copied
there in chunks (MAX_FILE_SIZE_MB is enforced during the copy), then ffmpeg
decodes it once, without blocking the event loop, to raw 16 kHz mono 16-bit
PCM (`audio.s16`). Pipeline stages receive the path of that file and load it
with load_pcm(), which returns the same array as whisperx.load_audio() on the
original upload. The decoded audio therefore never passes through the Ray
object store.

This module has no ML imports, so the CI tests and the watchdog can use it.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE
UPLOAD_CHUNK_BYTES = 1 << 20

PCM_NAME = "audio.s16"


def _env_or_none(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = _env_or_none(name)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        logger.warning(f"{name}={value!r} is not a number; using {default}")
        return default


JOB_DIR = _env_or_none("JOB_DIR") or os.path.join(tempfile.gettempdir(), "whisperx-jobs")
JOB_TTL_SECONDS = max(60, _env_int("JOB_TTL_SECONDS", 86400))

# Name patterns of upload copies left in the temp dir by versions before
# job directories existed (NamedTemporaryFile with the upload's suffix) and
# of Starlette's multipart spool files.
_LEGACY_TEMP_PREFIX = "tmp"


class UploadTooLarge(Exception):
    def __init__(self, size_bytes: int, limit_bytes: int):
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"File too large (more than {limit_bytes / (1024 * 1024):.0f}MB). "
            f"Maximum allowed: {limit_bytes / (1024 * 1024):.0f}MB."
        )


class AudioDecodeError(RuntimeError):
    pass


def ffmpeg_decode_args(src: str, dst: str, sr: int = SAMPLE_RATE) -> List[str]:
    """The ffmpeg command of whisperx.audio.load_audio, writing to `dst`
    instead of stdout. Keep the two in step (tests/unit/test_audio_io.py)."""
    return [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", src,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "-y", "-loglevel", "error", dst,
    ]


def ffprobe_args(src: str) -> List[str]:
    return [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", src,
    ]


def load_pcm(path: str) -> np.ndarray:
    """Load a decoded job file as float32, exactly as whisperx.load_audio
    converts ffmpeg's output."""
    return np.fromfile(path, dtype=np.int16).flatten().astype(np.float32) / 32768.0


def pcm_seconds(path: str) -> float:
    return os.path.getsize(path) / BYTES_PER_SECOND


def as_audio_array(audio) -> np.ndarray:
    """Stage helper: a job file path is loaded, an array is used as given."""
    if isinstance(audio, (str, os.PathLike)):
        return load_pcm(os.fspath(audio))
    return audio


async def _run(cmd: List[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except BaseException:
        # Cancelled request or timeout: do not leave ffmpeg running.
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


async def probe_duration(src: str, timeout: float = 30.0) -> Optional[float]:
    """Container duration from ffprobe, or None if it cannot tell."""
    if not shutil.which("ffprobe"):
        return None
    try:
        res = await _run(ffprobe_args(src), timeout=timeout)
    except (asyncio.TimeoutError, OSError):
        return None
    if res.returncode != 0:
        return None
    try:
        value = float(res.stdout.decode().strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
    return value if value > 0 else None


async def decode_to_pcm(src: str, dst: str) -> float:
    """Decode `src` to raw 16 kHz mono s16le at `dst` without blocking the
    event loop. Returns the duration in seconds."""
    if shutil.which("ffmpeg"):
        res = await _run(ffmpeg_decode_args(src, dst))
        if res.returncode != 0:
            raise AudioDecodeError(f"Failed to load audio: {res.stderr.decode(errors='replace')}")
    else:
        # No ffmpeg binary (some test setups): decode with whisperx in a
        # worker thread and store the same int16 samples.
        import whisperx

        audio = await asyncio.to_thread(whisperx.load_audio, src)
        pcm = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
        await asyncio.to_thread(pcm.tofile, dst)
    return pcm_seconds(dst)


class AudioJob:
    """One request's working directory. Use as an async context manager:
    the directory is removed on exit, also on errors and cancellation."""

    def __init__(self, root: Optional[str] = None):
        self.root = root or JOB_DIR
        self.id = uuid.uuid4().hex
        self.dir = os.path.join(self.root, self.id)
        self.upload_path: Optional[str] = None
        self.pcm_path = os.path.join(self.dir, PCM_NAME)
        self.size_bytes = 0
        self.duration: Optional[float] = None

    async def __aenter__(self):
        os.makedirs(self.dir, exist_ok=False)
        return self

    async def __aexit__(self, *exc):
        self.cleanup()
        return False

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    async def receive(self, upload, max_bytes: Optional[int] = None) -> int:
        """Copy an UploadFile into the job dir in chunks. Raises
        UploadTooLarge as soon as the copy passes `max_bytes`."""
        suffix = os.path.splitext(getattr(upload, "filename", None) or "")[1][:16] or ".bin"
        self.upload_path = os.path.join(self.dir, "upload" + suffix)
        total = 0
        with open(self.upload_path, "wb") as out:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes and total > max_bytes:
                    self.size_bytes = total
                    raise UploadTooLarge(total, max_bytes)
                out.write(chunk)
        self.size_bytes = total
        return total

    async def probe(self) -> Optional[float]:
        return await probe_duration(self.upload_path)

    async def decode(self) -> float:
        self.duration = await decode_to_pcm(self.upload_path, self.pcm_path)
        # The upload copy is no longer needed once the PCM file exists.
        try:
            os.unlink(self.upload_path)
        except OSError:
            pass
        return self.duration


def sweep(root: Optional[str] = None, max_age_seconds: Optional[float] = None,
          temp_dir: Optional[str] = None, now: Optional[float] = None) -> dict:
    """Remove job directories older than `max_age_seconds` (all of them when
    it is 0) and, in `temp_dir`, old upload copies and multipart spool files
    (`tmp*` regular files) left behind by a crash.

    Returns counts and bytes freed. Never raises.
    """
    root = root or JOB_DIR
    max_age = JOB_TTL_SECONDS if max_age_seconds is None else max_age_seconds
    now = time.time() if now is None else now
    stats = {"job_dirs": 0, "temp_files": 0, "bytes": 0}

    def _old(path: str) -> bool:
        try:
            return now - os.lstat(path).st_mtime >= max_age
        except OSError:
            return False

    def _size(path: str) -> int:
        total = 0
        for base, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(base, name)).st_size
                except OSError:
                    pass
        return total

    try:
        entries = os.listdir(root)
    except OSError:
        entries = []
    for name in entries:
        path = os.path.join(root, name)
        if os.path.isdir(path) and not os.path.islink(path) and _old(path):
            stats["bytes"] += _size(path)
            shutil.rmtree(path, ignore_errors=True)
            stats["job_dirs"] += 1

    if temp_dir:
        try:
            temp_entries = os.listdir(temp_dir)
        except OSError:
            temp_entries = []
        for name in temp_entries:
            if not name.startswith(_LEGACY_TEMP_PREFIX):
                continue
            path = os.path.join(temp_dir, name)
            if os.path.isfile(path) and not os.path.islink(path) and _old(path):
                try:
                    size = os.lstat(path).st_size
                    os.unlink(path)
                    stats["temp_files"] += 1
                    stats["bytes"] += size
                except OSError:
                    pass
    return stats


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Remove leftover job dirs and temp uploads.")
    parser.add_argument("--all", action="store_true",
                        help="remove every job dir and temp upload regardless of age "
                             "(for container start, when no request can be running)")
    parser.add_argument("--temp-dir", default=tempfile.gettempdir())
    args = parser.parse_args(argv)
    stats = sweep(max_age_seconds=0 if args.all else None, temp_dir=args.temp_dir)
    if stats["job_dirs"] or stats["temp_files"]:
        print(f"Startup cleanup: removed {stats['job_dirs']} job dirs and "
              f"{stats['temp_files']} temp files ({stats['bytes'] / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
