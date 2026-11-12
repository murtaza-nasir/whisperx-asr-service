"""Job files: decoding once to PCM, loading it in a stage, limits and cleanup.

The equality tests decode real audio with ffmpeg, so they are skipped when
ffmpeg is not installed. The reference is the decoding code of
whisperx.audio.load_audio (whisperx 3.7.4); when the real whisperx package is
importable (inside the service image), it is compared as well.
"""

import asyncio
import io
import os
import shutil
import subprocess
import time

import numpy as np
import pytest

from app import audio_io
from app.audio_io import AudioJob, UploadTooLarge, load_pcm, sweep

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")


def reference_load_audio(file, sr=16000):
    """whisperx.audio.load_audio as of whisperx 3.7.4."""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", file,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def _real_whisperx_load_audio():
    try:
        from whisperx.audio import load_audio  # the stub package has no whisperx.audio
    except Exception:
        return None
    return load_audio


def _make(path, codec_args, seconds=7.3, rate=44100, channels=2):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration={seconds}",
         "-f", "lavfi", "-i", f"anoisesrc=a=0.1:r={rate}:d={seconds}",
         "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]", "-map", "[a]",
         "-ac", str(channels), *codec_args, path],
        check=True,
    )
    return path


class _Upload:
    """Minimal UploadFile stand-in."""

    def __init__(self, data: bytes, filename="a.wav"):
        self._buf = io.BytesIO(data)
        self.filename = filename
        self.reads = 0

    async def read(self, n=-1):
        self.reads += 1
        return self._buf.read(n)


def test_decode_args_match_whisperx_load_audio():
    args = audio_io.ffmpeg_decode_args("in.mp3", "out.s16")
    reference = ["ffmpeg", "-nostdin", "-threads", "0", "-i", "in.mp3",
                 "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000"]
    assert args[:len(reference)] == reference
    # Only output handling follows: overwrite, quiet, and the file instead of "-".
    assert args[len(reference):] == ["-y", "-loglevel", "error", "out.s16"]


@needs_ffmpeg
@pytest.mark.parametrize("name,codec", [
    ("stereo44k.wav", ["-c:a", "pcm_s16le"]),
    ("a.mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    ("a.flac", ["-c:a", "flac"]),
    ("a.ogg", ["-c:a", "libvorbis"]),
    ("a.m4a", ["-c:a", "aac", "-b:a", "64k"]),
])
def test_loaded_pcm_is_bit_identical_to_whisperx_load_audio(tmp_path, name, codec):
    try:
        src = _make(str(tmp_path / name), codec)
    except subprocess.CalledProcessError:
        pytest.skip(f"this ffmpeg build cannot encode {name}")
    dst = str(tmp_path / "audio.s16")
    seconds = asyncio.run(audio_io.decode_to_pcm(src, dst))
    ours = load_pcm(dst)
    expected = reference_load_audio(src)
    assert ours.dtype == expected.dtype == np.float32
    assert ours.shape == expected.shape
    assert np.array_equal(ours, expected)
    assert seconds == pytest.approx(len(expected) / 16000)
    real = _real_whisperx_load_audio()
    if real is not None:
        assert np.array_equal(ours, real(src))


@needs_ffmpeg
def test_decoding_does_not_block_the_event_loop(tmp_path):
    src = _make(str(tmp_path / "long.mp3"), ["-c:a", "libmp3lame", "-b:a", "32k"],
                seconds=1200, rate=16000, channels=1)

    async def scenario():
        gaps = []
        stop = False

        async def ticker():
            last = time.monotonic()
            while not stop:
                await asyncio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        task = asyncio.create_task(ticker())
        started = time.monotonic()
        await audio_io.decode_to_pcm(src, str(tmp_path / "audio.s16"))
        took = time.monotonic() - started
        stop = True
        await task
        return took, max(gaps)

    took, worst_gap = asyncio.run(scenario())
    assert worst_gap < 0.25, f"event loop stalled {worst_gap:.2f}s during a {took:.2f}s decode"


@needs_ffmpeg
def test_a_corrupt_file_is_a_decode_error(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all" * 100)
    with pytest.raises(audio_io.AudioDecodeError):
        asyncio.run(audio_io.decode_to_pcm(str(bad), str(tmp_path / "out.s16")))


def test_upload_is_streamed_in_chunks_and_the_job_dir_removed(tmp_path):
    data = os.urandom(3 * audio_io.UPLOAD_CHUNK_BYTES + 17)

    async def scenario():
        upload = _Upload(data, "talk.mp3")
        async with AudioJob(root=str(tmp_path)) as job:
            size = await job.receive(upload, max_bytes=10 * audio_io.UPLOAD_CHUNK_BYTES)
            assert size == len(data)
            assert job.upload_path.endswith("upload.mp3")
            with open(job.upload_path, "rb") as f:
                assert f.read() == data
            assert upload.reads >= 4
            return job.dir

    job_dir = asyncio.run(scenario())
    assert not os.path.exists(job_dir)
    assert os.listdir(tmp_path) == []


def test_the_size_limit_stops_the_copy_early(tmp_path):
    limit = 2 * audio_io.UPLOAD_CHUNK_BYTES
    upload = _Upload(os.urandom(20 * audio_io.UPLOAD_CHUNK_BYTES))

    async def scenario():
        async with AudioJob(root=str(tmp_path)) as job:
            await job.receive(upload, max_bytes=limit)

    with pytest.raises(UploadTooLarge):
        asyncio.run(scenario())
    assert upload.reads <= 4  # stopped soon after the limit, not at the end
    assert os.listdir(tmp_path) == []


def test_the_job_dir_is_removed_when_the_request_is_cancelled(tmp_path):
    async def scenario():
        entered = asyncio.Event()

        async def request():
            async with AudioJob(root=str(tmp_path)) as job:
                await job.receive(_Upload(b"x" * 1000))
                entered.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(request())
        await entered.wait()
        assert len(os.listdir(tmp_path)) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert os.listdir(tmp_path) == []


def test_stages_accept_a_path_or_an_array(tmp_path):
    pcm = np.array([0, 1, -1, 32767, -32768, 1234], dtype=np.int16)
    path = tmp_path / "audio.s16"
    pcm.tofile(path)
    from_path = audio_io.as_audio_array(str(path))
    assert np.array_equal(from_path, pcm.astype(np.float32) / 32768.0)
    array = np.ones(4, dtype=np.float32)
    assert audio_io.as_audio_array(array) is array
    assert audio_io.pcm_seconds(str(path)) == pytest.approx(len(pcm) / 16000)


def test_sweep_removes_old_job_dirs_and_leftover_uploads(tmp_path):
    jobs = tmp_path / "jobs"
    temp = tmp_path / "temp"
    jobs.mkdir()
    temp.mkdir()
    old_job = jobs / "old"
    old_job.mkdir()
    (old_job / "audio.s16").write_bytes(b"\0" * 1000)
    new_job = jobs / "new"
    new_job.mkdir()
    old_upload = temp / "tmpab12cd.mp3"
    old_upload.write_bytes(b"\0" * 500)
    other = temp / "keep-me.txt"
    other.write_bytes(b"x")
    a_dir = temp / "tmpdir123"
    a_dir.mkdir()
    now = time.time()
    for p in (old_job, old_upload, other, a_dir):
        os.utime(p, (now - 7200, now - 7200))

    stats = sweep(root=str(jobs), max_age_seconds=3600, temp_dir=str(temp), now=now)
    assert stats["job_dirs"] == 1 and stats["temp_files"] == 1
    assert stats["bytes"] == 1500
    assert sorted(os.listdir(jobs)) == ["new"]
    assert sorted(os.listdir(temp)) == ["keep-me.txt", "tmpdir123"]

    stats = sweep(root=str(jobs), max_age_seconds=0, temp_dir=str(temp), now=now)
    assert stats["job_dirs"] == 1
    assert os.listdir(jobs) == []


def test_sweep_never_raises_on_missing_dirs(tmp_path):
    assert sweep(root=str(tmp_path / "nope"), max_age_seconds=0,
                 temp_dir=str(tmp_path / "nope2")) == {"job_dirs": 0, "temp_files": 0, "bytes": 0}
