"""Decoding of client audio streams to 16 kHz mono float32.

Two encodings arrive from clients (`encoding` query param):

- `linear16`: headerless signed 16-bit little-endian PCM, decoded with
  numpy/scipy (downmix + polyphase resample).
- `ogg-opus`: an OGG (or WebM) container streaming Opus, decoded incrementally
  with PyAV on a daemon thread (PyAV reads are blocking).
"""

import math
import os
import threading

import av
import numpy as np
from scipy import signal

TARGET_RATE = 16_000
TARGET_DTYPE = np.float32


class AudioDecodingError(Exception):
    """Raised when the audio payload cannot be decoded."""


def decode_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """Decode an in-memory 16 kHz mono s16 WAV to float32 samples.

    The gateway's window format (as produced by encode_wav) is the input;
    samples are scaled to [-1, 1].
    """
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        if wav_file.getframerate() != TARGET_RATE or wav_file.getnchannels() != 1:
            raise AudioDecodingError("expected 16 kHz mono WAV")
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return pcm.astype(TARGET_DTYPE) / 32768.0


def resample_to_mono_16k(samples: np.ndarray, sample_rate: int, channels: int) -> np.ndarray:
    """Downmix to mono and resample to TARGET_RATE."""
    frames = samples.reshape(-1, channels)
    mono = frames.mean(axis=1) if channels > 1 else frames[:, 0]
    if sample_rate != TARGET_RATE:
        divisor = math.gcd(sample_rate, TARGET_RATE)
        mono = signal.resample_poly(mono, TARGET_RATE // divisor, sample_rate // divisor)
    return mono.astype(TARGET_DTYPE)


class Linear16Decoder:
    """Decoder for headerless s16le PCM at a fixed rate/channel count."""

    def __init__(self, sample_rate: int = 48_000, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.error: Exception | None = None
        self._pending = bytearray()

    def feed(self, chunk: bytes) -> None:
        """Buffer a raw byte chunk; called from the event loop executor."""
        self._pending.extend(chunk)

    def drain(self) -> np.ndarray:
        """Return all complete samples decoded so far (16 kHz mono)."""
        frame_bytes = 2 * self.channels
        usable = len(self._pending) - (len(self._pending) % frame_bytes)
        raw = bytes(self._pending[:usable])
        del self._pending[:usable]
        if not raw:
            return np.empty(0, dtype=TARGET_DTYPE)
        pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, self.channels)
        return resample_to_mono_16k(pcm, self.sample_rate, self.channels)

    def close(self) -> None:
        """Discard buffered bytes."""
        self._pending.clear()


class ContainerDecoder:
    """Incremental decoder for ogg/webm Opus containers fed via feed().

    PyAV demuxing blocks on reads, so decoding runs on a daemon thread pulling
    from an os.pipe. Decoded samples are accumulated under a lock and returned
    by drain(). The first decode failure is stored and re-raised by feed().

    Because PyAV blocks indefinitely waiting for probe data on unknown input,
    the first bytes are validated against the container magics we support
    (OggS for ogg-opus, EBML for webm-opus) before reaching PyAV.
    """

    _MAGICS = (b"OggS", b"\x1a\x45\xdf\xa3")

    def __init__(self):
        self._read_fd, self._write_fd = os.pipe()
        self._samples: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._error: Exception | None = None
        self._prefix = bytearray()
        self._validated = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, chunk: bytes) -> None:
        """Write a container fragment; blocks when the pipe is full."""
        if self._error is not None:
            raise AudioDecodingError(f"audio decode failed: {self._error}") from self._error
        if not self._validated:
            self._prefix.extend(chunk)
            if len(self._prefix) < 4:
                return
            if not any(self._prefix.startswith(magic) for magic in self._MAGICS):
                self._error = AudioDecodingError("unsupported container: expected OggS or EBML header")
                raise self._error
            chunk = bytes(self._prefix)
            self._prefix.clear()
            self._validated = True
        try:
            os.write(self._write_fd, chunk)
        except OSError as e:
            raise AudioDecodingError(f"audio pipe closed: {e}") from e

    def drain(self) -> np.ndarray:
        """Return all complete samples decoded so far (16 kHz mono)."""
        with self._lock:
            if not self._samples:
                return np.empty(0, dtype=TARGET_DTYPE)
            data = np.concatenate(self._samples)
            self._samples.clear()
            return data  # noqa: R504 - mutation under lock requires the local

    @property
    def error(self) -> Exception | None:
        """First failure of the decoder thread, if any."""
        return self._error

    def close(self) -> None:
        """Signal end-of-stream and stop the decoder thread."""
        try:
            os.close(self._write_fd)
        except OSError:
            pass
        self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            with os.fdopen(self._read_fd, "rb", buffering=0) as source:
                with av.open(source) as container:
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_RATE)
                    for frame in container.decode(audio=0):
                        for output in resampler.resample(frame):
                            samples = output.to_ndarray().reshape(-1)
                            pcm = samples.astype(TARGET_DTYPE) / 32768.0
                            with self._lock:
                                self._samples.append(pcm)
        except Exception as e:  # noqa: BLE001 - any decode failure ends the stream
            self._error = e
