"""Round-trip test of the ogg-opus path using a PyAV-encoded stream."""

import io
import time

import av
import numpy as np
import pytest

from stt_gateway.audio import AudioDecodingError, ContainerDecoder, TARGET_RATE


def encode_sine_ogg(seconds: float, rate: int = 48_000, frame_samples: int = 960) -> bytes:
    """Encode a 48 kHz stereo sine as an OGG Opus stream, like browser clients do."""
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="ogg") as container:
        stream = container.add_stream("libopus", rate=rate)
        stream.layout = "stereo"
        total = 0
        while total < seconds * rate:
            time_ = np.arange(total, total + frame_samples, dtype=np.float64)
            sine = (0.5 * np.sin(2 * np.pi * 440.0 * time_ / rate)).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(np.vstack([sine, sine]), format="fltp", layout="stereo")
            frame.sample_rate = rate
            frame.pts = total
            for packet in stream.encode(frame):
                container.mux(packet)
            total += frame_samples
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


@pytest.fixture
def decoder() -> ContainerDecoder:
    instance = ContainerDecoder()
    yield instance
    instance.close()


def test_container_decoder_roundtrip(decoder: ContainerDecoder):
    payload = encode_sine_ogg(1.0)
    half = len(payload) // 2
    decoder.feed(payload[:half])
    time.sleep(0.2)
    decoder.feed(payload[half:])
    samples = _drain_until(decoder, TARGET_RATE)
    assert abs(len(samples) - TARGET_RATE) < TARGET_RATE * 0.05
    assert float(np.max(np.abs(samples))) > 0.1


def test_container_decoder_surfaces_garbage(decoder: ContainerDecoder):
    # Unknown magic is rejected deterministically, before PyAV can block on it.
    with pytest.raises(AudioDecodingError):
        decoder.feed(b"\x00" * 4096)
    assert isinstance(decoder.error, AudioDecodingError)
    with pytest.raises(AudioDecodingError):
        decoder.feed(b"\x00" * 4096)


def _drain_until(decoder: ContainerDecoder, wanted: int, deadline_seconds: float = 5.0) -> np.ndarray:
    deadline = time.monotonic() + deadline_seconds
    collected = []
    while time.monotonic() < deadline:
        samples = decoder.drain()
        if samples.size:
            collected.append(samples)
            if sum(len(part) for part in collected) >= wanted:
                break
        time.sleep(0.05)
    return np.concatenate(collected) if collected else np.empty(0, dtype=np.float32)
