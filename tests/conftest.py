"""Shared fixtures: fake warm engines and a configured gateway client."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stt_gateway import server as server_module
from stt_gateway.engine import Transcription
from stt_gateway.schema import Word


class FakeEngine:
    """Deterministic warm-engine stand-in: one word per second of audio.

    Transcripts intentionally lack terminal punctuation so the gateway's
    `punctuate=true` rule is observable from the outside.
    """

    def __init__(self, name: str):
        self.name = name
        self.process = None
        self.ready = True
        self.seen: list[tuple[bytes, str]] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.ready = False

    async def transcribe(self, wav_bytes: bytes, language: str) -> Transcription:
        self.seen.append((wav_bytes, language))
        samples = (len(wav_bytes) - 44) // 2
        duration = samples / 16_000
        count = max(1, int(duration))
        words = [
            Word(text=f"word{index}", start=index * 0.5, end=index * 0.5 + 0.4, confidence=0.9)
            for index in range(count)
        ]
        return Transcription(
            text=" ".join(word.text for word in words),
            words=words,
            language=language,
            duration=duration,
        )


def _sine_seconds(seconds: float, rate: int = 16_000, amplitude: float = 0.5) -> np.ndarray:
    time = np.arange(int(seconds * rate), dtype=np.float64)
    return (amplitude * np.sin(2 * np.pi * 440.0 * time / rate)).astype(np.float32)


def _to_linear16(samples: np.ndarray) -> bytes:
    """Encode mono float32 samples as s16le bytes (the linear16 wire format)."""
    pcm = np.clip(samples, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()


def _listen_url(**params) -> str:
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"/v1/listen?{query}"


@pytest.fixture
def sine_seconds():
    """Provide the 440 Hz sine sample factory."""
    return _sine_seconds


@pytest.fixture
def linear16_bytes():
    """Provide the float32-to-s16le encoder."""
    return _to_linear16


@pytest.fixture
def listen_url():
    """Provide the /v1/listen URL builder."""
    return _listen_url


@pytest.fixture
def gateway(monkeypatch):
    """Gateway TestClient wired to FakeEngines with fast timings."""
    monkeypatch.setenv("STT_WINDOW_SECONDS", "1.0")
    monkeypatch.setenv("STT_FLUSH_SILENCE_SECONDS", "0.3")
    monkeypatch.setenv("STT_TICK_SECONDS", "0.05")

    def fake_build_engines(config, settings):
        return {spec.name: FakeEngine(spec.name) for spec in config.engine_specs()}

    monkeypatch.setattr(server_module, "build_engines", fake_build_engines)
    with TestClient(server_module.app) as client:
        yield client
