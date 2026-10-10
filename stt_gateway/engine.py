"""Async client for the whisper.cpp HTTP server (`whisper-server`).

The engine runs as a long-lived process next to the gateway (spawned by the
app lifespan) and keeps the model warm; every transcription window is one
HTTP POST with a WAV body.
"""

import dataclasses
import math
import time

import httpx


class EngineError(Exception):
    """The engine answered with an error."""


class EngineUnavailableError(EngineError):
    """The engine cannot be reached."""


@dataclasses.dataclass(frozen=True)
class Transcription:
    text: str
    words: list  # list[schema.Word]; timings relative to the transcribed audio
    language: str
    duration: float


class WhisperEngine:
    """Thin async client for whisper-server's `/inference` endpoint."""

    def __init__(self, base_url: str, timeout_seconds: float = 60.0):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def wait_ready(self, deadline_seconds: float, poll_interval_seconds: float = 1.0) -> None:
        """Poll until the engine answers HTTP at all (model may still be loading).

        whisper-server answers `GET /` with 200 as soon as the HTTP loop runs;
        inference requests that arrive before the model is loaded block until
        it is, so any HTTP response means "ready to accept work".
        """
        attempts = max(int(deadline_seconds / poll_interval_seconds), 1)
        for attempt in range(attempts):
            try:
                await self._client.get("/")
                return
            except httpx.HTTPError:
                if attempt == attempts - 1:
                    raise EngineUnavailableError(
                        f"whisper-server not ready after {deadline_seconds}s"
                    ) from None
                time.sleep(poll_interval_seconds)  # sync sleep: keeps this call executor-friendly

    async def transcribe(self, wav_bytes: bytes, language: str) -> Transcription:
        """Transcribe one in-memory WAV (16 kHz mono s16) window."""
        try:
            response = await self._client.post(
                "/inference",
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"language": language, "response_format": "verbose_json"},
            )
        except httpx.HTTPError as e:
            raise EngineUnavailableError(f"whisper-server unreachable: {e}") from e
        if response.status_code != 200:
            raise EngineError(f"whisper-server answered {response.status_code}: {response.text[:200]}")
        return parse_transcription(response.json())

    async def aclose(self) -> None:
        """Release the HTTP client."""
        await self._client.aclose()


def parse_transcription(payload: dict) -> Transcription:
    """Convert a whisper-server verbose_json payload to a Transcription.

    Word timings are derived by splitting each segment's text and distributing
    the segment time span evenly: whisper token-level words are sub-word
    fragments, and no current consumer reads word timestamps.
    """
    text = str(payload.get("text") or "").strip()
    words: list = []
    for segment in payload.get("segments") or []:
        words.extend(_words_from_segment(segment))
    return Transcription(
        text=text,
        words=words,
        language=str(payload.get("language") or ""),
        duration=float(payload.get("duration") or 0.0),
    )


def _words_from_segment(segment: dict) -> list:
    from stt_gateway.schema import Word

    start = float(segment.get("start") or 0.0)
    end = float(segment.get("end") or start)
    logprob = segment.get("avg_logprob")
    confidence = min(max(math.exp(logprob), 0.0), 1.0) if logprob is not None else 0.5
    tokens = str(segment.get("text") or "").split()
    if not tokens:
        return []
    step = max(end - start, 1e-6) / len(tokens)
    return [
        Word(text=token, start=start + index * step, end=start + (index + 1) * step, confidence=confidence)
        for index, token in enumerate(tokens)
    ]
