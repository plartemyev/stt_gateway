"""FastAPI WebSocket gateway mimicking the Deepgram streaming API.

Endpoint: `ws://<host>:8030/v1/listen` — the same path and query parameters as
`wss://api.deepgram.com/v1/listen`, so existing Deepgram streaming clients can
be pointed at it by swapping the hostname.

All engines named in the routing table (`stt_gateway.routing`) are kept warm;
each connection is routed to one of them based on the requested `model` and
`language` query parameters.
"""

import asyncio
import contextlib
import io
import logging
import os
import signal
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import asdict

import fastapi
import numpy as np
import uvicorn
from starlette.websockets import WebSocketDisconnect

from stt_gateway import auth, log, schema
from stt_gateway.audio import AudioDecodingError, ContainerDecoder, Linear16Decoder, TARGET_RATE
from stt_gateway.config import Settings, load_settings
from stt_gateway.engine import EngineError
from stt_gateway.engines import Engine, build_engines, probe_engines
from stt_gateway.routing import RoutingConfig, load_routing
from stt_gateway.windower import Windower

logger = logging.getLogger(__name__)

# Deepgram language codes (lowercased by the gateway, as sample
# clients do) -> whisper language.
LANGUAGE_MAP = {
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "ru": "ru",
    "ru-ru": "ru",
    "hy": "hy",
    "hy-am": "hy",
    "hye": "hy",
    "am": "hy",
    "arm": "hy",
    "multi": "auto",
}
SUPPORTED_LANGUAGES = ", ".join(sorted(LANGUAGE_MAP))

ENCODING_LINEAR16 = "linear16"
ENCODING_OGG_OPUS = "ogg-opus"


async def _monitor_process(process) -> None:
    """Exit the process tree when a spawned engine dies, so Docker restarts us."""
    while True:
        await asyncio.sleep(5)
        if process.poll() is not None:
            logger.critical("Engine process exited with code %s; terminating gateway", process.returncode)
            os.kill(os.getpid(), signal.SIGTERM)
            return


@contextlib.asynccontextmanager
async def lifespan(application: fastapi.FastAPI) -> AsyncIterator[None]:
    """Start every warm engine named in the routing table for the app lifetime."""
    settings = load_settings()
    application.state.settings = settings
    config = load_routing(settings.routing_config)
    application.state.routing = config
    engines = build_engines(config, settings)
    application.state.engines = engines
    monitors = []
    for engine in engines.values():
        logger.info("Starting engine %s (%s)", engine.name, type(engine).__name__)
        await engine.start()
        logger.info("Engine %s ready", engine.name)
        process = getattr(engine, "process", None)
        if process is not None:
            monitors.append(asyncio.create_task(_monitor_process(process)))
    try:
        yield
    finally:
        for monitor in monitors:
            monitor.cancel()
        for engine in engines.values():
            await engine.stop()


app = fastapi.FastAPI(title="stt_gateway", lifespan=lifespan)


@app.get("/api/health")
async def health() -> dict:
    """Liveness/readiness for compose healthchecks and load balancers."""
    engines = getattr(app.state, "engines", {})
    return {
        "status": "ok",
        "engines": [asdict(probe) for probe in probe_engines(app.state.routing, engines)]
        if engines
        else [],
    }


def encode_wav(samples: np.ndarray) -> bytes:
    """Encode 16 kHz mono float32 samples as an in-memory WAV."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_RATE)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


class _Session:
    """Per-connection decoding and windowing state."""

    def __init__(self, settings: Settings, encoding: str, sample_rate: int, channels: int):
        self.settings = settings
        self.channels = channels
        if encoding == ENCODING_LINEAR16:
            self.decoder = Linear16Decoder(sample_rate=sample_rate, channels=channels)
        else:
            self.decoder = ContainerDecoder()
        self.windower = Windower(
            window_seconds=settings.window_seconds,
            cut_lookback_seconds=settings.cut_lookback_seconds,
            flush_silence_seconds=settings.flush_silence_seconds,
            min_utterance_seconds=settings.min_utterance_seconds,
            max_buffer_seconds=settings.max_buffer_seconds,
            silence_rms=settings.silence_rms,
        )

    def close(self) -> None:
        self.decoder.close()


@app.websocket("/v1/listen")
async def listen(websocket: fastapi.WebSocket) -> None:
    """Deepgram-compatible streaming transcription endpoint."""
    settings: Settings = websocket.app.state.settings
    token = auth.extract_token(websocket.headers)
    if not auth.is_valid(token, settings.token_max_len):
        # Before accept() uvicorn answers the upgrade with HTTP 403
        # (Deepgram answers 401; for our clients the distinction is moot).
        await websocket.close(code=1008, reason="invalid or missing token")
        return
    try:
        encoding, sample_rate, channels, language, model, punctuate, session = _parse_query(websocket.query_params)
    except ValueError as e:
        await websocket.close(code=1008, reason=str(e))
        return
    routing: RoutingConfig = websocket.app.state.routing
    engine: Engine = websocket.app.state.engines[routing.resolve(model, language)]

    if session:
        # Propagate the consumer's session id (master_backend appends it) so
        # gateway logs join with the rest of the pipeline.
        log.set_trace_id(session)
    session_state = _Session(settings, encoding, sample_rate, channels)
    await websocket.accept()
    logger.info("Session opened: encoding=%s sample_rate=%s channels=%s language=%s model=%s engine=%s",
                encoding, sample_rate, channels, language, model or "(default)", engine.name)
    await websocket.send_json(schema.results_message(
        words=[], transcript="", language=language, start=0.0, duration=0.0,
        channels=channels, model_name=engine.name, is_final=False, speech_final=False,
    ))
    try:
        try:
            await _run_session(websocket, session_state, engine, language, punctuate)
        except* (EngineError, AudioDecodingError) as group:
            error = group.exceptions[0]
            logger.warning("Session failed: %s", error)
            await _close_quietly(websocket, 1011, str(error)[:120])
        else:
            await _close_quietly(websocket, 1000, "done")
    except WebSocketDisconnect:
        pass
    finally:
        session_state.close()
        logger.info("Session closed; buffered=%.1fs dropped=%.1fs",
                    session_state.windower.buffered_seconds, session_state.windower.dropped_seconds)


def _parse_query(query) -> tuple[str, int, int, str, str, bool, str]:
    """Map Deepgram query parameters onto internal session settings.

    `session` is a non-Deepgram extension: an opaque trace id from the caller.
    """
    encoding = query.get("encoding", ENCODING_OGG_OPUS)
    if encoding not in (ENCODING_OGG_OPUS, ENCODING_LINEAR16):
        raise ValueError(f"unsupported encoding: {encoding}")
    sample_rate = int(query.get("sample_rate", "48000"))
    channels = int(query.get("channels", "2"))
    language = LANGUAGE_MAP.get(query.get("language", "multi").lower())
    if language is None:
        raise ValueError(f"unsupported language, supported: {SUPPORTED_LANGUAGES}")
    model = (query.get("model") or "").strip().lower()
    punctuate = query.get("punctuate", "true").lower() not in ("false", "0", "no")
    session = (query.get("session") or "").strip()
    return encoding, sample_rate, channels, language, model, punctuate, session


def _punctuate(transcript: str, utterance_end: bool, punctuate: bool) -> str:
    """Apply the `punctuate=true` guarantee at utterance ends.

    Mid-speech window cuts are left untouched (the next window continues the
    sentence); when the speaker stopped talking the result must close with a
    sentence-ending mark, whatever the engine produced.
    """
    if not (punctuate and utterance_end):
        return transcript
    return schema.ensure_terminal_punctuated(transcript)


async def _run_session(
        websocket: fastapi.WebSocket,
        session: _Session,
        engine: Engine,
        language: str,
        punctuate: bool,
) -> None:
    """Receive audio and emit Results windows until the client disconnects."""
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    async def receiver() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                chunk = message.get("bytes")
                if chunk:
                    # ContainerDecoder.feed may block on a full pipe.
                    await loop.run_in_executor(None, session.decoder.feed, chunk)
                # Text frames are Deepgram control messages (KeepAlive,
                # CloseStream); none of our clients send them, so they are
                # accepted and ignored.
        finally:
            stop_event.set()

    async def pump() -> None:
        while not stop_event.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), session.settings.tick_seconds)
            samples = await loop.run_in_executor(None, session.decoder.drain)
            error = session.decoder.error
            if error is not None:
                raise AudioDecodingError(f"audio decode failed: {error}") from error
            session.windower.push(samples)
            window = session.windower.pop_window()
            if window is None:
                continue
            t_infer = time.perf_counter()
            transcription = await engine.transcribe(encode_wav(window.samples), language)
            infer_s = time.perf_counter() - t_infer
            logger.info("window=[%.2f,%.2f]s utterance_end=%s words=%d infer_s=%.3f",
                        window.start_seconds, window.end_seconds, window.utterance_end,
                        len(transcription.words), infer_s)
            transcript = _punctuate(transcription.text, window.utterance_end, punctuate)
            words = [
                schema.Word(
                    text=word.text,
                    start=window.start_seconds + word.start,
                    end=window.start_seconds + word.end,
                    confidence=word.confidence,
                )
                for word in transcription.words
            ]
            message = schema.results_message(
                words=words,
                transcript=transcript,
                language=language,
                start=window.start_seconds,
                duration=window.end_seconds - window.start_seconds,
                channels=session.channels,
                model_name=engine.name,
                timings={"infer_s": round(infer_s, 3)},
            )
            await websocket.send_json(message)

    async with asyncio.TaskGroup() as task_group:
        task_group.create_task(receiver())
        task_group.create_task(pump())


async def _close_quietly(websocket: fastapi.WebSocket, code: int, reason: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.close(code=code, reason=reason)


def main() -> None:
    """Run the gateway with uvicorn (container entrypoint)."""
    log.setup_logging()
    settings = load_settings()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.port,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )


if __name__ == "__main__":
    main()
