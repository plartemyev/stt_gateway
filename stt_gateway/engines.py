"""Warm engine registry: whisper-server subprocesses and in-process Seamless.

Every engine named in the routing table (`stt_gateway.routing`) is started at
application startup and kept warm; sessions route to one of them via
`RoutingConfig.resolve()`.

- `WhisperServerEngine` spawns a dedicated whisper-server subprocess per
  model and talks to it over HTTP (`engine.WhisperEngine`).
- `SeamlessEngine` loads SeamlessM4T v2 in-process (transformers, fp16 on
  CUDA) and runs generation on a worker thread.
"""

import asyncio
import dataclasses
import logging
import subprocess  # noqa: S404 - spawns the pinned local whisper-server binary
import sys

from stt_gateway import routing as routing_module
from stt_gateway.audio import TARGET_RATE, decode_wav_bytes
from stt_gateway.config import Settings
from stt_gateway.engine import (
    EngineError,
    Transcription,
    WhisperEngine,
)
from stt_gateway.schema import Word

logger = logging.getLogger(__name__)

# Whisper language code -> SeamlessM4T language code.
SEAMLESS_LANGUAGES = {"en": "eng", "ru": "rus", "hy": "hye"}
SEAMLESS_MAX_NEW_TOKENS = 384
# SeamlessM4T does not expose token confidences through a simple API; the
# constant below only feeds the Deepgram `words[].confidence` field, which no
# consumer reads.
SEAMLESS_PLACEHOLDER_CONFIDENCE = 0.75


class Engine:
    """Interface shared by all warm engines."""

    name: str

    async def start(self) -> None:
        """Load/spawn the engine and block until it accepts work."""
        raise NotImplementedError

    async def transcribe(self, wav_bytes: bytes, language: str) -> Transcription:
        """Transcribe one 16 kHz mono WAV window."""
        raise NotImplementedError

    async def stop(self) -> None:
        """Release the engine's resources."""
        raise NotImplementedError

    @property
    def ready(self) -> bool:
        """Whether the engine finished starting."""
        raise NotImplementedError


class WhisperServerEngine(Engine):
    """One whisper model served by its own whisper-server subprocess."""

    def __init__(self, name: str, model: str, settings: Settings, port: int):
        self.name = name
        self.model = model
        self.settings = settings
        self.port = port
        self.process: subprocess.Popen | None = None
        self.client: WhisperEngine | None = None
        self._ready = False

    async def start(self) -> None:
        """Spawn the whisper-server subprocess and wait for its HTTP loop."""
        self._ready = False
        if not self.settings.engine_external:
            self.process = _spawn_whisper_server(self.settings, self.model, self.port)
        self.client = WhisperEngine(
            f"http://{self.settings.engine_host}:{self.port}",
            timeout_seconds=self.settings.engine_timeout_seconds,
        )
        await self.client.wait_ready(self.settings.engine_start_timeout_seconds)
        self._ready = True

    async def transcribe(self, wav_bytes: bytes, language: str) -> Transcription:
        """Transcribe one 16 kHz mono WAV window."""
        return await self.client.transcribe(wav_bytes, language)

    async def stop(self) -> None:
        """Terminate the subprocess and release the HTTP client."""
        self._ready = False
        if self.client is not None:
            await self.client.aclose()
        if self.process is not None:
            self.process.terminate()
            self.process = None

    @property
    def ready(self) -> bool:
        """Whether the engine finished starting."""
        return self._ready


class SeamlessEngine(Engine):
    """SeamlessM4T v2 in-process via transformers (fp16 on CUDA).

    The heavy imports happen in start() so that whisper-only deployments and
    unit tests never touch torch/transformers.
    """

    def __init__(self, name: str, model: str, settings: Settings, revision: str = ""):
        self.name = name
        self.model = model  # HuggingFace model id
        self.revision = revision
        self.settings = settings
        self._model = None
        self._processor = None
        self._torch = None
        self._lock = asyncio.Lock()
        self._ready = False

    async def start(self) -> None:
        """Load the model onto the GPU (blocking until ready)."""
        self._ready = False
        await asyncio.to_thread(self._load)

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model, revision=self.revision)
        self._model = (
            SeamlessM4Tv2ForSpeechToText.from_pretrained(self.model, revision=self.revision)
            .to("cuda", dtype=torch.float16)
            .eval()
        )
        self._ready = True

    def _generate(self, samples, language: str) -> str:
        """Run one greedy fp16 generation on the GPU."""
        inputs = self._processor(audio=samples, sampling_rate=TARGET_RATE, return_tensors="pt")
        inputs = {key: value.to("cuda", dtype=self._torch.float16) for key, value in inputs.items()}
        with self._torch.no_grad():
            token_ids = self._model.generate(
                **inputs,
                tgt_lang=SEAMLESS_LANGUAGES[language],
                num_beams=1,
                do_sample=False,
                max_new_tokens=SEAMLESS_MAX_NEW_TOKENS,
            )
        return self._processor.batch_decode(token_ids, skip_special_tokens=True)[0]

    async def transcribe(self, wav_bytes: bytes, language: str) -> Transcription:
        """Transcribe one 16 kHz mono WAV window (serialized by a lock)."""
        if language not in SEAMLESS_LANGUAGES:
            raise EngineError(f"seamless requires a concrete language, got {language!r}")
        samples = decode_wav_bytes(wav_bytes)
        duration = len(samples) / TARGET_RATE
        async with self._lock:
            text = await asyncio.to_thread(self._generate, samples, language)
        text = text.strip()
        return Transcription(
            text=text,
            words=_words_from_span(text, duration),
            language=language,
            duration=duration,
        )

    async def stop(self) -> None:
        """Drop the model and free GPU memory."""
        self._ready = False
        self._model = None
        self._processor = None
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    @property
    def ready(self) -> bool:
        """Whether the engine finished starting."""
        return self._ready


def _words_from_span(text: str, duration: float) -> list[Word]:
    """Distribute the words of `text` evenly across [0, duration].

    Same approximation as the whisper path: no current consumer reads word
    timestamps; the Deepgram schema requires the field to exist.
    """
    tokens = text.split()
    if not tokens:
        return []
    step = max(duration, 1e-6) / len(tokens)
    return [
        Word(
            text=token,
            start=index * step,
            end=(index + 1) * step,
            confidence=SEAMLESS_PLACEHOLDER_CONFIDENCE,
        )
        for index, token in enumerate(tokens)
    ]


def _spawn_whisper_server(settings: Settings, model: str, port: int) -> subprocess.Popen:
    """Start one whisper-server subprocess for the given ggml model."""
    command = [
        settings.engine_bin,
        "--model", settings.model_dir_and_slash() + f"{model}.bin",
        "--threads", str(settings.engine_threads),
        "--host", settings.engine_host,
        "--port", str(port),
        "--inference-path", "/inference",
    ]
    if settings.vad:
        command += ["--vad", "--vad-model", settings.vad_model_path()]
    logger.info("Starting engine: %s", " ".join(command))
    return subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr)  # noqa: S603 - pinned local binary


def build_engines(config: routing_module.RoutingConfig, settings: Settings) -> dict:
    """Instantiate every engine in the routing table.

    Whisper engines get consecutive ports starting at settings.engine_port.
    """
    engines: dict = {}
    port = settings.engine_port
    for spec in config.engine_specs():
        if spec.kind == "whisper":
            engines[spec.name] = WhisperServerEngine(spec.name, spec.model, settings, port)
            port += 1
        elif spec.kind == "seamless":
            engines[spec.name] = SeamlessEngine(spec.name, spec.model, settings, revision=spec.revision)
        else:
            raise ValueError(f"unknown engine kind {spec.kind!r}")
    return engines


@dataclasses.dataclass(frozen=True)
class EngineProbe:
    """Health endpoint summary of one engine."""

    name: str
    kind: str
    model: str
    ready: bool


def probe_engines(config: routing_module.RoutingConfig, engines: dict) -> list[EngineProbe]:
    """Summarize engine readiness for the health endpoint."""
    return [
        EngineProbe(
            name=spec.name,
            kind=spec.kind,
            model=spec.model,
            ready=getattr(engines.get(spec.name), "ready", False),
        )
        for spec in config.engine_specs()
    ]
