"""Environment-driven settings."""

import dataclasses
import os


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclasses.dataclass(frozen=True)
class Settings:
    """All tunables of the gateway. Every field has an `STT_*` env override."""

    # HTTP/WS server
    port: int = 8030
    # Any string is accepted as a token, but only up to this length.
    token_max_len: int = 256

    # whisper.cpp engines (models come from the routing table)
    models_dir: str = "/models"
    engine_bin: str = "/opt/whisper/bin/whisper-server"
    engine_host: str = "127.0.0.1"
    engine_port: int = 8081
    engine_threads: int = 8
    engine_timeout_seconds: float = 60.0
    engine_start_timeout_seconds: float = 300.0
    engine_external: bool = False  # true: connect to an already-running whisper-server
    vad: bool = True
    vad_model: str = "ggml-silero-v6.2.0.bin"
    # Optional JSON file overriding the built-in engine routing table.
    routing_config: str = ""

    # Audio pipeline (samples are 16 kHz mono float32 internally)
    sample_rate: int = 16_000
    window_seconds: float = 4.0
    cut_lookback_seconds: float = 0.75
    flush_silence_seconds: float = 1.2
    min_utterance_seconds: float = 0.2
    max_buffer_seconds: float = 300.0
    silence_rms: float = 0.005
    tick_seconds: float = 0.25

    def engine_base_url(self) -> str:
        """Return the whisper-server HTTP base URL."""
        return f"http://{self.engine_host}:{self.engine_port}"

    def vad_model_path(self) -> str:
        """Absolute path of the Silero VAD model file."""
        return f"{self.model_dir_and_slash()}{self.vad_model}"

    def model_dir_and_slash(self) -> str:
        """Models directory with a trailing slash."""
        return self.models_dir.rstrip("/") + "/"


def load_settings() -> Settings:
    """Build Settings from STT_* environment variables."""
    return Settings(
        port=_env_int("STT_PORT", 8030),
        token_max_len=_env_int("STT_TOKEN_MAX_LEN", 256),
        models_dir=_env_str("STT_MODELS_DIR", "/models"),
        engine_bin=_env_str("STT_ENGINE_BIN", "/opt/whisper/bin/whisper-server"),
        engine_host=_env_str("STT_ENGINE_HOST", "127.0.0.1"),
        engine_port=_env_int("STT_ENGINE_PORT", 8081),
        engine_threads=_env_int("STT_ENGINE_THREADS", 8),
        engine_timeout_seconds=_env_float("STT_ENGINE_TIMEOUT", 60.0),
        engine_start_timeout_seconds=_env_float("STT_ENGINE_START_TIMEOUT", 300.0),
        engine_external=_env_bool("STT_ENGINE_EXTERNAL", False),
        vad=_env_bool("STT_VAD", True),
        vad_model=_env_str("STT_VAD_MODEL", "ggml-silero-v6.2.0.bin"),
        routing_config=_env_str("STT_ROUTING_CONFIG", ""),
        sample_rate=_env_int("STT_SAMPLE_RATE", 16_000),
        window_seconds=_env_float("STT_WINDOW_SECONDS", 4.0),
        cut_lookback_seconds=_env_float("STT_CUT_LOOKBACK_SECONDS", 0.75),
        flush_silence_seconds=_env_float("STT_FLUSH_SILENCE_SECONDS", 1.2),
        min_utterance_seconds=_env_float("STT_MIN_UTTERANCE_SECONDS", 0.2),
        max_buffer_seconds=_env_float("STT_MAX_BUFFER_SECONDS", 300.0),
        silence_rms=_env_float("STT_SILENCE_RMS", 0.005),
        tick_seconds=_env_float("STT_TICK_SECONDS", 0.25),
    )
