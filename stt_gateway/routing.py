"""Model routing: API `model` values + language -> warm engine.

The default table is defined here and can be overridden wholesale with a JSON
file (`STT_ROUTING_CONFIG`), so deployments can remap engines without code
changes:

    {
      "engines": {
        "whisper-large-v3": {"kind": "whisper", "model": "ggml-large-v3"},
        "seamless-m4t-v2-large": {"kind": "seamless", "model": "seamless-m4t-v2-large"}
      },
      "supported_models": {
        "whisper-large-v3": "whisper-large-v3",
        "seamless-m4t-v2-large": "seamless-m4t-v2-large"
      },
      "fallback_by_language": {"en": "whisper-large-v3", "ru": "whisper-large-v3", "hy": "seamless-m4t-v2-large"},
      "default_engine": "whisper-large-v3"
    }

Semantics (mirrors the Deepgram API contract):
- `model=<supported value>`  -> that engine is honored.
- `model=<unsupported value>` (e.g. `nova-2`) -> fallback_by_language[lang].
- `language` missing/auto     -> default_engine.

Every engine in `engines` is kept warm.
"""

import dataclasses
import json
import logging

logger = logging.getLogger(__name__)

# Built-in default: seamless for Armenian, whisper large-v3 otherwise.
DEFAULT_ROUTING = {
    "engines": {
        "whisper-large-v3": {"kind": "whisper", "model": "ggml-large-v3"},
        # HF repo id + revision pinned (S615): reproducible weights for fp16.
        "seamless-m4t-v2-large": {
            "kind": "seamless",
            "model": "facebook/seamless-m4t-v2-large",
            "revision": "5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
        },
    },
    "supported_models": {
        "whisper-large-v3": "whisper-large-v3",
        "large-v3": "whisper-large-v3",
        "whisper": "whisper-large-v3",
        "seamless-m4t-v2-large": "seamless-m4t-v2-large",
        "seamless": "seamless-m4t-v2-large",
    },
    "fallback_by_language": {
        "en": "whisper-large-v3",
        "ru": "whisper-large-v3",
        "hy": "seamless-m4t-v2-large",
    },
    "default_engine": "whisper-large-v3",
}


@dataclasses.dataclass(frozen=True)
class EngineSpec:
    """One warm engine: whisper (whisper-server subprocess) or seamless."""

    name: str
    kind: str  # "whisper" | "seamless"
    model: str
    revision: str = ""  # HF revision pin for seamless models


@dataclasses.dataclass(frozen=True)
class RoutingConfig:
    engines: dict  # engine name -> EngineSpec
    supported_models: dict  # API `model` value -> engine name
    fallback_by_language: dict  # whisper language code -> engine name
    default_engine: str

    def resolve(self, model: str | None, language: str) -> str:
        """Pick the engine name for an API call.

        A supported model request wins; an unsupported one (nova-2 etc.)
        routes by language; unknown languages fall back to the default.
        """
        if model and model in self.supported_models:
            return self.supported_models[model]
        return self.fallback_by_language.get(language, self.default_engine)

    def engine_specs(self) -> list[EngineSpec]:
        """All engines that must be kept warm, in stable order."""
        return [
            EngineSpec(
                name=name,
                kind=spec["kind"],
                model=spec["model"],
                revision=spec.get("revision", ""),
            )
            for name, spec in self.engines.items()
        ]


def load_routing(config_path: str | None) -> RoutingConfig:
    """Build the routing table from defaults plus an optional JSON override.

    The override replaces the whole table, so deployments can remap engines
    without code changes.
    """
    table = DEFAULT_ROUTING
    if config_path:
        with open(config_path, encoding="utf-8") as config_file:
            table = json.load(config_file)
        logger.info("Loaded routing config from %s", config_path)
    config = RoutingConfig(
        engines=dict(table["engines"]),
        supported_models=dict(table.get("supported_models", {})),
        fallback_by_language=dict(table.get("fallback_by_language", {})),
        default_engine=table["default_engine"],
    )
    _validate(config)
    return config


def _validate(config: RoutingConfig) -> None:
    """Fail fast on dangling references instead of at connection time."""
    if config.default_engine not in config.engines:
        raise ValueError(f"default_engine {config.default_engine!r} is not a defined engine")
    for target in (*config.fallback_by_language.values(), *config.supported_models.values()):
        if target not in config.engines:
            raise ValueError(f"routing references undefined engine {target!r}")
    for spec in config.engines.values():
        if spec["kind"] not in ("whisper", "seamless"):
            raise ValueError(f"unknown engine kind {spec['kind']!r}")
