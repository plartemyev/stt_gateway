import pytest

from stt_gateway.routing import DEFAULT_ROUTING, RoutingConfig, load_routing


def test_default_routing_keeps_two_engines_warm():
    config = load_routing(None)
    assert set(config.engines) == {"whisper-large-v3", "seamless-m4t-v2-large"}
    assert len(config.engine_specs()) == 2


def test_supported_model_request_is_honored():
    config = load_routing(None)
    assert config.resolve("seamless-m4t-v2-large", "en") == "seamless-m4t-v2-large"
    assert config.resolve("whisper-large-v3", "hy") == "whisper-large-v3"
    assert config.resolve("large-v3", "ru") == "whisper-large-v3"


def test_unsupported_model_routes_by_language():
    config = load_routing(None)
    assert config.resolve("nova-2", "hy") == "seamless-m4t-v2-large"
    assert config.resolve("nova-2", "en") == "whisper-large-v3"
    assert config.resolve("nova-2", "ru") == "whisper-large-v3"


def test_missing_language_falls_back_to_default_whisper():
    config = load_routing(None)
    assert config.resolve("nova-2", "auto") == "whisper-large-v3"
    assert config.resolve("", "auto") == "whisper-large-v3"


def test_json_override_replaces_whole_table(tmp_path):
    override = tmp_path / "routing.json"
    override.write_text("""
    {
      "engines": {"turbo": {"kind": "whisper", "model": "ggml-large-v3-turbo"}},
      "supported_models": {"nova-2": "turbo"},
      "fallback_by_language": {"hy": "turbo"},
      "default_engine": "turbo"
    }
    """)
    config = load_routing(str(override))
    assert config.resolve("nova-2", "hy") == "turbo"
    assert [spec.model for spec in config.engine_specs()] == ["ggml-large-v3-turbo"]


def test_dangling_references_fail_fast(tmp_path):
    override = tmp_path / "routing.json"
    override.write_text("""
    {
      "engines": {"whisper-large-v3": {"kind": "whisper", "model": "ggml-large-v3"}},
      "fallback_by_language": {"hy": "missing-engine"},
      "default_engine": "whisper-large-v3"
    }
    """)
    with pytest.raises(ValueError, match="undefined engine"):
        load_routing(str(override))


def test_default_table_shape():
    assert DEFAULT_ROUTING["default_engine"] in DEFAULT_ROUTING["engines"]
    assert isinstance(RoutingConfig, type)
