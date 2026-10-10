"""Unit tests for the punctuation rule and the Seamless word approximation."""

import pytest

from stt_gateway.engine import EngineError
from stt_gateway.engines import SEAMLESS_LANGUAGES, SeamlessEngine, _words_from_span
from stt_gateway.schema import ensure_terminal_punctuated


@pytest.mark.parametrize("text", [
    "Hello world.",
    "Really?!",
    "վերջ։",
    "orman ;",
])
def test_terminal_punctuated_text_is_untouched(text):
    assert ensure_terminal_punctuated(text) == text


def test_missing_terminal_punctuation_is_appended():
    assert ensure_terminal_punctuated("Hello world") == "Hello world."
    assert ensure_terminal_punctuated("  spaced  ") == "spaced."


def test_empty_text_is_untouched():
    assert ensure_terminal_punctuated("") == ""


def test_words_from_span_splits_evenly():
    words = _words_from_span("a b c", duration=1.5)
    assert [word.text for word in words] == ["a", "b", "c"]
    assert words[0].start == 0.0
    assert words[2].end == 1.5


def test_words_from_span_empty_text():
    assert _words_from_span("", duration=1.0) == []


def test_seamless_language_map_covers_pipeline_languages():
    assert SEAMLESS_LANGUAGES == {"en": "eng", "ru": "rus", "hy": "hye"}


@pytest.mark.asyncio
async def test_seamless_rejects_auto_language():
    engine = SeamlessEngine("seamless-m4t-v2-large", "facebook/seamless-m4t-v2-large", settings=None)
    with pytest.raises(EngineError, match="concrete language"):
        await engine.transcribe(b"", "auto")
