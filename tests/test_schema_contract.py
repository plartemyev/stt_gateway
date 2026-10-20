"""Contract tests: our Results messages must satisfy the reference consumer.

`tests/deepgram_consumer_snapshot.py` pins the only field path any known
consumer reads (`channel.alternatives[0].transcript` of Deepgram's streaming
message). These tests keep the gateway protocol-compatible with it.
"""

import json

import pytest

from stt_gateway.schema import Word, results_message
from tests.deepgram_consumer_snapshot import process_deepgram_transcript_message


def make_message(**overrides) -> dict:
    params = {
        "words": [Word(text="Hello", start=0.1, end=0.5, confidence=0.9),
                  Word(text="world.", start=0.5, end=0.9, confidence=0.8)],
        "transcript": "Hello world.",
        "language": "en",
        "start": 4.0,
        "duration": 1.0,
        "channels": 2,
        "model_name": "whisper-large-v3",
    }
    params.update(overrides)
    return results_message(**params)


@pytest.mark.asyncio
async def test_results_message_satisfies_reference_consumer():
    message = json.dumps(make_message()).encode()
    assert await process_deepgram_transcript_message(message) == "Hello world."


@pytest.mark.asyncio
async def test_empty_transcript_message_yields_empty_string():
    message = json.dumps(make_message(words=[], transcript="")).encode()
    assert await process_deepgram_transcript_message(message) == ""


def test_message_shape_matches_deepgram_streaming_schema():
    message = make_message()
    assert message["type"] == "Results"
    assert message["channel_index"] == [0, 2]
    assert message["is_final"] is True
    assert message["speech_final"] is True
    channel = message["channel"]["alternatives"][0]
    assert channel["transcript"] == "Hello world."
    assert [word["word"] for word in channel["words"]] == ["Hello", "world."]
    # Word timings are absolute (stream-relative), like Deepgram's.
    assert channel["words"][1]["start"] == 0.5
    assert set(message["metadata"]) >= {"request_id", "model_info", "channels"}


def test_timings_are_exposed_in_metadata():
    message = make_message(timings={"infer_s": 0.123})
    assert message["metadata"]["timings"] == {"infer_s": 0.123}


def test_timings_omitted_when_not_measured():
    message = make_message()
    assert "timings" not in message["metadata"]
