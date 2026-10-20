"""Deepgram streaming `Results` message construction.

The gateway emits messages shaped exactly like Deepgram's live streaming
responses (see https://developers.deepgram.com/docs/lower-level-websockets):

    {
      "type": "Results",
      "channel_index": [0, 2],
      "duration": 4.0, "start": 12.0,
      "is_final": true, "speech_final": true,
      "channel": {"alternatives": [{"transcript": ..., "confidence": ..., "words": [...]}]},
      "metadata": {"request_id": ..., "model_info": {...}, "channels": 2}
    }

The pinned reference consumer reads `channel.alternatives[0].transcript` only; the remaining
fields are kept for protocol fidelity with other Deepgram clients.
"""

import dataclasses
import statistics
import uuid

# Sentence-ending marks across the supported languages (Latin, Cyrillic and
# Armenian included); `punctuate=true` guarantees one per utterance end.
TERMINAL_PUNCTUATION = set(".!?；;。！？։՝۔")


def ensure_terminal_punctuated(text: str) -> str:
    """Return text guaranteed to end with sentence-ending punctuation.

    Deepgram's `punctuate=true` closes every utterance with a mark; engines
    with language-dependent punctuation (SeamlessM4T) get the same guarantee
    through this rule.
    """
    stripped = text.strip()
    if not stripped or stripped[-1] in TERMINAL_PUNCTUATION:
        return stripped
    return stripped + "."


@dataclasses.dataclass(frozen=True)
class Word:
    """One transcribed word with absolute (stream-relative) timing."""

    text: str
    start: float
    end: float
    confidence: float


def results_message(
        *,
        words: list[Word],
        transcript: str,
        language: str,
        start: float,
        duration: float,
        channels: int,
        model_name: str,
        is_final: bool = True,
        speech_final: bool = True,
        timings: dict | None = None,
) -> dict:
    """Build a Deepgram streaming `Results` message.

    `timings`, when given, is merged into `metadata` (non-standard extension
    carrying gateway-side latency measurements, e.g. {"infer_s": 0.123}).
    """
    confidence = statistics.fmean(w.confidence for w in words) if words else 0.0
    alternative_words = [
        {
            "word": word.text,
            "start": round(word.start, 6),
            "end": round(word.end, 6),
            "confidence": round(word.confidence, 6),
            # Whisper output is punctuated/capitalized natively; the Deepgram
            # `punctuate=true` behavior is therefore always on.
            "punctuated_word": word.text,
        }
        for word in words
    ]
    metadata = {
        "request_id": str(uuid.uuid4()),
        "model_info": {"name": model_name, "version": "", "arch": "whisper"},
        "channels": channels,
        "language": language,
    }
    if timings:
        metadata["timings"] = timings
    return {
        "type": "Results",
        "channel_index": [0, channels],
        "duration": round(duration, 6),
        "start": round(start, 6),
        "is_final": is_final,
        "speech_final": speech_final,
        "from_finalize": False,
        "channel": {
            "alternatives": [
                {
                    "transcript": transcript,
                    "confidence": round(confidence, 6),
                    "words": alternative_words,
                }
            ],
        },
        "metadata": metadata,
    }
