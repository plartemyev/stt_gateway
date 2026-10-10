from stt_gateway.engine import parse_transcription

# Recorded from a real whisper-server verbose_json response (sine tone probe).
PROBE_PAYLOAD = {
    "task": "transcribe",
    "language": "english",
    "duration": 2.0,
    "text": " [Bell]\n",
    "segments": [
        {
            "id": 0,
            "text": " [Bell]",
            "start": 0.0,
            "end": 2.0,
            "tokens": [542, 33, 898, 60],
            "words": [
                {"word": " [", "start": 0.0, "end": 0.33, "t_dtw": -1, "probability": 0.6},
                {"word": "B", "start": 0.33, "end": 0.66, "t_dtw": -1, "probability": 0.99},
            ],
            "temperature": 0.0,
            "avg_logprob": -0.08555618673563004,
            "no_speech_prob": 5.00838154948724e-07,
        }
    ],
    "detected_language": "english",
}


def test_parse_transcription_from_real_payload():
    transcription = parse_transcription(PROBE_PAYLOAD)
    assert transcription.text == "[Bell]"
    assert transcription.language == "english"
    assert transcription.duration == 2.0
    # One Deepgram word per whitespace-separated token of the segment text.
    assert [word.text for word in transcription.words] == ["[Bell]"]
    assert transcription.words[0].start == 0.0
    assert abs(transcription.words[0].end - 2.0) < 1e-6
    # exp(avg_logprob) clipped to [0, 1].
    assert abs(transcription.words[0].confidence - 0.917966) < 1e-4


def test_parse_transcription_empty_payload():
    transcription = parse_transcription({"text": ""})
    assert transcription.text == ""
    assert transcription.words == []


def test_parse_transcription_multiple_segments():
    payload = {
        "text": "a b",
        "duration": 4.0,
        "segments": [
            {"text": " a", "start": 0.0, "end": 1.0, "avg_logprob": -0.1},
            {"text": " b", "start": 1.0, "end": 4.0, "avg_logprob": -0.2},
        ],
    }
    transcription = parse_transcription(payload)
    assert [word.text for word in transcription.words] == ["a", "b"]
    assert transcription.words[1].start == 1.0
    assert transcription.words[1].end == 4.0
