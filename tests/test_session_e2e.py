"""End-to-end session tests over the real WS stack with the FakeEngine."""

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDisconnect


def test_rejects_missing_token(gateway: TestClient, listen_url):
    with pytest.raises(WebSocketDisconnect):
        with gateway.websocket_connect(listen_url(encoding="linear16", language="en-US")):
            pass


def test_rejects_oversized_token(gateway: TestClient, listen_url):
    with pytest.raises(WebSocketDisconnect):
        with gateway.websocket_connect(
                listen_url(encoding="linear16", language="en-US"),
                headers={"Authorization": "Token " + "x" * 257},
        ):
            pass


def test_accepts_any_reasonable_token(gateway: TestClient, listen_url):
    token = "any-string-123 with spaces"
    with gateway.websocket_connect(
            listen_url(encoding="linear16", sample_rate=16000, channels=1, language="en-US"),
            headers={"Authorization": f"Token {token}"},
    ) as ws:
        assert ws.receive_json()["type"] == "Results"


def test_rejects_unsupported_language(gateway: TestClient, listen_url):
    with pytest.raises(WebSocketDisconnect):
        with gateway.websocket_connect(
                listen_url(encoding="linear16", language="xx-XX"),
                headers={"Authorization": "Token abc"},
        ):
            pass


def test_rejects_unsupported_encoding(gateway: TestClient, listen_url):
    with pytest.raises(WebSocketDisconnect):
        with gateway.websocket_connect(
                listen_url(encoding="flac", language="en"),
                headers={"Authorization": "Token abc"},
        ):
            pass


def test_streaming_session_returns_results(gateway: TestClient, listen_url, sine_seconds, linear16_bytes):
    import numpy as np

    url = listen_url(encoding="linear16", sample_rate=16000, channels=1, language="en-US")
    with gateway.websocket_connect(url, headers={"Authorization": "Token abc"}) as ws:
        hello = ws.receive_json()
        assert hello["type"] == "Results"
        assert hello["is_final"] is False
        # 0.5s of speech + 0.4s silence: stays below the 1s window, so the
        # silence flush (utterance end) fires and the punctuate rule applies.
        audio = linear16_bytes(np.concatenate([sine_seconds(0.5), np.zeros(6_400, dtype=np.float32)]))
        ws.send_bytes(audio)
        result = ws.receive_json()
        assert result["type"] == "Results"
        assert result["is_final"] is True
        assert result["speech_final"] is True
        alternative = result["channel"]["alternatives"][0]
        assert alternative["transcript"]
        assert alternative["words"], "words should carry absolute timestamps"
        assert all(word["end"] > word["start"] for word in alternative["words"])
        assert result["metadata"]["model_info"]["arch"] == "whisper"
        # nova-2 + en routes to the default whisper engine (FakeEngine text has
        # no terminal punctuation; the silence-flushed utterance must gain it).
        assert result["metadata"]["model_info"]["name"] == "whisper-large-v3"
        assert alternative["transcript"].endswith(".")


def test_decode_failure_closes_session_with_internal_error(gateway: TestClient, listen_url):
    def stream_garbage() -> None:
        with gateway.websocket_connect(listen_url(encoding="ogg-opus", language="en"),
                                       headers={"Authorization": "Token abc"}) as ws:
            ws.receive_json()
            ws.send_bytes(b"\x00" * 4096)
            # The gateway must close the session instead of hanging silently;
            # interim (empty) Results may arrive before the close.
            while True:
                ws.receive_json()

    with pytest.raises(WebSocketDisconnect):
        stream_garbage()
