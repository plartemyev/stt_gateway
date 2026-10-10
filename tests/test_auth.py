from stt_gateway.auth import extract_token, is_valid


def test_extracts_deepgram_style_token():
    assert extract_token({"Authorization": "Token my-key-123"}) == "my-key-123"


def test_extracts_bearer_token():
    assert extract_token({"authorization": "Bearer abc"}) == "abc"


def test_missing_or_malformed_header_returns_none():
    assert extract_token({}) is None
    assert extract_token({"Authorization": "Basic dXNlcjpwYXNz"}) is None
    assert extract_token({"Authorization": "Token"}) is None
    assert extract_token({"Authorization": "Token   "}) is None


def test_any_nonempty_string_within_cap_is_valid():
    assert is_valid("a", max_len=256)
    assert is_valid("❤ поддержанные языки", max_len=256)
    assert is_valid("x" * 256, max_len=256)


def test_oversized_and_empty_tokens_are_invalid():
    assert not is_valid("x" * 257, max_len=256)
    assert not is_valid(None, max_len=256)
    assert not is_valid("", max_len=256)
