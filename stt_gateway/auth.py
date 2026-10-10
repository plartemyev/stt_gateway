"""Deepgram-style token authentication.

Any string is accepted as a token value, bounded by a configurable maximum
length. A missing or empty token is rejected, mirroring the real API.
"""

from collections.abc import Mapping

_AUTH_SCHEMES = ("token", "bearer")


def extract_token(headers: Mapping[str, str]) -> str | None:
    """Extract the token from an `Authorization` header value.

    Accepts both `Token <value>` (Deepgram style) and `Bearer <value>`.
    Returns None when the header is absent, malformed or empty.
    """
    value = ""
    for name, header_value in headers.items():
        if name.lower() == "authorization":
            value = header_value
            break
    scheme, _, rest = value.partition(" ")
    if scheme.lower() not in _AUTH_SCHEMES:
        return None
    token = rest.strip()
    return token if token else None


def is_valid(token: str | None, max_len: int) -> bool:
    """Return True when a token is present and within the length cap."""
    return token is not None and 0 < len(token) <= max_len
