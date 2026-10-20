"""Structured single-line JSON logging with trace-id context."""

import json
import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def set_trace_id(trace_id: str) -> Token:
    """Tag every log record emitted from this context (e.g. the client session id)."""
    return _trace_id.set(trace_id)


class TraceContextFilter(logging.Filter):
    """Attach the current trace id (or None) to every record passing through."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Mutate the record in place; always process it."""
        record.trace = _trace_id.get() or None
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: parseable and stitchable across services."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Render one record as a single-line JSON object."""
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        trace = getattr(record, "trace", None)
        if trace:
            entry["trace"] = trace
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Replace root handlers with the JSON formatter."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(TraceContextFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
