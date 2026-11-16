"""Process logging setup for the CLI, engine and entrypoints.

:func:`init_logger` installs one handler on a logger and selects either a
human-readable text formatter or a JSON formatter that emits one object per
line. :func:`log_once` drops repeated identical diagnostics, so a fallback or a
startup warning appears exactly once per process.

The module is deliberately small and dependency-free: importing it must never
load torch, NVML or any other optional package.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any, TextIO, override

__all__ = ["JsonFormatter", "TextFormatter", "init_logger", "log_once"]

_HANDLER_MARKER = "_ayaka_handler"
_INIT_LOCK = threading.Lock()
_ONCE_LOCK = threading.Lock()
_LOGGED_ONCE: set[tuple[str, int, str]] = set()


def _format_timestamp(created: float) -> str:
    """Render a record's ``created`` value as UTC ISO 8601 with milliseconds."""
    return (
        datetime.fromtimestamp(created, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class TextFormatter(logging.Formatter):
    """Human-readable lines: ``<UTC timestamp> <LEVEL> <logger>: <message>``."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Format the record time in UTC so logs from different hosts line up."""
        return _format_timestamp(record.created)


class JsonFormatter(logging.Formatter):
    """One JSON object per record, carrying the fields the text format shows."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": _format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _resolve_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        raise ValueError(f"unknown log level: {level!r}")
    return resolved


def init_logger(
    name: str = "ayaka",
    *,
    level: int | str = logging.INFO,
    json_format: bool = False,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure ``name`` with a single text or JSON handler and return it.

    Calling this again for the same logger replaces the handler installed here
    instead of adding a second one, so a re-initialised process never prints
    every line twice. The logger stops propagating so the root logger cannot
    duplicate the output; other handlers already attached by the caller are
    left alone.

    Args:
        name: Logger name; the default covers every ``ayaka.*`` logger.
        level: Threshold as a numeric level or a case-insensitive name.
        json_format: Emit one JSON object per line instead of text.
        stream: Destination of the handler; defaults to ``sys.stderr``.

    Raises:
        ValueError: If ``level`` is a string that is not a known level name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level(level))
    formatter: logging.Formatter = JsonFormatter() if json_format else TextFormatter()

    with _INIT_LOCK:
        for handler in tuple(logger.handlers):
            if getattr(handler, _HANDLER_MARKER, False):
                logger.removeHandler(handler)
                handler.close()
        handler = logging.StreamHandler(stream)
        setattr(handler, _HANDLER_MARKER, True)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def log_once(
    logger: logging.Logger,
    level: int,
    msg: str,
    *args: object,
    **kwargs: Any,
) -> bool:
    """Log a message only the first time this logger emits it at this level.

    The key is the fully formatted message, so two calls that produce the same
    text (with the same or different arguments) count as one. Formatting
    keywords such as ``exc_info`` are forwarded to :meth:`logging.Logger.log`.

    Args:
        logger: Logger to write through.
        level: Numeric logging level.
        msg: ``%``-style format string.
        *args: Values for ``msg``.
        **kwargs: Extra arguments forwarded to :meth:`logging.Logger.log`.

    Returns:
        ``True`` if this call emitted the message, ``False`` if an earlier call
        with the same logger, level and text already did.
    """
    try:
        formatted = msg % args if args else msg
    except (TypeError, ValueError):
        formatted = repr((msg, args))
    key = (logger.name, level, formatted)

    with _ONCE_LOCK:
        if key in _LOGGED_ONCE:
            return False
        _LOGGED_ONCE.add(key)

    logger.log(level, msg, *args, **kwargs)
    return True
