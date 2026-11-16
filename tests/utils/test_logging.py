from __future__ import annotations

import io
import json
import logging
import re
import sys
from typing import Any

import pytest

from ayaka.utils.logging import JsonFormatter, TextFormatter, init_logger, log_once

pytestmark = pytest.mark.unit

_TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")


def _record(
    msg: str = "message",
    *,
    name: str = "ayaka.test",
    level: int = logging.INFO,
    args: tuple[object, ...] = (),
    exc_info: Any = None,
) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, args, exc_info)


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _collecting_logger(name: str) -> tuple[logging.Logger, list[logging.LogRecord]]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = _CollectingHandler()
    logger.addHandler(handler)
    return logger, handler.records


class TestTextFormatter:
    def test_renders_timestamp_level_logger_and_message(self) -> None:
        line = TextFormatter().format(_record("loaded %d shards", args=(3,)))

        match = re.fullmatch(rf"{_TIMESTAMP.pattern} INFO\s+ayaka\.test: loaded 3 shards", line)

        assert match is not None

    def test_includes_the_traceback(self) -> None:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            record = _record("failed", exc_info=sys.exc_info())

        line = TextFormatter().format(record)

        assert "failed" in line
        assert "Traceback (most recent call last):" in line
        assert "RuntimeError: boom" in line


class TestJsonFormatter:
    def test_emits_one_object_per_record_with_the_text_fields(self) -> None:
        output = JsonFormatter().format(_record("loaded %s", args=("gpt2",)))

        payload = json.loads(output)

        assert "\n" not in output
        assert _TIMESTAMP.fullmatch(str(payload["timestamp"]))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "ayaka.test"
        assert payload["message"] == "loaded gpt2"

    def test_includes_the_traceback_in_a_field(self) -> None:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            record = _record("failed", exc_info=sys.exc_info())

        payload = json.loads(JsonFormatter().format(record))

        assert "RuntimeError: boom" in payload["exception"]


class TestInitLogger:
    def test_writes_text_to_the_given_stream(self) -> None:
        stream = io.StringIO()
        logger = init_logger("ayaka.test.text", stream=stream)

        logger.info("loaded %s", "gpt2")

        assert "INFO" in stream.getvalue()
        assert "ayaka.test.text: loaded gpt2" in stream.getvalue()
        assert logger.propagate is False

    def test_writes_json_to_the_given_stream(self) -> None:
        stream = io.StringIO()
        logger = init_logger("ayaka.test.json", json_format=True, stream=stream)

        logger.warning("fallback to %s", "torch")

        payload = json.loads(stream.getvalue().strip())
        assert payload["level"] == "WARNING"
        assert payload["logger"] == "ayaka.test.json"
        assert payload["message"] == "fallback to torch"

    def test_reinitialising_replaces_its_own_handler(self) -> None:
        first, second = io.StringIO(), io.StringIO()
        logger = init_logger("ayaka.test.once", stream=first)

        init_logger("ayaka.test.once", stream=second)
        logger.info("one line")

        assert len(logger.handlers) == 1
        assert first.getvalue() == ""
        assert second.getvalue().count("one line") == 1

    def test_respects_the_level(self) -> None:
        stream = io.StringIO()
        logger = init_logger("ayaka.test.level", level="WARNING", stream=stream)

        logger.info("hidden")
        logger.warning("shown")

        assert "hidden" not in stream.getvalue()
        assert "shown" in stream.getvalue()

    def test_rejects_an_unknown_level(self) -> None:
        with pytest.raises(ValueError, match="unknown log level"):
            init_logger("ayaka.test.bad-level", level="LOUD")


class TestLogOnce:
    def test_drops_a_repeated_message(self) -> None:
        logger, records = _collecting_logger("ayaka.test.log-once.repeat")

        first = log_once(logger, logging.WARNING, "using backup backend %s", "torch")
        second = log_once(logger, logging.WARNING, "using backup backend %s", "torch")

        assert first is True
        assert second is False
        assert [record.getMessage() for record in records] == ["using backup backend torch"]

    def test_keeps_distinct_text_and_level(self) -> None:
        logger, records = _collecting_logger("ayaka.test.log-once.distinct")

        assert log_once(logger, logging.WARNING, "using backup backend %s", "torch") is True
        assert log_once(logger, logging.WARNING, "using backup backend %s", "kops") is True
        assert log_once(logger, logging.INFO, "using backup backend %s", "torch") is True
        assert len(records) == 3

    def test_forwards_logging_keywords(self) -> None:
        logger, records = _collecting_logger("ayaka.test.log-once.exc")

        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logged = log_once(logger, logging.ERROR, "step failed", exc_info=True)

        assert logged is True
        assert records[0].exc_info is not None
