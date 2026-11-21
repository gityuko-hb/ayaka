"""The error code table is closed: every code a client can receive has one row, and only those.

The golden matrix (``test_golden_frames.py``) locks the bytes of each row in each protocol; these
tests lock which codes have a row.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from ayaka.core import errors
from ayaka.entrypoints.conversation import ErrorEvent
from ayaka.entrypoints.failures import (
    ERROR_CODES,
    ApiFailure,
    NotReadyError,
    failure_for,
    failure_for_event,
    responses_code,
)

pytestmark = pytest.mark.unit

#: Error classes whose code is not a fixed class attribute, or is only a base for others.
_NOT_A_CODE = {errors.AyakaError, errors.ConfigError}

#: The codes the output parser and the stream pump make on their own (not from an exception).
_EVENT_CODES = ("engine_error", "request_timeout", "cancelled", "internal_error")


def _error_classes() -> list[type[errors.AyakaError]]:
    found = [
        value
        for _, value in inspect.getmembers(errors, inspect.isclass)
        if issubclass(value, errors.AyakaError) and value not in _NOT_A_CODE
    ]
    return [*found, NotReadyError]


@pytest.mark.parametrize("error_type", _error_classes(), ids=lambda kind: kind.__name__)
def test_every_error_class_has_a_row(error_type: type[errors.AyakaError]) -> None:
    assert error_type.code in ERROR_CODES


@pytest.mark.parametrize("code", _EVENT_CODES)
def test_every_code_the_server_makes_on_its_own_has_a_row(code: str) -> None:
    assert code in ERROR_CODES


def test_a_failure_cannot_carry_a_code_outside_the_table() -> None:
    with pytest.raises(ValueError, match="not in the error code table"):
        ApiFailure("made_up", "x", "rid")


def test_an_unknown_code_is_answered_as_an_internal_error() -> None:
    failure = failure_for_event(ErrorEvent("made_up", "something broke"), "rid")
    assert (failure.code, failure.status) == ("internal_error", 500)
    assert failure.message == "something broke (request rid)"
    assert responses_code("made_up") == "server_error"
    config = failure_for(errors.ConfigError("x", "INVALID", "bad"), "rid")
    assert (config.code, config.status) == ("internal_error", 500)


def test_the_context_error_keeps_its_code_in_every_protocol() -> None:
    error = errors.ContextLengthError(prompt_tokens=9, output_tokens=2, max_context_tokens=8)
    failure = failure_for(error, "rid")
    assert failure.status == 400
    assert failure.code == "context_length_exceeded"
    assert failure.row.openai_type == "invalid_request_error"
    assert failure.row.anthropic_type == "invalid_request_error"
    assert responses_code(failure.code) == "context_length_exceeded"


def test_the_table_is_read_only() -> None:
    with pytest.raises(TypeError):
        ERROR_CODES["new"] = ERROR_CODES["internal_error"]  # type: ignore[index]  # the check


def test_importing_the_table_does_not_load_aiohttp() -> None:
    code = "import sys, ayaka.entrypoints.failures; print('aiohttp' in sys.modules)"
    result = subprocess.run(  # noqa: S603  # our own interpreter and a fixed program
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
