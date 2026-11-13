from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from ayaka.core.errors import (
    AyakaError,
    BackendError,
    CancelledError,
    CapabilityError,
    CapacityError,
    ConfigError,
    EngineDeadError,
    EngineError,
    ErrorInfo,
    InvalidRequestError,
    ModelLoadError,
    OverloadedError,
    RequestAborted,
    RequestTimeout,
    UnsupportedFeatureError,
)

pytestmark = pytest.mark.unit


def _round_trip[E: AyakaError](error: E) -> E:
    """Deserialize our own freshly pickled error; the payload is trusted test data."""
    return cast("E", pickle.loads(pickle.dumps(error)))  # noqa: S301


class TestErrorInfo:
    def test_defaults(self) -> None:
        info = ErrorInfo(code="bad", message="something broke")

        assert info.code == "bad"
        assert info.message == "something broke"
        assert info.retryable is False

    def test_is_frozen(self) -> None:
        info = ErrorInfo(code="bad", message="something broke")

        with pytest.raises(FrozenInstanceError):
            cast("Any", info).code = "other"


class TestAyakaError:
    def test_defaults_and_message(self) -> None:
        error = AyakaError("boom")

        assert error.message == "boom"
        assert str(error) == "boom"
        assert error.args == ("boom",)
        assert error.code == "ayaka_error"
        assert error.retryable is False
        assert error.context == {}

    def test_context_is_copied(self) -> None:
        context: dict[str, object] = {"request_id": "req-1"}
        error = AyakaError("boom", context=context)

        assert error.context == {"request_id": "req-1"}
        assert error.context is not context

        context["late"] = True
        assert "late" not in error.context

    def test_to_info(self) -> None:
        error = AyakaError("boom", context={"request_id": "req-1"})

        assert error.to_info() == ErrorInfo(
            code="ayaka_error",
            message="boom",
            retryable=False,
        )

    def test_pickle_round_trip_preserves_context(self) -> None:
        error = AyakaError("boom", context={"request_id": "req-1", "attempt": 2})

        restored = _round_trip(error)

        assert type(restored) is AyakaError
        assert restored.message == "boom"
        assert restored.args == ("boom",)
        assert restored.context == {"request_id": "req-1", "attempt": 2}


class TestConfigError:
    def test_fields_and_message(self) -> None:
        error = ConfigError(
            "model.path",
            "bad_value",
            "must be positive",
            context={"value": 0},
        )

        assert error.path == "model.path"
        assert error.code == "bad_value"
        assert error.message == "must be positive"
        assert error.context == {"value": 0}
        assert str(error) == "model.path: must be positive [bad_value]"
        assert error.args == ("model.path: must be positive [bad_value]",)
        assert isinstance(error, ValueError)
        assert isinstance(error, AyakaError)
        assert error.to_info() == ErrorInfo("bad_value", "must be positive", False)

    def test_context_is_copied(self) -> None:
        context = {"value": 0}
        error = ConfigError("path", "code", "message", context=context)

        context["late"] = True
        assert "late" not in error.context

    def test_pickle_round_trip(self) -> None:
        error = ConfigError("model.path", "bad_value", "must be positive", context={"value": 0})

        restored = _round_trip(error)

        assert type(restored) is ConfigError
        assert restored.path == "model.path"
        assert restored.code == "bad_value"
        assert restored.message == "must be positive"
        assert restored.context == {"value": 0}
        assert str(restored) == str(error)


class TestCapabilityError:
    def test_fields_and_message(self) -> None:
        error = CapabilityError(
            "flash-attn",
            detail="package is not installed",
            remedy="pip install flash-attn",
        )

        assert error.capability == "flash-attn"
        assert error.detail == "package is not installed"
        assert error.remedy == "pip install flash-attn"
        assert error.code == "capability_unavailable"
        assert error.retryable is False
        assert isinstance(error, ImportError)
        assert str(error) == (
            "capability 'flash-attn' unavailable: package is not installed. "
            "Remedy: pip install flash-attn"
        )

    def test_to_info(self) -> None:
        error = CapabilityError("flash-attn", detail="missing", remedy="install it")

        assert error.to_info() == ErrorInfo(
            "capability_unavailable",
            str(error),
            False,
        )

    def test_pickle_round_trip(self) -> None:
        error = CapabilityError("flash-attn", detail="missing", remedy="install it")

        restored = _round_trip(error)

        assert type(restored) is CapabilityError
        assert restored.capability == "flash-attn"
        assert restored.detail == "missing"
        assert restored.remedy == "install it"
        assert str(restored) == str(error)


class TestCancelledError:
    def test_default_reason(self) -> None:
        error = CancelledError()

        assert error.reason == "cancelled"
        assert error.message == "cancelled"
        assert error.code == "cancelled"
        assert isinstance(error, RuntimeError)

    def test_custom_reason(self) -> None:
        error = CancelledError("client disconnected")

        assert error.reason == "client disconnected"
        assert str(error) == "client disconnected"

    def test_pickle_round_trip(self) -> None:
        error = CancelledError("client disconnected")

        restored = _round_trip(error)

        assert type(restored) is CancelledError
        assert restored.reason == "client disconnected"
        assert restored.message == "client disconnected"


@pytest.mark.parametrize(
    ("error_type", "code", "retryable", "builtin"),
    [
        (InvalidRequestError, "invalid_request", False, ValueError),
        (UnsupportedFeatureError, "unsupported_feature", False, ValueError),
        (CapacityError, "capacity_exceeded", True, RuntimeError),
        (OverloadedError, "overloaded", True, RuntimeError),
        (EngineError, "engine_error", False, RuntimeError),
        (EngineDeadError, "engine_dead", False, RuntimeError),
        (RequestAborted, "request_aborted", False, RuntimeError),
        (RequestTimeout, "request_timeout", True, RuntimeError),
        (ModelLoadError, "model_load_error", False, RuntimeError),
        (BackendError, "backend_error", False, RuntimeError),
    ],
)
def test_simple_error_types(
    error_type: type[AyakaError],
    code: str,
    retryable: bool,
    builtin: type[BaseException],
) -> None:
    error = error_type("boom", context={"request_id": "req-1"})

    assert isinstance(error, AyakaError)
    assert isinstance(error, builtin)
    assert error.message == "boom"
    assert error.code == code
    assert error.retryable is retryable
    assert error.context == {"request_id": "req-1"}
    assert error.to_info() == ErrorInfo(code, "boom", retryable)

    restored = _round_trip(error)
    assert type(restored) is error_type
    assert restored.message == "boom"
    assert restored.code == code
    assert restored.context == {"request_id": "req-1"}


def test_error_hierarchy_relationships() -> None:
    assert issubclass(UnsupportedFeatureError, InvalidRequestError)
    assert issubclass(OverloadedError, CapacityError)
    assert issubclass(EngineDeadError, EngineError)
    assert issubclass(ConfigError, ValueError)
    assert issubclass(CancelledError, RuntimeError)
    assert issubclass(CapabilityError, ImportError)
