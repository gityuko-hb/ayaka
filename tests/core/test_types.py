from __future__ import annotations

import pytest

from ayaka.core.types import (
    FinishReason,
    ForwardMode,
    KVLayoutKind,
    Layout,
    RequestId,
)

pytestmark = pytest.mark.unit


def test_request_id_is_a_string_newtype() -> None:
    request_id = RequestId("req-1")

    assert request_id == "req-1"
    assert isinstance(request_id, str)


def test_layout_values() -> None:
    assert {mode.value for mode in Layout} == {
        "row_major",
        "col_major",
        "packed_sub_byte",
        "blocked",
    }
    assert Layout.ROW_MAJOR == "row_major"
    assert isinstance(Layout.ROW_MAJOR, str)


def test_kv_layout_kind_values() -> None:
    assert {kind.value for kind in KVLayoutKind} == {"nhd", "hnd", "NLD"}


def test_finish_reason_values() -> None:
    assert {reason.value for reason in FinishReason} == {
        "stop",
        "length",
        "cancelled",
        "timeout",
        "error",
    }


@pytest.mark.parametrize(
    ("mode", "decode_like", "prefill_like"),
    [
        (ForwardMode.PREFILL, False, True),
        (ForwardMode.DECODE, True, False),
        (ForwardMode.MIXED, False, True),
        (ForwardMode.IDLE, False, False),
        (ForwardMode.VERIFY, True, False),
    ],
)
def test_forward_mode_classification(
    mode: ForwardMode, decode_like: bool, prefill_like: bool
) -> None:
    assert mode.is_decode_like is decode_like
    assert mode.is_prefill_like is prefill_like


def test_forward_mode_values() -> None:
    assert {mode.value for mode in ForwardMode} == {
        "prefill",
        "decode",
        "mixed",
        "idle",
        "verify",
    }
