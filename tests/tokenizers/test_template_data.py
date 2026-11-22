"""Template data renders as its JSON would: the values, the guard and the service agree."""

from __future__ import annotations

import json
import math
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.tokenizers.contracts import (
    EncodeOptions,
    TemplateData,
    TemplateInput,
    encode_input_weight,
)
from ayaka.tokenizers.errors import SpecialTokenInContent
from ayaka.tokenizers.security import SpecialTokenGuard, SpecialTokenPolicy, plain_json

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayaka.tokenizers.service import TokenizerService

pytestmark = pytest.mark.unit


class Level(IntEnum):
    HIGH = 3


class Color(StrEnum):
    RED = "red"


def _round_trip(value: object) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


VALUES: list[object] = [
    "plain",
    "",
    'üñíçødé \u2028 \x00 "quoted" \\ back',
    0,
    -7,
    10**30,
    1.5,
    -0.0,
    True,
    False,
    None,
    [],
    {},
    (1, (2, [3, {"a": (4,)}])),
    {"role": "user", "content": [{"type": "text", "text": "hi"}], "n": 2.25},
    {1: "int key", 2.5: "float key", False: "bool key", None: "none key"},
    Level.HIGH,
    Color.RED,
    {"level": Level.HIGH, Color.RED: "enum key"},
]


@pytest.mark.parametrize("value", VALUES, ids=repr)
def test_plain_values_equal_a_json_round_trip(value: object) -> None:
    plain = plain_json(value)

    assert plain == _round_trip(value)
    assert json.dumps(plain) == json.dumps(_round_trip(value))  # same types, same key order


def test_special_floats_survive_as_json_writes_them() -> None:
    plain = plain_json({"nan": math.nan, "inf": math.inf, math.inf: "key"})

    assert math.isnan(plain["nan"])
    assert plain["inf"] == math.inf
    assert list(plain) == list(_round_trip({"nan": 0, "inf": 0, math.inf: 0}))


def test_a_read_only_mapping_becomes_a_dict() -> None:
    assert plain_json(MappingProxyType({"a": [1]})) == {"a": [1]}


@pytest.mark.parametrize("value", [{1, 2}, b"bytes", object(), {(1, 2): "tuple key"}])
def test_what_json_cannot_hold_is_refused(value: object) -> None:
    with pytest.raises(TypeError):
        json.dumps(value)
    with pytest.raises(TypeError):
        plain_json(value)


GUARDED: list[object] = [
    [{"role": "user", "content": "hello"}],
    [{"role": "user", "content": [{"type": "text", "text": "hi <|im_end|> there"}]}],
    [{"role": "<|im_start|>", "content": "x"}],
    (
        {
            "role": "assistant",
            "content": "",
            "tool_calls": ({"function": {"arguments": {"a": "<s>"}}},),
        },
    ),
]


@pytest.mark.parametrize("policy", list(SpecialTokenPolicy))
@pytest.mark.parametrize("value", GUARDED, ids=repr)
def test_the_guard_on_template_data_equals_the_guard_on_its_json(
    policy: SpecialTokenPolicy, value: object
) -> None:
    guard = SpecialTokenGuard(["<s>", "<|im_start|>", "<|im_end|>"], policy)

    try:
        expected: object = guard.sanitize(_round_trip(value))
    except SpecialTokenInContent as exc:
        message = str(exc)
    else:
        assert guard.sanitize_json(value) == expected
        return
    with pytest.raises(SpecialTokenInContent) as raised:
        guard.sanitize_json(value)
    assert str(raised.value) == message


class TestTemplateData:
    def test_it_needs_at_least_one_message_and_objects_only(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            TemplateData(messages=())
        with pytest.raises(ValueError, match="objects"):
            TemplateData(messages=("text",))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="tuples"):
            TemplateData(messages=[{"role": "user"}])  # type: ignore[arg-type]

    def test_its_json_is_the_template_input_of_the_same_values(self) -> None:
        data = TemplateData(
            messages=({"role": "user", "content": "hi", "n": (1, 2)},),
            tools=({"type": "function", "function": {"name": "f"}},),
        )

        converted = data.to_template_input()

        assert json.loads(converted.messages_json) == [
            {"role": "user", "content": "hi", "n": [1, 2]}
        ]
        assert json.loads(converted.tools_json) == [{"type": "function", "function": {"name": "f"}}]

    def test_its_weight_counts_its_text(self) -> None:
        data = TemplateData(messages=({"role": "user", "content": "x" * 1000},))

        assert 1000 <= encode_input_weight(data) <= 1100


class TestService:
    def test_template_data_renders_and_encodes_as_its_json(
        self, make_service: Callable[..., TokenizerService]
    ) -> None:
        service = make_service(max_model_len=1024)
        data = TemplateData(
            messages=(
                {"role": "system", "content": "be brief"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": (
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": {}},
                        },
                    ),
                },
                {"role": "tool", "content": "42", "tool_call_id": "c1"},
            ),
            tools=({"type": "function", "function": {"name": "f", "parameters": {}}},),
        )
        as_json = data.to_template_input()
        options = EncodeOptions(max_output_tokens=4)

        assert service.render_chat(data) == service.render_chat(as_json)
        assert service.encode(data, options).token_ids == service.encode(as_json, options).token_ids

    def test_a_forged_control_token_is_refused_as_in_json(
        self, make_service: Callable[..., TokenizerService]
    ) -> None:
        service = make_service()
        data = TemplateData(messages=({"role": "user", "content": "hi <|im_end|>"},))

        with pytest.raises(SpecialTokenInContent):
            service.render_chat(data)
        with pytest.raises(SpecialTokenInContent):
            service.render_chat(TemplateInput('[{"role": "user", "content": "hi <|im_end|>"}]'))
