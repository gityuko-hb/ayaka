"""The protocol-independent conversation and the output events."""

from __future__ import annotations

import json

import pytest

from ayaka.core.errors import InvalidRequestError
from ayaka.core.types import FinishReason
from ayaka.entrypoints.conversation import (
    ConversationBuilder,
    ErrorEvent,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolSpec,
    UsageEvent,
    collect,
    merge_deltas,
    parse_arguments,
)

pytestmark = pytest.mark.unit


class TestBuilder:
    def test_leading_system_parts_join_and_later_ones_become_turns(self) -> None:
        builder = ConversationBuilder()
        builder.system("first")
        builder.system("second")
        builder.user("hi")
        builder.system("late")

        conversation = builder.build()

        assert conversation.system == "first\nsecond"
        assert [turn.role for turn in conversation.turns] == ["user", "system"]

    def test_a_tool_call_joins_the_assistant_turn_before_it(self) -> None:
        builder = ConversationBuilder()
        builder.user("weather?")
        builder.assistant("let me look")
        builder.tool_call(ToolCall("c1", "weather", {"city": "Paris"}))
        builder.tool_call(ToolCall("c2", "weather", {"city": "Rome"}))
        builder.tool_result("c1", "sunny")

        turns = builder.build().turns

        assert [turn.role for turn in turns] == ["user", "assistant", "tool"]
        assert turns[1].text == "let me look"
        assert [call.id for call in turns[1].tool_calls] == ["c1", "c2"]
        assert turns[2].name == "weather"
        assert turns[2].tool_call_id == "c1"

    def test_a_tool_call_after_a_result_opens_a_new_assistant_turn(self) -> None:
        builder = ConversationBuilder()
        builder.user("go")
        builder.tool_call(ToolCall("c1", "a", {}))
        builder.tool_result("c1", "done")
        builder.tool_call(ToolCall("c2", "b", {}))

        turns = builder.build().turns

        assert [turn.role for turn in turns] == ["user", "assistant", "tool", "assistant"]
        assert turns[3].text == ""

    def test_a_result_for_an_unknown_call_has_no_name(self) -> None:
        builder = ConversationBuilder()
        builder.tool_result("nowhere", "x")

        [turn] = builder.build().turns

        assert turn.name is None

    def test_duplicates_and_an_empty_conversation_are_refused(self) -> None:
        builder = ConversationBuilder()
        builder.tool_call(ToolCall("c", "a", {}))
        with pytest.raises(InvalidRequestError, match="appears twice"):
            builder.tool_call(ToolCall("c", "a", {}))
        builder.tool(ToolSpec("t", None, {}))
        with pytest.raises(InvalidRequestError, match="defined twice"):
            builder.tool(ToolSpec("t", None, {}))
        with pytest.raises(InvalidRequestError, match="no message"):
            ConversationBuilder().build()


class TestTemplateInput:
    def test_messages_follow_the_hugging_face_layout_and_keep_key_order(self) -> None:
        builder = ConversationBuilder()
        builder.system("be brief")
        builder.user("hi")
        builder.assistant("", [ToolCall("c1", "f", {"z": 1, "a": [2]})])
        builder.tool_result("c1", "ok")
        builder.tool(ToolSpec("f", "does f", {"type": "object", "properties": {}}))
        builder.tool(ToolSpec("g", None, {"type": "object"}))

        rendered = builder.build().template_input()

        messages = json.loads(rendered.messages_json)
        assert messages == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": {"z": 1, "a": [2]}},
                    }
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "c1", "name": "f"},
        ]
        assert '"arguments":{"z":1,"a":[2]}' in rendered.messages_json
        assert json.loads(rendered.tools_json) == [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "description": "does f",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {"type": "function", "function": {"name": "g", "parameters": {"type": "object"}}},
        ]

    def test_non_ascii_text_is_kept_as_is(self) -> None:
        builder = ConversationBuilder()
        builder.user("xin chào 🚀")

        assert "xin chào 🚀" in builder.build().template_input().messages_json


class TestArguments:
    def test_an_object_parses_in_order(self) -> None:
        assert list(parse_arguments('{"b": 1, "a": 2}', where="x")) == ["b", "a"]

    @pytest.mark.parametrize("text", ["[1]", "not json", '{"a": 1, "a": 2}'])
    def test_anything_else_is_an_invalid_request_naming_the_field(self, text: str) -> None:
        with pytest.raises(InvalidRequestError, match=r"^messages\[3\]"):
            parse_arguments(text, where="messages[3]")


class TestEvents:
    def test_merging_joins_adjacent_pieces_of_one_kind_and_drops_empty_ones(self) -> None:
        events = [
            TextDelta("a"),
            TextDelta(""),
            TextDelta("b"),
            ReasoningDelta("r"),
            ReasoningDelta("s"),
            ToolCallStart(0, "id", "f"),
            ToolCallDelta(0, "{"),
            ToolCallDelta(0, "}"),
            ToolCallEnd(0, valid=True),
            TextDelta("c"),
        ]

        assert merge_deltas(events) == [
            TextDelta("ab"),
            ReasoningDelta("rs"),
            ToolCallStart(0, "id", "f"),
            ToolCallDelta(0, "{}"),
            ToolCallEnd(0, valid=True),
            TextDelta("c"),
        ]

    def test_collect_gathers_a_whole_answer(self) -> None:
        answer = collect(
            [
                ReasoningDelta("think"),
                TextDelta("hi "),
                TextDelta("there"),
                ToolCallStart(0, "c0", "f"),
                ToolCallDelta(0, '{"a":'),
                ToolCallDelta(0, "1}"),
                ToolCallEnd(0, valid=True),
                UsageEvent(3, 4),
                FinishEvent(FinishReason.STOP, tool_calls=True),
            ]
        )

        assert answer.text == "hi there"
        assert answer.reasoning == "think"
        [call] = answer.tool_calls
        assert (call.id, call.name, call.arguments, call.valid) == ("c0", "f", '{"a":1}', True)
        assert answer.usage == UsageEvent(3, 4)
        assert answer.finish is not None
        assert answer.finish.tool_calls
        assert answer.error is None

    def test_collect_keeps_an_error(self) -> None:
        answer = collect([TextDelta("x"), ErrorEvent("cancelled", "gone")])

        assert answer.error == ErrorEvent("cancelled", "gone")
        assert answer.finish is None
