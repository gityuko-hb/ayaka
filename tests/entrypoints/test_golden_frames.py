"""The wire, byte for byte: SSE frames, whole answers and error bodies against golden files.

The other serving tests check that the official SDKs can parse what the server sends. They do not
fail when an encoder changes the bytes in a way a parser tolerates: a key order, an escape, a
separator, a frame split in two. These tests do. Every protocol's stream encoder is fed a fixed
sequence of events, every whole answer is built from the same events, and every error kind is
rendered in each protocol's shape; the bytes must equal the files under
``tests/entrypoints/fixtures/golden/``.

A change to the wire is then a reviewed diff of those files: run the test with
``AYAKA_UPDATE_GOLDEN=1`` to write them again (``tests/_support/golden.py``), and read the diff
before committing it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web

from ayaka.core.errors import (
    CapacityError,
    EngineError,
    InvalidRequestError,
    OverloadedError,
    RequestTimeout,
    UnsupportedFeatureError,
)
from ayaka.core.outputs import TokenLogprob
from ayaka.core.types import FinishReason
from ayaka.entrypoints.anthropic.messages import MessagesStreamEncoder, message_body
from ayaka.entrypoints.anthropic.protocol import anthropic_error
from ayaka.entrypoints.conversation import (
    ChoiceEvent,
    ErrorEvent,
    FinishEvent,
    ReasoningDelta,
    TextDelta,
    TokenLogprobs,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
    collect,
    collect_choices,
)
from ayaka.entrypoints.http import NotReadyError, failure_for, failure_for_event
from ayaka.entrypoints.openai.chat import (
    ChatStreamEncoder,
    CompletionStreamEncoder,
    _Meta,  # pyright: ignore[reportPrivateUsage]
    chat_completion_body,
    completion_body,
)
from ayaka.entrypoints.openai.protocol import openai_error
from ayaka.entrypoints.openai.responses import (
    ResponsesStreamEncoder,
    _Snapshot,  # pyright: ignore[reportPrivateUsage]
    response_body,
)
from golden import GOLDEN, UPDATE_GOLDEN, check_golden

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayaka.entrypoints.conversation import OutputEvent
    from ayaka.entrypoints.stream import StreamEncoder

pytestmark = pytest.mark.unit

_PUBLIC = "0123456789abcdef0123456789abcdef"
_CREATED = 1700000000
_MODEL = "tiny"
_PROMPT_TOKENS = 11


# -- the event sequences ----------------------------------------------------------------------

_USAGE = UsageEvent(prompt_tokens=_PROMPT_TOKENS, completion_tokens=7, cached_tokens=4)
_LOGPROBS = (
    TokenLogprob(72, -0.25, rank=1, decoded="H", top=((72, -0.25),), top_decoded=("H",)),
    TokenLogprob(
        105,
        -1.5,
        rank=2,
        decoded="é",
        top=((9, -0.75), (105, -1.5)),
        top_decoded=("\n", "é"),
    ),
)

#: Text with what JSON must escape or may not: quotes, a backslash, a newline, a control
#: character, U+2028, a non-ASCII letter and an emoji outside the BMP.
_AWKWARD = 'say "hi"\\\n\x01\u2028wörld 😀'

CASES: dict[str, list[OutputEvent]] = {
    "text": [
        TextDelta("Hel"),
        TextDelta(_AWKWARD),
        _USAGE,
        FinishEvent(FinishReason.STOP),
    ],
    "reasoning": [
        ReasoningDelta("let me "),
        ReasoningDelta("think"),
        TextDelta("Answer."),
        _USAGE,
        FinishEvent(FinishReason.STOP),
    ],
    "tool_call": [
        TextDelta("Checking."),
        ToolCallStart(0, "call_0123_0", "get_weather"),
        ToolCallDelta(0, '{"city": '),
        ToolCallDelta(0, '"Hu\\u1ebf"}'),
        ToolCallEnd(0, valid=True),
        ToolCallStart(1, "call_0123_1", "get_time"),
        ToolCallDelta(1, "{}"),
        ToolCallEnd(1, valid=True),
        _USAGE,
        FinishEvent(FinishReason.STOP, tool_calls=True),
    ],
    "broken_tool_call": [
        ToolCallStart(0, "call_0123_0", "get_weather"),
        ToolCallDelta(0, '{"city": "Hu'),
        ToolCallEnd(0, valid=False),
        _USAGE,
        FinishEvent(FinishReason.LENGTH, tool_calls=True),
    ],
    "logprobs": [
        TextDelta("H"),
        TokenLogprobs(_LOGPROBS[:1]),
        TextDelta("é"),
        TokenLogprobs(_LOGPROBS[1:]),
        _USAGE,
        FinishEvent(FinishReason.STOP),
    ],
    "stop_sequence": [
        TextDelta("one two"),
        _USAGE,
        FinishEvent(FinishReason.STOP, stop_sequence="###"),
    ],
    "length": [
        TextDelta("cut"),
        _USAGE,
        FinishEvent(FinishReason.LENGTH),
    ],
    "error": [
        TextDelta("partial"),
        ErrorEvent("engine_dead", "the engine stopped"),
    ],
}

#: ``n = 2``: each choice's events, then the request's usage and end (``ChoicesParser``).
CHOICES: list[OutputEvent] = [
    ChoiceEvent(0, TextDelta("a")),
    ChoiceEvent(1, TextDelta("b")),
    ChoiceEvent(0, TokenLogprobs(_LOGPROBS[:1])),
    ChoiceEvent(0, FinishEvent(FinishReason.STOP)),
    ChoiceEvent(1, TextDelta("c")),
    ChoiceEvent(1, FinishEvent(FinishReason.LENGTH)),
    _USAGE,
    FinishEvent(FinishReason.STOP),
]


def _stream(encoder: StreamEncoder, events: Sequence[OutputEvent]) -> bytes:
    """What ``pump`` writes for ``events``, with a keepalive after the first frame."""
    frames = [encoder.start(), encoder.keepalive()]
    frames += [encoder.encode(event) for event in events]
    if isinstance(events[-1], (FinishEvent, ErrorEvent)):
        frames.append(encoder.end())
    return b"".join(frames)


def _meta(prefix: str) -> _Meta:
    return _Meta(f"{prefix}-{_PUBLIC}", _CREATED, _MODEL)


def _snapshot() -> _Snapshot:
    echo: dict[str, object] = {
        "instructions": None,
        "max_output_tokens": None,
        "parallel_tool_calls": True,
        "reasoning": {"effort": None, "summary": None},
        "temperature": None,
        "top_p": None,
        "tool_choice": "auto",
        "tools": [],
        "metadata": {},
    }
    return _Snapshot(f"resp_{_PUBLIC}", _CREATED, _MODEL, echo)


ENCODERS: dict[str, Callable[[], StreamEncoder]] = {
    "openai-chat": lambda: ChatStreamEncoder(_meta("chatcmpl"), include_usage=False),
    "openai-chat-usage": lambda: ChatStreamEncoder(_meta("chatcmpl"), include_usage=True),
    "openai-completions": lambda: CompletionStreamEncoder(_meta("cmpl"), include_usage=False),
    "openai-completions-usage": lambda: CompletionStreamEncoder(_meta("cmpl"), include_usage=True),
    "anthropic-messages": lambda: MessagesStreamEncoder(
        f"msg_{_PUBLIC}", _MODEL, prompt_tokens=_PROMPT_TOKENS
    ),
    "openai-responses": lambda: ResponsesStreamEncoder(_snapshot(), _PUBLIC),
}


def _unary(protocol: str, events: Sequence[OutputEvent]) -> dict[str, object] | None:
    """The whole-answer body of ``events``, or ``None`` when the answer is an error."""
    answer = collect(events)
    if answer.error is not None:
        return None
    match protocol:
        case "openai-chat":
            return chat_completion_body(_meta("chatcmpl"), [answer])
        case "openai-completions":
            return completion_body(_meta("cmpl"), [answer])
        case "anthropic-messages":
            return message_body(f"msg_{_PUBLIC}", _MODEL, answer, prompt_tokens=_PROMPT_TOKENS)
        case "openai-responses":
            return response_body(_snapshot(), answer, _PUBLIC)
        case _:
            raise AssertionError(protocol)


def _json_bytes(body: object) -> bytes:
    """The body as ``web.json_response`` writes it (``json.dumps`` with its defaults)."""
    response = web.json_response(body)
    assert response.body is not None
    return bytes(response.body)  # pyright: ignore[reportArgumentType]  # bytes for a json body


# -- streams and whole answers -----------------------------------------------------------------

_USAGE_ENCODERS = ("openai-chat-usage", "openai-completions-usage")
_UNARY_PROTOCOLS = ("anthropic-messages", "openai-chat", "openai-completions", "openai-responses")
_CHOICES_BODIES = ("openai-chat", "openai-completions")

#: ``include_usage`` adds one chunk after the finish and changes nothing else, so those two
#: encoders are locked for a plain answer and for two choices, not again for every case.
_STREAMS = [
    *((p, c) for p in sorted(ENCODERS) if p not in _USAGE_ENCODERS for c in sorted(CASES)),
    *((p, "text") for p in _USAGE_ENCODERS),
]
#: A whole answer that is an error is not a body: ``errors.json`` holds it.
_UNARY = [(p, c) for p in _UNARY_PROTOCOLS for c in sorted(CASES) if c != "error"]


class TestStreams:
    @pytest.mark.parametrize(("protocol", "case"), _STREAMS)
    def test_a_stream_writes_the_golden_frames(self, protocol: str, case: str) -> None:
        check_golden(f"stream/{protocol}/{case}.sse", _stream(ENCODERS[protocol](), CASES[case]))

    @pytest.mark.parametrize("protocol", _USAGE_ENCODERS)
    def test_two_choices_write_the_golden_frames(self, protocol: str) -> None:
        def encoder() -> StreamEncoder:
            if protocol.startswith("openai-chat"):
                return ChatStreamEncoder(_meta("chatcmpl"), include_usage=True, choices=2)
            return CompletionStreamEncoder(_meta("cmpl"), include_usage=True, choices=2)

        check_golden(f"stream/{protocol}/choices.sse", _stream(encoder(), CHOICES))


class TestWholeAnswers:
    @pytest.mark.parametrize(("protocol", "case"), _UNARY)
    def test_a_whole_answer_is_the_golden_body(self, protocol: str, case: str) -> None:
        body = _unary(protocol, CASES[case])
        assert body is not None
        check_golden(f"unary/{protocol}/{case}.json", _json_bytes(body))

    @pytest.mark.parametrize("protocol", _CHOICES_BODIES)
    def test_two_choices_are_the_golden_body(self, protocol: str) -> None:
        answers = collect_choices(CHOICES, 2)
        build = chat_completion_body if protocol == "openai-chat" else completion_body
        prefix = "chatcmpl" if protocol == "openai-chat" else "cmpl"
        check_golden(f"unary/{protocol}/choices.json", _json_bytes(build(_meta(prefix), answers)))

    def test_every_case_has_a_golden_file_and_no_file_is_stale(self) -> None:
        """A golden file no test writes would silently stop guarding anything.

        ``replay/`` belongs to ``test_replay.py``, which checks its own files.
        """
        if UPDATE_GOLDEN:
            pytest.skip("the files are being written")
        expected = {f"stream/{p}/{c}.sse" for p, c in _STREAMS}
        expected |= {f"stream/{p}/choices.sse" for p in _USAGE_ENCODERS}
        expected |= {f"unary/{p}/{c}.json" for p, c in _UNARY}
        expected |= {f"unary/{p}/choices.json" for p in _CHOICES_BODIES}
        expected.add("errors.json")
        found = {
            path.relative_to(GOLDEN).as_posix()
            for path in GOLDEN.rglob("*")
            if path.is_file() and path.relative_to(GOLDEN).parts[0] != "replay"
        }
        assert found == expected


# -- errors -------------------------------------------------------------------------------------

#: Every way a request fails before its answer starts, as the middleware sees it.
_FAILURES: dict[str, Callable[[], BaseException]] = {
    "invalid_request": lambda: InvalidRequestError(
        "temperature: must be a number", context={"param": "temperature"}
    ),
    "unsupported_feature": lambda: UnsupportedFeatureError(
        "audio: not available on this server", context={"features": ("audio",), "param": "audio"}
    ),
    "capacity_exceeded": lambda: CapacityError("the prompt needs 900 blocks; 512 exist"),
    "overloaded": lambda: OverloadedError("the waiting queue is full"),
    "request_timeout": lambda: RequestTimeout("the request ran out of time"),
    "not_ready": lambda: NotReadyError("the server is starting; retry later"),
    "engine_dead": lambda: EngineError("the engine thread died"),
    "unauthorized": lambda: web.HTTPUnauthorized(text="a valid API key is required"),
    "not_found": lambda: web.HTTPNotFound(text="model 'x' is not served here; available: tiny"),
    "too_large": lambda: web.HTTPRequestEntityTooLarge(max_size=10, actual_size=20),
    "internal": lambda: RuntimeError("a bug"),
}

#: Every way an admitted request can end in error, as the parser reports it.
_EVENTS: dict[str, ErrorEvent] = {
    "request_timeout": ErrorEvent("request_timeout", "the request ran out of time", True),
    "cancelled": ErrorEvent("cancelled", "the request was cancelled"),
    "engine_dead": ErrorEvent("engine_dead", "the engine stopped"),
    "not_ready": ErrorEvent("not_ready", "the server is draining", True),
    "backend_error": ErrorEvent("backend_error", "a kernel failed"),
}

_RENDERERS: dict[str, Callable[[Any], dict[str, object]]] = {
    "openai": openai_error,
    "anthropic": anthropic_error,
}


def _envelopes(failure: Any) -> dict[str, object]:
    """A failure's status and headers, and the body each protocol renders for it."""
    return {
        "status": failure.status,
        "headers": failure.headers,
        **{
            protocol: _json_bytes(render(failure)).decode()
            for protocol, render in _RENDERERS.items()
        },
    }


def _error_frames(encoder: StreamEncoder, event: ErrorEvent) -> str:
    """The frames an encoder writes for an error event, then its end."""
    return (encoder.encode(event) + encoder.end()).decode()


def _error_table() -> dict[str, object]:
    table: dict[str, object] = {}
    for name, make in _FAILURES.items():
        table[f"unary/{name}"] = _envelopes(failure_for(make(), _PUBLIC))
    for name, event in _EVENTS.items():
        table[f"answer/{name}"] = _envelopes(failure_for_event(event, _PUBLIC))
        table[f"stream/{name}"] = {
            protocol: _error_frames(encoder(), event)
            for protocol, encoder in ENCODERS.items()
            if not protocol.endswith("-usage")
        }
    return table


class TestErrors:
    def test_every_error_kind_is_the_golden_shape_in_every_protocol(self) -> None:
        data = json.dumps(_error_table(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        check_golden("errors.json", data.encode())
