"""Compiled constraints on a byte vocabulary: states, masks, and outputs that always parse.

A random walk that only ever picks a token its mask allows must end in text the grammar accepts,
whatever the walk chooses; the tests run many such walks.
"""

from __future__ import annotations

import json
import random
import time
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.constraints import ConstraintKind, StructuredParams, ToolCallSyntax, ToolSchema
from ayaka.core.errors import CapabilityError, InvalidRequestError, UnsupportedFeatureError
from ayaka.sampling.structured import llguidance_backend as llg_backend_module
from ayaka.sampling.structured.compiler import GrammarCompiler
from ayaka.sampling.structured.llguidance_backend import LLGuidanceBackend
from tokenizer_fakes import (
    CLOSE_CALL,
    EOS,
    IM_END,
    IM_START,
    MODEL_VOCAB,
    OPEN_CALL,
    ToolTokenizer,
)

if TYPE_CHECKING:
    from ayaka.sampling.structured.matcher import ConstraintMatcher

pytest.importorskip("llguidance")

pytestmark = pytest.mark.unit

_EOS = {EOS, IM_END}
_SYNTAX = ToolCallSyntax("<tool_call>", "</tool_call>", OPEN_CALL, CLOSE_CALL)
_WEATHER = ToolSchema(
    "get_weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string", "maxLength": 8}},
        "required": ["city"],
        "additionalProperties": False,
    },
)


def _compiler(**options: Any) -> GrammarCompiler:
    return GrammarCompiler(ToolTokenizer(), vocab_size=MODEL_VOCAB, eos_token_ids=_EOS, **options)


def _start(compiler: GrammarCompiler, params: StructuredParams) -> ConstraintMatcher:
    return compiler.matcher(compiler.compile(params), params)


def _allowed(mask: Any) -> list[int]:
    return [t for t in range(MODEL_VOCAB) if int(mask[t // 32]) >> (t % 32) & 1]


def _walk(matcher: Any, rng: random.Random, limit: int = 120) -> list[int]:
    """Pick allowed tokens at random until an end token, preferring to close the JSON."""
    out: list[int] = []
    for _ in range(limit):
        mask = matcher.mask()
        if mask is None:
            return out
        allowed = _allowed(mask)
        assert allowed, f"empty mask in state {matcher.state}"
        ends = [t for t in allowed if t in _EOS]
        closers = [t for t in allowed if t in (ord("}"), ord('"'), CLOSE_CALL)]
        if ends:
            out.append(ends[0])
            return out
        pool = closers if len(out) > 40 and closers else allowed
        token = rng.choice(pool)
        out.append(token)
        matcher.accept([token])
    return out


def _text(tokens: list[int]) -> str:
    return ToolTokenizer().decode([t for t in tokens if t not in _EOS])


class TestJson:
    @pytest.mark.parametrize("seed", range(8))
    def test_a_schema_constrained_walk_always_validates(self, seed: int) -> None:
        compiler = _compiler()
        schema = {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "minimum": 0, "maximum": 99},
                "ok": {"type": "boolean"},
            },
            "required": ["n", "ok"],
            "additionalProperties": False,
        }
        matcher = _start(compiler, StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=schema))

        tokens = _walk(matcher, random.Random(seed))  # noqa: S311  # a seeded walk

        value = json.loads(_text(tokens))
        assert set(value) == {"n", "ok"}
        assert 0 <= value["n"] <= 99
        assert isinstance(value["ok"], bool)
        assert tokens[-1] in _EOS
        assert matcher.state == "END"

    def test_start_after_leaves_the_output_free_until_the_token(self) -> None:
        compiler = _compiler()
        matcher = _start(compiler, StructuredParams(kind=ConstraintKind.JSON_OBJECT, start_after=7))

        assert matcher.mask() is None
        assert not matcher.active
        matcher.accept([ord("x"), ord("y")])
        assert matcher.mask() is None
        matcher.accept([7])
        assert matcher.active
        assert _allowed(matcher.mask()) == [ord("{")]

    def test_a_block_the_model_may_open_does_not_delay_the_constraint(self) -> None:
        compiler = _compiler()
        params = StructuredParams(kind=ConstraintKind.JSON_OBJECT, start_after=7, reasoning_open=6)

        matcher = _start(compiler, params)

        assert matcher.state == "MAYBE"
        assert _allowed(matcher.mask()) == [6, ord("{")]  # the block, or the answer itself
        matcher.accept([ord("{")])
        assert matcher.state == "ON"
        assert 6 not in _allowed(matcher.mask())  # a block cannot open after the answer began

    @pytest.mark.parametrize("first", ["thinking off", "thinking on"])
    def test_requests_sharing_a_grammar_each_start_from_their_own_thinking(
        self, first: str
    ) -> None:
        # Request-scoped thinking: same schema, so one compiled template, but one request starts
        # the grammar at once and the other only after the reasoning closes (token 7).
        compiler = _compiler()
        off = StructuredParams(kind=ConstraintKind.JSON_OBJECT)
        on = StructuredParams(kind=ConstraintKind.JSON_OBJECT, start_after=7, reasoning_open=6)
        order = (off, on) if first == "thinking off" else (on, off)
        ids = {compiler.compile(params) for params in order}

        assert len(ids) == 1
        (constraint_id,) = ids
        assert compiler.matcher(constraint_id, off).state == "ON"
        assert compiler.matcher(constraint_id, on).state == "MAYBE"

    def test_parameters_of_another_grammar_are_refused(self) -> None:
        compiler = _compiler()
        constraint_id = compiler.compile(StructuredParams(kind=ConstraintKind.JSON_OBJECT))

        with pytest.raises(ValueError, match="do not match"):
            compiler.matcher(
                constraint_id,
                StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema={"type": "integer"}),
            )

    def test_the_block_it_opens_leaves_the_output_free_until_it_closes(self) -> None:
        compiler = _compiler()
        params = StructuredParams(kind=ConstraintKind.JSON_OBJECT, start_after=7, reasoning_open=6)
        matcher = _start(compiler, params)

        matcher.accept([6, ord("x")])

        assert matcher.state == "WAIT_START"
        assert matcher.mask() is None
        matcher.accept([7])
        assert _allowed(matcher.mask()) == [ord("{")]


class TestToolCalls:
    def test_a_lazy_call_is_free_until_the_opening_marker(self) -> None:
        compiler = _compiler()
        matcher = _start(
            compiler,
            StructuredParams(kind=ConstraintKind.TOOL_CALL, tools=(_WEATHER,), syntax=_SYNTAX),
        )

        assert matcher.state == "WAIT_OPEN"
        matcher.accept(list(b"I will check."))
        assert matcher.mask() is None
        matcher.accept([OPEN_CALL])
        assert matcher.state == "ON"

    @pytest.mark.parametrize("seed", range(6))
    def test_a_forced_strict_call_always_parses_and_validates(self, seed: int) -> None:
        compiler = _compiler()
        matcher = _start(
            compiler,
            StructuredParams(
                kind=ConstraintKind.TOOL_CALL,
                tools=(_WEATHER, ToolSchema("get_time", {})),
                syntax=_SYNTAX,
                tool_choice="get_weather",
                strict=True,
                parallel=False,
            ),
        )

        assert _allowed(matcher.mask()) == [OPEN_CALL]
        tokens = _walk(matcher, random.Random(seed))  # noqa: S311  # a seeded walk

        assert tokens[0] == OPEN_CALL
        assert CLOSE_CALL in tokens
        text = _text(tokens)
        body = text.removeprefix("<tool_call>").split("</tool_call>")[0]
        call = json.loads(body)
        assert call["name"] == "get_weather"
        assert isinstance(call["arguments"]["city"], str)
        assert set(call["arguments"]) == {"city"}
        assert tokens[-1] in _EOS  # parallel=False: only an end may follow the call

    def test_with_parallel_calls_the_text_after_a_call_is_free(self) -> None:
        compiler = _compiler()
        matcher = _start(
            compiler,
            StructuredParams(
                kind=ConstraintKind.TOOL_CALL,
                tools=(_WEATHER,),
                syntax=_SYNTAX,
                tool_choice="required",
            ),
        )
        body = '{"name": "get_weather", "arguments": {"city": "Hue"}}'

        matcher.accept([OPEN_CALL, *body.encode(), CLOSE_CALL])

        assert matcher.state == "WAIT_OPEN"
        assert matcher.mask() is None

    def test_a_marker_token_never_appears_inside_the_body(self) -> None:
        compiler = _compiler()
        matcher = _start(
            compiler,
            StructuredParams(
                kind=ConstraintKind.TOOL_CALL,
                tools=(_WEATHER,),
                syntax=_SYNTAX,
                tool_choice="required",
                strict=True,
            ),
        )
        matcher.accept([OPEN_CALL, *b'{"name": "get_weather", "arguments": {"city": "'])

        inside = _allowed(matcher.mask())
        matcher.accept(list(b'Hue"}}'))
        after = _allowed(matcher.mask())

        # The parser ends a call at the marker token, so the token inside a string would cut it.
        assert OPEN_CALL not in inside
        assert CLOSE_CALL not in inside
        assert set(after) == {10, CLOSE_CALL}  # a newline, or the closing marker

    def test_an_opening_marker_without_its_tokens_cannot_be_constrained(self) -> None:
        syntax = ToolCallSyntax("<call>", "</call>", None)

        with pytest.raises(UnsupportedFeatureError, match="tokens of the opening marker"):
            _compiler().compile(
                StructuredParams(kind=ConstraintKind.TOOL_CALL, tools=(_WEATHER,), syntax=syntax)
            )

    @pytest.mark.parametrize("seed", range(3))
    def test_a_spelled_opening_is_forced_token_by_token(self, seed: int) -> None:
        syntax = ToolCallSyntax("<call>", "</call>", open_ids=tuple(b"<call>"))
        compiler = _compiler()
        matcher = _start(
            compiler,
            StructuredParams(
                kind=ConstraintKind.TOOL_CALL,
                tools=(_WEATHER,),
                syntax=syntax,
                tool_choice="required",
                strict=True,
                parallel=False,
            ),
        )

        tokens = _walk(matcher, random.Random(seed))  # noqa: S311  # a seeded walk

        text = _text(tokens)
        assert text.startswith("<call>")
        call = json.loads(text.removeprefix("<call>").split("</call>")[0])
        assert call["name"] == "get_weather"

    def test_a_lazy_spelled_opening_starts_the_grammar_when_written(self) -> None:
        syntax = ToolCallSyntax("<call>", "</call>", open_ids=tuple(b"<call>"))
        compiler = _compiler()
        matcher = _start(
            compiler,
            StructuredParams(kind=ConstraintKind.TOOL_CALL, tools=(_WEATHER,), syntax=syntax),
        )

        matcher.accept(list(b"free <ca"))
        assert matcher.state == "WAIT_OPEN"
        matcher.accept(list(b"ll>"))
        assert matcher.state == "ON"


class TestCompiler:
    def test_the_same_constraint_compiles_once(self) -> None:
        compiler = _compiler()
        params = StructuredParams(kind=ConstraintKind.JSON_OBJECT)

        first = compiler.compile(params)
        second = compiler.compile(StructuredParams(kind=ConstraintKind.JSON_OBJECT))

        assert first == second
        assert compiler.compiles == 1

    def test_the_cache_is_bounded(self) -> None:
        compiler = _compiler(max_entries=2)
        ids = [
            compiler.compile(
                StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema={"const": value})
            )
            for value in range(3)
        ]

        assert len(compiler) == 2
        with pytest.raises(KeyError):
            compiler.matcher(ids[0], StructuredParams(kind=ConstraintKind.JSON_OBJECT))

    def test_an_invalid_schema_is_an_invalid_request(self) -> None:
        params = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema={"type": "nonsense"})

        with pytest.raises(InvalidRequestError, match="not valid"):
            _compiler().compile(params)

    def test_a_slow_compile_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def slow(*args: object) -> object:
            time.sleep(0.5)
            raise AssertionError("never reached in time")

        monkeypatch.setattr(LLGuidanceBackend, "_build", staticmethod(slow))
        compiler = _compiler(timeout=0.05)

        with pytest.raises(InvalidRequestError, match="longer than"):
            compiler.compile(StructuredParams(kind=ConstraintKind.JSON_OBJECT))
        compiler.close()

    def test_without_the_extra_the_request_names_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(name: str, **_: object) -> object:
            raise CapabilityError(name, detail="not installed", remedy="install it")

        monkeypatch.setattr(llg_backend_module, "require_module", missing)

        with pytest.raises(UnsupportedFeatureError, match="structured extra"):
            _compiler().compile(StructuredParams(kind=ConstraintKind.JSON_OBJECT))


class TestOptionalCallWithoutMarkers:
    """``llama3_json`` with ``tool_choice: "auto"``: free until the answer shows what it is."""

    _BARE = ToolCallSyntax("", "", arguments_key="parameters", separator="; ")
    _COUNT = ToolSchema(
        "count",
        {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 0, "maximum": 9}},
            "required": ["n"],
            "additionalProperties": False,
        },
    )

    def _matcher(self, compiler: GrammarCompiler) -> Any:
        params = StructuredParams(
            kind=ConstraintKind.TOOL_CALL, tools=(self._COUNT,), syntax=self._BARE, parallel=False
        )
        return _start(compiler, params)

    def test_text_is_never_constrained(self) -> None:
        matcher = self._matcher(_compiler())

        assert matcher.mask() is None
        matcher.accept([ord(" "), ord("H")])  # a blank waits, a letter decides

        assert matcher.mask() is None
        matcher.accept([ord("{")])  # a brace later in the text changes nothing
        assert matcher.mask() is None
        assert not matcher.active

    @pytest.mark.parametrize("seed", range(6))
    def test_an_answer_that_opens_with_a_brace_is_a_call_with_typed_arguments(
        self, seed: int
    ) -> None:
        matcher = self._matcher(_compiler())
        matcher.accept([ord("\n"), ord("{")])  # the blank is the model's; the brace starts it

        tokens = [ord("{"), *_walk(matcher, random.Random(seed))]  # noqa: S311  # a seeded walk

        assert tokens[-1] in _EOS, _text(tokens)
        call = json.loads(_text(tokens))
        assert call["name"] == "count"
        assert isinstance(call["parameters"]["n"], int)  # never the string "3"

    def test_a_special_token_before_the_brace_keeps_waiting(self) -> None:
        """Llama 3.1 writes ``<|python_tag|>`` before a call about a third of the time."""
        matcher = self._matcher(_compiler())

        matcher.accept([IM_START])  # special, not an end: decodes to no text
        assert matcher.mask() is None
        matcher.accept([ord("{")])

        assert matcher.active

    def test_a_first_token_the_grammar_refuses_leaves_the_answer_free(self) -> None:
        """A token such as ``{%`` opens no JSON object: the text goes on, never cut to an end."""
        from ayaka.sampling.structured.matcher import ConstraintMatcher

        class Refuses:
            def compute_bitmask(self) -> bytes:
                raise AssertionError("no mask while free")

            def consume_token(self, sampled_token: int) -> bool:
                return False

            def is_stopped(self) -> bool:
                return False

            def is_error(self) -> bool:
                return True

            def get_error(self) -> str:
                return "refused"

            def deep_copy(self) -> Refuses:
                return self

        matcher = ConstraintMatcher(
            Refuses(), vocab_size=64, eos_token_ids={0}, brace={7}, skip={1}
        )
        matcher.accept([1, 7])

        assert matcher.state == "FREE"
        assert matcher.mask() is None
