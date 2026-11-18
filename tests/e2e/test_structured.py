"""Structured output end to end on the tiny GPT-2: whatever the model samples, the grammar holds.

The tiny model has random weights, so without a constraint its text is noise; with one, every
sampled answer must parse and validate. That makes it a strict test of the mask path: one token
outside the grammar would break the JSON.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.constraints import ConstraintKind, StructuredParams, ToolCallSyntax, ToolSchema
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from tokenizer_fakes import CLOSE_CALL, OPEN_CALL, ToolTokenizer, loaded_for

torch: Any = pytest.importorskip("torch")
pytest.importorskip("llguidance")

from ayaka.engine.llm import LLM  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    TinyCheckpoint,
    assert_cache_released,
    tiny_engine_config,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ayaka.core.outputs import RequestOutput

pytestmark = pytest.mark.unit

_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "maxLength": 6},
        "score": {"type": "integer", "minimum": 0, "maximum": 9},
    },
    "required": ["answer", "score"],
    "additionalProperties": False,
}
_WEATHER = ToolSchema(
    "get_weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string", "maxLength": 5}},
        "required": ["city"],
        "additionalProperties": False,
    },
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("e2e-structured"))


@pytest.fixture
def llm(checkpoint: TinyCheckpoint) -> Iterator[LLM]:
    tokenizer = loaded_for(ToolTokenizer(), stream=False)
    with LLM.from_config(
        tiny_engine_config(checkpoint.directory, max_tokens_per_step=64), tokenizer=tokenizer
    ) as engine:
        yield engine


def _one(result: RequestOutput | list[RequestOutput]) -> RequestOutput:
    assert not isinstance(result, list)
    return result


def _ask(
    llm: LLM, name: str, structured: StructuredParams, sampling: SamplingParams
) -> RequestOutput:
    return _one(
        llm.generate(
            GenerateRequest(
                request_id=RequestId(name),
                prompt="answer in json:",
                sampling=sampling,
                structured=structured,
            )
        )
    )


@pytest.mark.parametrize("seed", range(5))
def test_a_schema_answer_always_parses_and_validates(llm: LLM, seed: int) -> None:
    params = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=_SCHEMA)

    out = _ask(llm, f"s{seed}", params, SamplingParams(temperature=1.0, seed=seed, max_tokens=60))

    value = json.loads(out.outputs[0].text)
    assert set(value) == {"answer", "score"}
    assert len(value["answer"]) <= 6
    assert 0 <= value["score"] <= 9
    assert out.outputs[0].finish_reason is FinishReason.STOP
    assert len(llm.engine.runner.constraints) == 0
    assert_cache_released(llm.engine)


def test_a_json_object_answer_is_an_object(llm: LLM) -> None:
    out = _ask(
        llm,
        "obj",
        StructuredParams(kind=ConstraintKind.JSON_OBJECT),
        SamplingParams(temperature=0.0, max_tokens=60),
    )

    text = out.outputs[0].text
    assert text.startswith("{")
    if out.outputs[0].finish_reason is FinishReason.STOP:
        assert isinstance(json.loads(text), dict)


def test_max_tokens_cuts_the_json_and_invents_no_bracket(llm: LLM) -> None:
    params = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=_SCHEMA)

    out = _ask(llm, "cut", params, SamplingParams(temperature=1.0, seed=1, max_tokens=4))

    completion = out.outputs[0]
    assert completion.finish_reason is FinishReason.LENGTH
    assert len(completion.token_ids) == 4
    assert completion.text.startswith("{")
    with pytest.raises(json.JSONDecodeError):
        json.loads(completion.text)


def test_a_forced_strict_tool_call_is_a_valid_call(llm: LLM) -> None:
    params = StructuredParams(
        kind=ConstraintKind.TOOL_CALL,
        tools=(_WEATHER,),
        syntax=ToolCallSyntax("<tool_call>", "</tool_call>", OPEN_CALL, CLOSE_CALL),
        tool_choice="required",
        strict=True,
        parallel=False,
    )

    out = _ask(llm, "call", params, SamplingParams(temperature=1.0, seed=3, max_tokens=60))

    text = out.outputs[0].text
    assert text.startswith("<tool_call>")
    body = text.removeprefix("<tool_call>").split("</tool_call>")[0]
    call = json.loads(body)
    assert call["name"] == "get_weather"
    assert set(call["arguments"]) == {"city"}
    assert out.outputs[0].token_ids[0] == OPEN_CALL


def test_two_sequences_both_follow_the_schema(llm: LLM) -> None:
    params = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=_SCHEMA)

    out = _ask(llm, "pair", params, SamplingParams(temperature=1.0, seed=2, n=2, max_tokens=60))

    assert len(out.outputs) == 2
    for completion in out.outputs:
        assert set(json.loads(completion.text)) == {"answer", "score"}


def test_the_same_constraint_compiles_once_and_free_requests_pay_nothing(llm: LLM) -> None:
    params = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=_SCHEMA)
    sampling = SamplingParams(temperature=1.0, seed=4, max_tokens=60)
    compiler = llm.engine._input.compiler  # pyright: ignore[reportPrivateUsage]

    _ask(llm, "a", params, sampling)
    _ask(llm, "b", params, sampling)
    llm.generate("plain text", SamplingParams(temperature=0.0, max_tokens=4))

    assert compiler.compiles == 1
    assert len(llm.engine.runner.constraints) == 0
