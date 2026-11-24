"""Predicted outputs (OpenAI ``prediction``), CLI speculative flags, and HTTP usage details."""

from __future__ import annotations

import asyncio

import pytest

from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.speculative import SpeculativeConfig
from ayaka.core.errors import InvalidRequestError
from ayaka.core.outputs import CompletionOutput, RequestOutput, Usage
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.entrypoints.cli import _parser, main  # pyright: ignore[reportPrivateUsage]
from ayaka.entrypoints.conversation import UsageEvent
from ayaka.entrypoints.openai.protocol import openai_usage, parse_chat_request
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.wire.reader import JsonObject

pytest.importorskip("torch")
pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestClient, TestServer

from ayaka.configs.server import ServerConfig
from ayaka.engine.async_llm import AsyncLLM
from ayaka.engine.build import build_engine
from ayaka.engine.llm import LLM
from ayaka.entrypoints.http import create_app
from ayaka.entrypoints.serving import Serving
from tiny_model import (
    TinyCheckpoint,
    patch_tokenizer_loading,
    tiny_tokenizer,
    write_tiny_gpt2,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("pred-gpt2"))


@pytest.fixture(autouse=True)
def tiny_tokenizer_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_tokenizer_loading(monkeypatch)


def test_parse_chat_prediction_variants_and_errors() -> None:
    chat_str = parse_chat_request(
        JsonObject(
            {
                "messages": [{"role": "user", "content": "edit"}],
                "prediction": {"type": "content", "content": "hello"},
            }
        )
    )
    assert chat_str.prediction == "hello"

    chat_parts = parse_chat_request(
        JsonObject(
            {
                "messages": [{"role": "user", "content": "edit"}],
                "prediction": {
                    "type": "content",
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ],
                },
            }
        )
    )
    assert chat_parts.prediction == "hello world"

    for bad_pred in (
        {"type": "other", "content": "x"},
        {"type": "content", "content": 123},
        {"type": "content", "content": [{"type": "image", "text": "x"}]},
    ):
        with pytest.raises(InvalidRequestError):
            parse_chat_request(
                JsonObject(
                    {"messages": [{"role": "user", "content": "edit"}], "prediction": bad_pred}
                )
            )


def test_http_chat_prediction_reports_usage_details(checkpoint: TinyCheckpoint) -> None:
    parser = OutputFormat.resolve().parser(call_id_prefix="c", tools=False)
    events = parser.feed(
        RequestOutput(
            request_id=RequestId("u"),
            outputs=(
                CompletionOutput(
                    index=0, text="hi", token_ids=(1, 2), finish_reason=FinishReason.STOP
                ),
            ),
            finished=True,
            usage=Usage(
                prompt_tokens=3,
                completion_tokens=2,
                accepted_prediction_tokens=4,
                rejected_prediction_tokens=1,
            ),
        )
    )
    usage_ev = [ev for ev in events if isinstance(ev, UsageEvent)][-1]
    serialized = openai_usage(usage_ev)
    assert serialized is not None
    assert serialized["completion_tokens_details"] == {
        "reasoning_tokens": 0,
        "accepted_prediction_tokens": 4,
        "rejected_prediction_tokens": 1,
    }
    plain = openai_usage(UsageEvent(prompt_tokens=10, completion_tokens=5))
    assert plain is not None
    assert plain["completion_tokens_details"] == {"reasoning_tokens": 0}

    async def run() -> None:
        cfg = EngineConfig(
            source=ModelSourceConfig(str(checkpoint.directory), offline=True),
            speculative=SpeculativeConfig(
                enabled=True, method="lookup", controller="fixed", fixed_k=3, max_draft_tokens=4
            ),
        )
        core = build_engine(cfg, tokenizer=tiny_tokenizer())
        llm = AsyncLLM(core)
        llm.start()
        serving = Serving(ServerConfig(), model_name="tiny")
        serving.attach(llm, core.tokenizer, output_format=None)
        server = TestServer(create_app(serving), handler_cancellation=True)
        try:
            async with TestClient(server) as client:
                base_req = {
                    "messages": [{"role": "user", "content": "rewrite"}],
                    "max_tokens": 8,
                    "temperature": 0.0,
                }
                base_resp = await (await client.post("/v1/chat/completions", json=base_req)).json()
                predicted = base_resp["choices"][0]["message"]["content"]
                assert predicted

                pred_req = {
                    **base_req,
                    "prediction": {"type": "content", "content": predicted},
                }
                resp = await (await client.post("/v1/chat/completions", json=pred_req)).json()
                assert resp["choices"][0]["message"]["content"] == predicted
                details = resp["usage"]["completion_tokens_details"]
                assert details["accepted_prediction_tokens"] > 0
        finally:
            await llm.shutdown()
            core.shutdown()

    asyncio.run(run())


def test_engine_prediction_hint_matches_plain_generation(checkpoint: TinyCheckpoint) -> None:
    prompt_text = "alpha beta gamma"
    base_cfg = EngineConfig(source=ModelSourceConfig(str(checkpoint.directory), offline=True))
    with LLM.from_config(base_cfg) as plain_llm:
        expected_out = plain_llm.generate(
            prompt_text,
            SamplingParams(temperature=0.0, max_tokens=12),
        )
        assert not isinstance(expected_out, list)
        expected = expected_out.outputs[0]

    spec_cfg = EngineConfig(
        source=ModelSourceConfig(str(checkpoint.directory), offline=True),
        speculative=SpeculativeConfig(
            enabled=True, method="lookup", controller="fixed", fixed_k=3, max_draft_tokens=4
        ),
    )
    with LLM.from_config(spec_cfg) as spec_llm:
        last_with_hint = spec_llm.generate(
            GenerateRequest(
                request_id=RequestId("pred-1"),
                prompt=prompt_text,
                sampling=SamplingParams(temperature=0.0, max_tokens=12),
                prediction=expected.text,
            )
        )
        assert not isinstance(last_with_hint, list)
        assert last_with_hint.outputs[0].token_ids == expected.token_ids
        assert (
            last_with_hint.usage.accepted_prediction_tokens
            + last_with_hint.usage.rejected_prediction_tokens
        ) > 0


def test_cli_speculative_flags_and_lookup_run(
    checkpoint: TinyCheckpoint, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "run",
                str(checkpoint.directory),
                "--offline",
                "-p",
                "hello",
                "--max-tokens",
                "8",
                "-q",
            ]
        )
        == 0
    )
    plain_out = capsys.readouterr().out

    assert (
        main(
            [
                "run",
                str(checkpoint.directory),
                "--offline",
                "-p",
                "hello",
                "--max-tokens",
                "8",
                "--speculative",
                "lookup",
                "--spec-max-draft-tokens",
                "4",
                "--spec-controller",
                "fixed",
                "--spec-fixed-k",
                "2",
            ]
        )
        == 0
    )
    spec_cap = capsys.readouterr()
    assert spec_cap.out == plain_out
    assert "speculative=lookup" in spec_cap.err

    parsed = _parser().parse_args(
        [
            "serve",
            str(checkpoint.directory),
            "--speculative",
            "dflash",
            "--spec-draft-model",
            "z-lab/Qwen3-8B-DFlash-b16",
            "--spec-confidence-threshold",
            "0.7",
            "--spec-context-cap",
            "256",
            "--spec-mtp-depth",
            "2",
            "--spec-mtp-vocab-limit",
            "8192",
            "--spec-lookup-override",
            "6",
        ]
    )
    assert parsed.speculative == "dflash"
    assert parsed.spec_draft_model == "z-lab/Qwen3-8B-DFlash-b16"
    assert parsed.spec_confidence_threshold == 0.7
    assert parsed.spec_context_cap == 256
    assert parsed.spec_mtp_depth == 2
    assert parsed.spec_mtp_vocab_limit == 8192
    assert parsed.spec_lookup_override == 6
