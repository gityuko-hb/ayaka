"""Unit tests for step-level out-of-memory recovery inside ModelRunner, EngineCore, and AsyncLLM."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest
from prometheus_client import generate_latest

from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, ForwardMode, RequestId
from ayaka.metrics.loggers import PrometheusStatLogger

torch: Any = pytest.importorskip("torch")

from ayaka.engine.async_llm import AsyncLLM  # noqa: E402
from ayaka.engine.build import build_engine  # noqa: E402
from ayaka.engine.seam import EngineState  # noqa: E402
from tiny_model import (  # noqa: E402
    TinyCheckpoint,
    assert_cache_released,
    greedy_reference,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)
from tokenizer_fakes import BOS  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("oom-recovery"))


def _request(name: str, prompt: str, max_tokens: int) -> GenerateRequest:
    return GenerateRequest(
        request_id=RequestId(name),
        prompt=prompt,
        sampling=SamplingParams(temperature=0.0, max_tokens=max_tokens),
    )


def _expected(checkpoint: TinyCheckpoint, prompt: str, count: int) -> list[int]:
    return greedy_reference(checkpoint.state, checkpoint.config, [BOS, *prompt.encode()], count)


def test_decode_oom_preempts_victim_retries_and_preserves_tokens(
    checkpoint: TinyCheckpoint,
) -> None:
    logger = PrometheusStatLogger()
    engine = build_engine(
        tiny_engine_config(checkpoint.directory),
        tokenizer=tiny_tokenizer(),
        stat_loggers=(logger,),
    )
    runner = engine._runner
    original_forward = runner._forward
    injected = {"done": False}

    def forward_with_oom(step: Any, inputs: Any) -> Any:
        if not injected["done"] and step.mode is ForwardMode.DECODE and len(step.request_ids) == 2:
            injected["done"] = True
            raise torch.OutOfMemoryError("synthetic decode OOM")
        return original_forward(step, inputs)

    runner._forward = forward_with_oom  # type: ignore[method-assign]

    try:
        engine.add_request(_request("r1", "alpha beta", 5))
        engine.add_request(_request("r2", "gamma delta", 5))
        tokens: dict[RequestId, list[int]] = {RequestId("r1"): [], RequestId("r2"): []}
        reasons: dict[RequestId, FinishReason | None] = {}
        while engine.has_unfinished:
            for out in engine.step():
                if out.outputs and out.outputs[0].token_ids:
                    tokens[out.request_id].extend(out.outputs[0].token_ids)
                if out.finished:
                    reasons[out.request_id] = out.outputs[0].finish_reason
        assert injected["done"] is True
        assert reasons[RequestId("r1")] is FinishReason.LENGTH
        assert reasons[RequestId("r2")] is FinishReason.LENGTH
        assert tokens[RequestId("r1")] == _expected(checkpoint, "alpha beta", 5)
        assert tokens[RequestId("r2")] == _expected(checkpoint, "gamma delta", 5)
        assert_cache_released(engine)
        prom_text = generate_latest(logger.registry).decode("utf-8")
        assert "ayaka_engine_oom_recoveries_total 1.0" in prom_text
    finally:
        engine.shutdown()


def test_single_request_large_prefill_oom_halves_chunk_budget_and_recovers(
    checkpoint: TinyCheckpoint,
) -> None:
    cfg = tiny_engine_config(checkpoint.directory)
    cfg = dataclasses.replace(
        cfg,
        scheduler=dataclasses.replace(cfg.scheduler, max_tokens_per_step=64),
    )
    engine = build_engine(cfg, tokenizer=tiny_tokenizer())
    runner = engine._runner
    original_forward = runner._forward
    seen_chunks: list[int] = []
    injected = {"done": False}

    def forward_with_prefill_oom(step: Any, inputs: Any) -> Any:
        if step.mode is ForwardMode.PREFILL:
            seen_chunks.append(step.rows_per_request[0])
            if not injected["done"] and step.rows_per_request[0] > 8:
                injected["done"] = True
                raise RuntimeError("CUDA out of memory during prefill")
        return original_forward(step, inputs)

    runner._forward = forward_with_prefill_oom  # type: ignore[method-assign]
    prompt = "long prompt enough for chunk halving across blocks"
    try:
        engine.add_request(_request("single", prompt, 4))
        tokens: list[int] = []
        while engine.has_unfinished:
            for out in engine.step():
                if out.outputs and out.outputs[0].token_ids:
                    tokens.extend(out.outputs[0].token_ids)
        assert injected["done"] is True
        assert len(seen_chunks) >= 2
        assert seen_chunks[1] < seen_chunks[0]
        assert tokens == _expected(checkpoint, prompt, 4)
        # Budget restores gradually back to full cap once clean steps succeed.
        assert engine._scheduler._effective_budget == cfg.scheduler.max_tokens_per_step
        assert_cache_released(engine)
    finally:
        engine.shutdown()


def test_persistent_minimal_oom_fails_only_bad_request_and_keeps_async_llm_alive(
    checkpoint: TinyCheckpoint,
) -> None:
    engine = build_engine(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
    runner = engine._runner
    original_forward = runner._forward

    def always_oom_for_bad(step: Any, inputs: Any) -> Any:
        if RequestId("bad") in step.request_ids:
            raise torch.OutOfMemoryError("persistent OOM for bad request")
        return original_forward(step, inputs)

    runner._forward = always_oom_for_bad  # type: ignore[method-assign]
    llm = AsyncLLM(engine)

    async def body() -> None:
        llm.start()
        try:
            bad_outputs = [o async for o in llm.generate(_request("bad", "hi", 4))]
            assert bad_outputs[-1].finished is True
            assert bad_outputs[-1].outputs[0].finish_reason is FinishReason.ERROR
            assert bad_outputs[-1].error is not None
            assert bad_outputs[-1].error.code == "out_of_memory"
            assert bad_outputs[-1].error.retryable is True
            assert llm.health().state is EngineState.RUNNING

            good_outputs = [o async for o in llm.generate(_request("good", "ok", 4))]
            good_tokens = [t for o in good_outputs for t in o.outputs[0].token_ids]
            assert good_outputs[-1].finished is True
            assert good_outputs[-1].outputs[0].finish_reason is FinishReason.LENGTH
            assert good_tokens == _expected(checkpoint, "ok", 4)
            assert await llm.drain(5.0)
            assert_cache_released(llm.engine)
        finally:
            await llm.shutdown(0.0)

    asyncio.run(body())
