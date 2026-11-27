"""End-to-end fault injection for tight KV pools, step OOM recovery, and preemption timeouts."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest

from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, ForwardMode, RequestId

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
    return write_tiny_gpt2(tmp_path_factory.mktemp("e2e-memory-faults"))


def _request(name: str, prompt: str, max_tokens: int, **extra: Any) -> GenerateRequest:
    return GenerateRequest(
        request_id=RequestId(name),
        prompt=prompt,
        sampling=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        **extra,
    )


def _expected(checkpoint: TinyCheckpoint, prompt: str, count: int) -> list[int]:
    return greedy_reference(checkpoint.state, checkpoint.config, [BOS, *prompt.encode()], count)


def test_three_long_concurrent_requests_in_small_pool_complete_with_exact_tokens(
    checkpoint: TinyCheckpoint,
) -> None:
    cfg = tiny_engine_config(checkpoint.directory)
    # Tight pool forces pressure admission pauses and recompute preemption across 3 requests.
    cfg = dataclasses.replace(
        cfg,
        cache=dataclasses.replace(cfg.cache, block_size=4, prefix_cache=True),
        scheduler=dataclasses.replace(
            cfg.scheduler,
            max_running=3,
            max_tokens_per_step=16,
            pressure_high=0.70,
            pressure_critical=0.90,
        ),
    )
    engine = build_engine(cfg, tokenizer=tiny_tokenizer())
    llm = AsyncLLM(engine)
    prompts = [
        ("c1", "first concurrent sequence in tight pool"),
        ("c2", "second concurrent sequence in tight pool"),
        ("c3", "third concurrent sequence in tight pool"),
    ]

    async def body() -> dict[str, list[int]]:
        llm.start()
        try:

            async def collect(name: str, text: str) -> list[int]:
                outs = [o async for o in llm.generate(_request(name, text, 8))]
                assert outs[-1].outputs[0].finish_reason is FinishReason.LENGTH
                return [t for o in outs for t in o.outputs[0].token_ids]

            results = await asyncio.gather(*(collect(n, p) for n, p in prompts))
            assert await llm.drain(5.0)
            engine._kv.check_integrity()
            assert_cache_released(engine)
            return {prompts[i][0]: results[i] for i in range(len(prompts))}
        finally:
            await llm.shutdown(0.0)

    outputs = asyncio.run(body())
    for name, prompt in prompts:
        assert outputs[name] == _expected(checkpoint, prompt, 8)


def test_injected_step_oom_keeps_async_llm_alive_and_finishes_unrelated_streams(
    checkpoint: TinyCheckpoint,
) -> None:
    engine = build_engine(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
    runner = engine._runner
    original_forward = runner._forward
    injected = {"count": 0}

    def oom_once(step: Any, inputs: Any) -> Any:
        if (
            injected["count"] == 0
            and step.mode is ForwardMode.DECODE
            and len(step.request_ids) >= 2
        ):
            injected["count"] += 1
            raise torch.OutOfMemoryError("injected transient OOM")
        return original_forward(step, inputs)

    runner._forward = oom_once  # type: ignore[method-assign]
    llm = AsyncLLM(engine)

    async def body() -> tuple[list[int], list[int]]:
        llm.start()
        try:

            async def run_one(name: str, text: str) -> list[int]:
                outs = [o async for o in llm.generate(_request(name, text, 6))]
                assert outs[-1].outputs[0].finish_reason is FinishReason.LENGTH
                return [t for o in outs for t in o.outputs[0].token_ids]

            t1, t2 = await asyncio.gather(
                run_one("stream-a", "stream alpha stays alive"),
                run_one("stream-b", "stream beta stays alive"),
            )
            assert llm.health().state is EngineState.RUNNING
            assert await llm.drain(5.0)
            engine._kv.check_integrity()
            assert_cache_released(engine)
            return t1, t2
        finally:
            await llm.shutdown(0.0)

    out_a, out_b = asyncio.run(body())
    assert injected["count"] == 1
    assert out_a == _expected(checkpoint, "stream alpha stays alive", 6)
    assert out_b == _expected(checkpoint, "stream beta stays alive", 6)


def test_sustained_preemption_timeout_under_synthetic_clock() -> None:
    from ayaka.configs.scheduler import SchedulerConfig
    from ayaka.scheduler.request import RequestStatus
    from ayaka.scheduler.scheduler import Scheduler
    from fake_runner import FakeRunner, StepLoop, make_kv, make_request

    now = [1.0]
    kv = make_kv(num_blocks=5, max_requests=4, max_blocks=8)
    scheduler = Scheduler(
        SchedulerConfig(max_running=2, max_tokens_per_step=128, preemption_timeout=1.0),
        kv,
        clock=lambda: now[0],
    )
    runner = FakeRunner(kv)
    loop = StepLoop(scheduler, runner, kv)

    keeper = make_request("keeper", 8, max_tokens=6, priority=2)
    starved = make_request("starved", 8, max_tokens=6, priority=0)
    scheduler.add_request(keeper)
    scheduler.add_request(starved)

    for _ in range(10):
        step = loop.step()
        assert step is not None
        req = scheduler.get(starved.request_id)
        if req is not None and req.status is RequestStatus.PREEMPTED:
            break
    now[0] += 2.0
    loop.step()
    loop.run()
    assert loop.finish_reason(starved.request_id) is FinishReason.TIMEOUT
    assert loop.finish_reason(keeper.request_id) is FinishReason.LENGTH
    kv.check_integrity()


@pytest.mark.gpu
def test_gpu_step_oom_recovery_smoke(checkpoint: TinyCheckpoint) -> None:
    base = tiny_engine_config(checkpoint.directory)
    cfg = dataclasses.replace(base, device=dataclasses.replace(base.device, device="cuda"))
    engine = build_engine(cfg, tokenizer=tiny_tokenizer())
    runner = engine._runner
    original_forward = runner._forward
    tripped = {"done": False}

    def inject_gpu_oom(step: Any, inputs: Any) -> Any:
        if not tripped["done"] and step.mode is ForwardMode.DECODE:
            tripped["done"] = True
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original_forward(step, inputs)

    runner._forward = inject_gpu_oom  # type: ignore[method-assign]
    try:
        engine.add_request(_request("gpu-1", "hello cuda", 4))
        engine.add_request(_request("gpu-2", "hello recovery", 4))
        while engine.has_unfinished:
            engine.step()
        assert tripped["done"] is True
        engine._kv.check_integrity()
        assert_cache_released(engine)
    finally:
        engine.shutdown()
