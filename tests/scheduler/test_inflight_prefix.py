"""Unit tests for in-flight prefix coalescing and EngineCore.prewarm."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest, TokenizedRequest
from ayaka.core.types import FinishReason, ForwardMode, RequestId
from ayaka.kvcache.inflight import InFlightPrefixTracker
from ayaka.scheduler.scheduler import Scheduler
from fake_runner import FakeRunner, StepLoop, expected_tokens, make_kv

torch: Any = pytest.importorskip("torch")

from ayaka.engine.build import build_engine  # noqa: E402
from tiny_model import (  # noqa: E402
    TinyCheckpoint,
    assert_cache_released,
    prompt_ids,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

pytestmark = pytest.mark.unit


class _Clock:
    def __init__(self, start: float = 10.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("prefill-coalesce"))


def _tokenized(name: str, prompt: tuple[int, ...], max_tokens: int = 4) -> TokenizedRequest:
    return TokenizedRequest(
        request_id=RequestId(name),
        prompt_token_ids=prompt,
        sampling=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        prefix_cache=True,
    )


def test_inflight_prefix_tracker_lifecycle() -> None:
    tracker = InFlightPrefixTracker()
    assert tracker.leader_for(()) is None

    tracker.register(RequestId("r1"), (b"a", b"b"))
    assert tracker.leader_for((b"a", b"c")) == RequestId("r1")
    assert tracker.leader_for((b"z", b"b")) is None
    assert len(tracker) == 1

    tracker.complete(RequestId("r1"))
    assert tracker.leader_for((b"a", b"b")) is None
    assert len(tracker) == 0


def test_cold_requests_with_shared_prefix_coalesce_and_reuse_cache() -> None:
    kv = make_kv(num_blocks=20, max_requests=4, max_blocks=10, prefix=True)
    clock = _Clock()
    scheduler = Scheduler(
        SchedulerConfig(max_tokens_per_step=128, prefix_wait_timeout=5.0),
        kv,
        clock=clock,
    )
    runner = FakeRunner(kv)
    loop = StepLoop(scheduler, runner, kv)

    shared = (10, 11, 12, 13, 14, 15, 16, 17, 18)  # 2 full blocks (4 tok/block) + 1 token
    r1 = _tokenized("r1", shared, max_tokens=3)
    r2 = _tokenized("r2", shared, max_tokens=3)
    scheduler.add_request(r1)
    scheduler.add_request(r2)

    step1 = loop.step()
    assert step1 is not None
    # r2 is held while r1 is prefilling the shared prefix.
    assert step1.request_ids == (r1.request_id,)
    assert scheduler.num_waiting == 1

    # Once r1 finishes prefill, r2 is admitted and reuses the 8 prefix tokens.
    step2 = loop.step()
    assert step2 is not None
    assert r2.request_id in step2.request_ids
    r2_live = scheduler.get(r2.request_id)
    assert r2_live is not None
    assert r2_live.cached_tokens == 8

    loop.run()
    assert loop.tokens(r1.request_id) == expected_tokens(r1)[0]
    assert loop.tokens(r2.request_id) == expected_tokens(r2)[0]
    kv.check_integrity()


def test_leader_abort_and_wait_timeout_unblock_waiting_request() -> None:
    kv = make_kv(num_blocks=20, max_requests=4, max_blocks=10, prefix=True)
    clock = _Clock()
    scheduler = Scheduler(
        SchedulerConfig(max_tokens_per_step=4, prefix_wait_timeout=2.0),  # 2 steps to prefill r1
        kv,
        clock=clock,
    )
    shared = (20, 21, 22, 23, 24, 25, 26, 27, 28)
    r1 = _tokenized("r1", shared, max_tokens=2)
    r2 = _tokenized("r2", shared, max_tokens=2)
    scheduler.add_request(r1)
    scheduler.add_request(r2)

    step1 = scheduler.schedule()
    assert step1 is not None
    assert step1.mode is ForwardMode.PREFILL
    assert step1.request_ids == (r1.request_id,)
    scheduler.update(FakeRunner(kv).execute(step1))

    # Abort leader mid-prefill -> r2 must be unblocked immediately on next schedule().
    assert scheduler.abort(r1.request_id)
    step2 = scheduler.schedule()
    assert step2 is not None
    assert step2.request_ids == (r2.request_id,)
    scheduler.update(FakeRunner(kv).execute(step2))
    scheduler.abort(r2.request_id)

    # Also test timeout release when leader takes longer than prefix_wait_timeout.
    cold_shared = (30, 31, 32, 33, 34, 35, 36, 37, 38)
    r3 = _tokenized("r3", cold_shared, max_tokens=2)
    r4 = _tokenized("r4", cold_shared, max_tokens=2)
    scheduler.add_request(r3)
    scheduler.add_request(r4)
    step3 = scheduler.schedule()
    assert step3 is not None
    assert step3.request_ids == (r3.request_id,)
    scheduler.update(FakeRunner(kv).execute(step3))

    clock.now += 3.0  # past prefix_wait_timeout=2.0
    step4 = scheduler.schedule()
    assert step4 is not None
    # Note r3 continues its remaining chunk and r4 is no longer blocked from admission.
    scheduler.abort(r3.request_id)
    scheduler.abort(r4.request_id)
    kv.check_integrity()


def test_engine_prewarm_caches_and_pins_prefix_for_follow_up_requests(
    checkpoint: TinyCheckpoint,
) -> None:
    cfg = tiny_engine_config(checkpoint.directory)
    cfg = dataclasses.replace(cfg, cache=dataclasses.replace(cfg.cache, prefix_cache=True))
    engine = build_engine(cfg, tokenizer=tiny_tokenizer())
    try:
        tokens = tuple(prompt_ids("system instruction header shared across calls"))
        pinned_blocks = engine.prewarm(tokens, pin=True)
        assert pinned_blocks > 0

        engine.add_request(
            GenerateRequest(
                request_id=RequestId("warm-user"),
                prompt=tokens,
                sampling=SamplingParams(temperature=0.0, max_tokens=4),
            )
        )
        outs = []
        while engine.has_unfinished:
            outs.extend(engine.step())
        final = [o for o in outs if o.finished][-1]
        assert final.outputs[0].finish_reason is FinishReason.LENGTH
        assert final.usage is not None
        assert final.usage.cached_tokens > 0

        engine.kv.unpin_prefix(tokens)
        assert_cache_released(engine)
    finally:
        engine.shutdown()
