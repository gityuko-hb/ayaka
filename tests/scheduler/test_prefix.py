"""The scheduler with a prefix cache: matched admission, registration and the opt-out."""

from __future__ import annotations

import dataclasses

import pytest

from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.types import FinishReason, ForwardMode
from ayaka.scheduler.scheduler import Scheduler
from fake_runner import BLOCK_SIZE, FakeRunner, StepLoop, make_kv, make_request, rid

pytestmark = pytest.mark.unit


def _loop(*, budget: int = 2048, num_blocks: int = 33) -> StepLoop:
    kv = make_kv(num_blocks, prefix=True)
    scheduler = Scheduler(SchedulerConfig(max_tokens_per_step=budget), kv)
    return StepLoop(scheduler, FakeRunner(kv), kv)


def test_a_second_request_with_the_same_prompt_prefills_only_after_the_match() -> None:
    loop = _loop()
    loop.scheduler.add_request(make_request("a", 13, max_tokens=3))
    loop.run()
    loop.scheduler.add_request(make_request("b", 13, max_tokens=3))  # same prompt: same name length

    first = loop.step()

    assert first is not None
    assert first.mode is ForwardMode.PREFILL
    assert first.start_positions == (12,)
    assert first.rows_per_request == (1,)
    loop.run()
    finals = [e for e in loop.events if e.request_id == rid("b") and e.finished]
    assert finals[0].cached_tokens == 12
    assert loop.finish_reason(rid("b")) is FinishReason.LENGTH
    assert loop.kv.stats().hit_tokens == 12


def test_a_prompt_of_whole_blocks_still_prefills_its_last_block() -> None:
    loop = _loop()
    loop.scheduler.add_request(make_request("a", 2 * BLOCK_SIZE, max_tokens=1))
    loop.run()
    loop.scheduler.add_request(make_request("b", 2 * BLOCK_SIZE, max_tokens=1))

    step = loop.step()

    assert step is not None
    assert step.start_positions == (BLOCK_SIZE,)
    assert step.rows_per_request == (BLOCK_SIZE,)


def test_a_request_that_opts_out_neither_matches_nor_registers() -> None:
    loop = _loop()
    loop.scheduler.add_request(make_request("a", 13, max_tokens=3))
    loop.run()
    request = dataclasses.replace(make_request("b", 13, max_tokens=3), prefix_cache=False)
    loop.scheduler.add_request(request)

    step = loop.step()

    assert step is not None
    assert step.start_positions == (0,)
    assert loop.kv.stats().lookups == 1  # only "a"
    loop.run()
    assert all(e.cached_tokens == 0 for e in loop.events if e.request_id == rid("b"))


def test_generated_blocks_are_registered_as_they_fill() -> None:
    loop = _loop()
    loop.scheduler.add_request(make_request("a", 5, max_tokens=9))

    loop.run()

    # 5 prompt + 8 committed generated tokens (the last one is never run): 13 tokens, 3 blocks.
    assert loop.kv.stats().cached_blocks == (3,)


def test_two_running_requests_share_the_prefix_and_both_finish() -> None:
    loop = _loop(budget=8)
    loop.scheduler.add_request(make_request("a", 13, max_tokens=20))
    loop.step()
    loop.step()  # "a" prefilled 8 + 5 tokens in two chunks
    loop.scheduler.add_request(make_request("b", 13, max_tokens=20))

    loop.run()

    assert loop.finish_reason(rid("a")) is FinishReason.LENGTH
    assert loop.finish_reason(rid("b")) is FinishReason.LENGTH
    b_events = [e for e in loop.events if e.request_id == rid("b")]
    assert b_events[-1].cached_tokens == 12
    stats = loop.kv.stats()
    assert stats.free_blocks == stats.usable_blocks


class TestCacheWrites:
    """``cache_written_tokens``: the prompt tokens a request computed and registered."""

    @staticmethod
    def _final(loop: StepLoop, name: str) -> int:
        [final] = [e for e in loop.events if e.request_id == rid(name) and e.finished]
        return final.cache_written_tokens

    def test_the_first_request_writes_its_whole_prompt_blocks_and_a_repeat_writes_none(
        self,
    ) -> None:
        loop = _loop()
        loop.scheduler.add_request(make_request("a", 13, max_tokens=3))
        loop.run()
        loop.scheduler.add_request(make_request("b", 13, max_tokens=3))
        loop.run()

        assert self._final(loop, "a") == 12  # three whole blocks of four; the 13th stays private
        [b] = [e for e in loop.events if e.request_id == rid("b") and e.finished]
        assert (b.cached_tokens, b.cache_written_tokens) == (12, 0)

    def test_a_prompt_prefilled_in_chunks_counts_every_chunk(self) -> None:
        loop = _loop(budget=8)
        loop.scheduler.add_request(make_request("a", 13, max_tokens=2))

        loop.run()

        assert self._final(loop, "a") == 12

    def test_blocks_of_the_output_are_not_prompt_writes(self) -> None:
        loop = _loop()
        loop.scheduler.add_request(make_request("a", 5, max_tokens=9))

        loop.run()

        assert loop.kv.stats().cached_blocks == (3,)  # one block of prompt, two of output
        assert self._final(loop, "a") == BLOCK_SIZE

    def test_a_request_that_opts_out_writes_nothing(self) -> None:
        loop = _loop()
        request = dataclasses.replace(make_request("a", 13, max_tokens=3), prefix_cache=False)
        loop.scheduler.add_request(request)

        loop.run()

        assert self._final(loop, "a") == 0
