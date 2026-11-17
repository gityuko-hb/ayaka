"""The engine end to end on the tiny GPT-2: build it, generate, stop, fail, and shut down.

The oracle is ``greedy_reference``, GPT-2's formulas with no cache, and the byte tokenizer's own
``decode``: whatever the engine returns for a greedy request must be those tokens and that text.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.core.errors import (
    CapacityError,
    ConfigError,
    EngineDeadError,
    EngineError,
    ErrorInfo,
    InvalidRequestError,
    ModelLoadError,
    UnsupportedFeatureError,
)
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.device import info as device_info
from ayaka.device.info import DeviceInfo
from ayaka.scheduler.step import StepOutput
from ayaka.tokenizers.contracts import ChatMessage
from ayaka.tokenizers.factory import DefaultTokenizerFactory
from ayaka.utils.dtypes import DType
from tokenizer_fakes import BOS, FakeTokenizer

torch: Any = pytest.importorskip("torch")

from ayaka.engine.llm import LLM  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    MAX_POSITIONS,
    TinyCheckpoint,
    greedy_reference,
    tiny_engine_config,
    tiny_tokenizer,
    write_tiny_gpt2,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.unit

_FAKE = FakeTokenizer()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TinyCheckpoint:
    return write_tiny_gpt2(tmp_path_factory.mktemp("tiny-gpt2"))


@pytest.fixture
def llm(checkpoint: TinyCheckpoint) -> Iterator[LLM]:
    with LLM.from_config(
        tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer()
    ) as it:
        yield it


def _ids(text: str) -> list[int]:
    """The prompt ids the engine makes for ``text``: the byte tokenizer's, with its BOS."""
    return [BOS, *text.encode()]


def _greedy(max_tokens: int, **more: Any) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, **more)


def _baseline(
    llm: LLM, checkpoint: TinyCheckpoint, prompt: str, count: int
) -> tuple[list[int], str]:
    tokens = greedy_reference(checkpoint.state, checkpoint.config, _ids(prompt), count)
    return tokens, _FAKE.decode(tokens, skip_special_tokens=True)


class TestGenerate:
    def test_greedy_text_equals_the_reference_tokens_decoded(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        tokens, text = _baseline(llm, checkpoint, "hello", 10)

        out = llm.generate("hello", _greedy(10))

        assert not isinstance(out, list)
        [completion] = out.outputs
        assert completion.token_ids == tuple(tokens)
        assert completion.text == text
        assert completion.finish_reason is FinishReason.LENGTH
        assert out.finished
        assert out.prompt_token_ids == tuple(_ids("hello"))
        assert out.usage.prompt_tokens == 6
        assert out.usage.completion_tokens == 10
        assert out.metrics is not None
        assert out.metrics.ttft >= 0

    def test_a_list_comes_back_in_order_and_each_matches_its_own_reference(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        config = tiny_engine_config(checkpoint.directory, max_running=2, max_tokens_per_step=8)
        prompts = ["a first prompt", "hi", "the third one is longer than a block", "x"]
        with LLM.from_config(config, tokenizer=tiny_tokenizer()) as llm:
            outs = llm.generate(prompts, _greedy(6))

        assert isinstance(outs, list)
        assert len(outs) == 4
        for prompt, out in zip(prompts, outs, strict=True):
            expected = greedy_reference(checkpoint.state, checkpoint.config, _ids(prompt), 6)
            assert out.outputs[0].token_ids == tuple(expected), prompt

    def test_token_id_prompts_and_a_full_request(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        ids = tuple(_ids("abc"))
        request = GenerateRequest(request_id=RequestId("mine"), prompt=ids, sampling=_greedy(4))

        [by_ids, by_request] = llm.generate([ids, request], [_greedy(4), None])  # type: ignore[misc]

        expected = greedy_reference(checkpoint.state, checkpoint.config, list(ids), 4)
        assert by_ids.outputs[0].token_ids == tuple(expected)
        assert by_request.request_id == "mine"
        assert by_request.outputs[0].token_ids == tuple(expected)

    def test_a_request_may_not_bring_sampling_twice(self, llm: LLM) -> None:
        request = GenerateRequest(request_id=RequestId("r"), prompt="x", sampling=_greedy(2))

        with pytest.raises(ValueError, match="carries its own"):
            llm.generate([request], _greedy(2))

    def test_chat_renders_the_template_and_answers(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        out = llm.chat([ChatMessage(role="user", content="hi")], _greedy(5))

        assert out.finished
        assert out.prompt_token_ids is not None
        # The template text reached the model: the byte tokenizer has no control-token parsing,
        # so the rendered prompt is its bytes.
        assert bytes(out.prompt_token_ids).startswith(b"<|im_start|>user\nhi")
        prompt = list(out.prompt_token_ids)
        expected = greedy_reference(checkpoint.state, checkpoint.config, prompt, 5)
        assert out.outputs[0].token_ids == tuple(expected)

    def test_chat_accepts_plain_mappings(self, llm: LLM) -> None:
        out = llm.chat([{"role": "user", "content": "hi"}], _greedy(2))

        assert len(out.outputs[0].token_ids) == 2

    def test_sampling_is_reproducible_and_independent_of_the_batch(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        sampling = SamplingParams(temperature=1.0, max_tokens=12, seed=1234)
        config = tiny_engine_config(checkpoint.directory)
        with LLM.from_config(config, tokenizer=tiny_tokenizer()) as llm:
            alone = llm.generate("seeded", sampling)
            again = llm.generate("seeded", sampling)
            together = llm.generate(
                [
                    GenerateRequest(request_id=RequestId("s"), prompt="seeded", sampling=sampling),
                    GenerateRequest(
                        request_id=RequestId("o"),
                        prompt="some other prompt entirely",
                        sampling=SamplingParams(temperature=1.0, max_tokens=12, seed=7),
                    ),
                ]
            )

        assert not isinstance(alone, list)
        assert not isinstance(again, list)
        assert isinstance(together, list)
        assert alone.outputs[0].token_ids == again.outputs[0].token_ids
        assert alone.outputs[0].token_ids == together[0].outputs[0].token_ids


class TestStream:
    def test_chunks_add_up_to_the_whole_text_and_the_last_one_finishes(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        tokens, text = _baseline(llm, checkpoint, "stream me", 8)

        chunks = list(llm.stream("stream me", _greedy(8)))

        assert "".join(c.outputs[0].text for c in chunks) == text
        assert [t for c in chunks for t in c.outputs[0].token_ids] == tokens
        assert [c.finished for c in chunks] == [False] * (len(chunks) - 1) + [True]
        assert chunks[-1].outputs[0].finish_reason is FinishReason.LENGTH

    def test_closing_the_stream_early_cancels_the_request(self, llm: LLM) -> None:
        stream = llm.stream("walk away", _greedy(50))
        next(stream)

        stream.close()

        assert not llm.engine.has_unfinished
        assert llm.engine.kv.stats().num_requests == 0
        llm.engine.kv.check_integrity()

    def test_a_chat_stream_matches_the_chat_answer(self, llm: LLM) -> None:
        message = [{"role": "user", "content": "hi"}]
        whole = llm.chat(message, _greedy(6))

        text = "".join(c.outputs[0].text for c in llm.stream_chat(message, _greedy(6)))

        assert text == whole.outputs[0].text

    def test_a_refused_prompt_raises_when_the_stream_is_made(self, llm: LLM) -> None:
        with pytest.raises(UnsupportedFeatureError):
            llm.stream("x", SamplingParams(logprobs=1))

    def test_a_chat_message_without_a_role_is_invalid(self, llm: LLM) -> None:
        with pytest.raises(InvalidRequestError, match="role and content"):
            llm.chat([{"content": "no role"}])


class TestConcurrentStreams:
    """Two open generators on one LLM each see only their own request's outputs."""

    def test_two_streams_each_receive_only_their_own_tokens(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        prompts = ("first stream", "a second prompt that is longer")
        counts = (5, 8)
        expected = {
            key: _baseline(llm, checkpoint, prompt, count)[0]
            for key, prompt, count in zip((1, 2), prompts, counts, strict=True)
        }
        generators = {
            key: llm.stream(prompt, _greedy(count))
            for key, prompt, count in zip((1, 2), prompts, counts, strict=True)
        }
        chunks: dict[int, list[Any]] = {1: [], 2: []}
        finished = {1: False, 2: False}
        while not all(finished.values()):
            for key, generator in generators.items():
                if finished[key]:
                    continue
                try:
                    output = next(generator)
                except StopIteration:
                    finished[key] = True
                    continue
                chunks[key].append(output)

        request_ids = {key: {chunk.request_id for chunk in chunks[key]} for key in (1, 2)}
        assert len(request_ids[1]) == 1
        assert len(request_ids[2]) == 1
        assert request_ids[1] != request_ids[2]
        for key in (1, 2):
            assert [token for chunk in chunks[key] for token in chunk.outputs[0].token_ids] == (
                expected[key]
            )
            assert [chunk.finished for chunk in chunks[key]] == [False] * (len(chunks[key]) - 1) + [
                True
            ]
        assert not llm.engine.has_unfinished
        llm.engine.kv.check_integrity()

    def test_closing_one_stream_leaves_the_other_complete(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        other_tokens = _baseline(llm, checkpoint, "the other stream", 7)[0]
        abandoned = llm.stream("walk away", _greedy(50))
        other = llm.stream("the other stream", _greedy(7))
        next(abandoned)

        abandoned.close()

        chunks = list(other)
        assert [token for chunk in chunks for token in chunk.outputs[0].token_ids] == other_tokens
        assert chunks[-1].finished
        assert not llm.engine.has_unfinished
        llm.engine.kv.check_integrity()

    def test_a_blocking_generate_does_not_steal_a_streams_chunks(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        streamed_tokens = _baseline(llm, checkpoint, "streamed prompt", 6)[0]
        stream = llm.stream("streamed prompt", _greedy(6))
        first = next(stream)

        generated = llm.generate("blocking prompt", _greedy(3))
        rest = list(stream)

        assert not isinstance(generated, list)
        assert generated.outputs[0].token_ids
        assert [token for chunk in (first, *rest) for token in chunk.outputs[0].token_ids] == (
            streamed_tokens
        )
        assert rest[-1].finished
        assert not llm.engine.has_unfinished
        llm.engine.kv.check_integrity()


class TestStops:
    """Stop by token id, by string (split across tokens) and by length."""

    def test_max_tokens(self, llm: LLM) -> None:
        out = llm.generate("count", _greedy(7))

        assert not isinstance(out, list)
        completion = out.outputs[0]
        assert completion.finish_reason is FinishReason.LENGTH
        assert len(completion.token_ids) == 7
        assert completion.stop_reason is None

    def test_a_stop_token_id_ends_the_request_with_that_token(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        tokens, _ = _baseline(llm, checkpoint, "stop id", 12)
        stop = tokens[4]
        first = tokens.index(stop)

        out = llm.generate("stop id", _greedy(12, stop_token_ids=frozenset({stop})))

        assert not isinstance(out, list)
        completion = out.outputs[0]
        assert completion.token_ids == tuple(tokens[: first + 1])
        assert completion.finish_reason is FinishReason.STOP
        assert completion.stop_reason == stop

    def test_a_stop_string_cuts_the_text_and_the_tokens_after_it(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        tokens, text = _baseline(llm, checkpoint, "stop string", 16)
        stop = text[4:8]  # four characters: it spans four tokens
        at = text.index(stop)

        out = llm.generate("stop string", _greedy(16, stop=(stop,)))

        assert not isinstance(out, list)
        completion = out.outputs[0]
        assert completion.text == text[:at]
        assert completion.finish_reason is FinishReason.STOP
        assert completion.stop_reason == stop
        # The token that completed the match is the last one kept; nothing after it.
        assert completion.token_ids == tuple(tokens[: at + len(stop)])
        # The request is gone from the scheduler and its KV blocks are back.
        assert llm.engine.kv.stats().num_requests == 0
        assert not llm.engine.has_unfinished
        llm.engine.kv.check_integrity()

    def test_include_stop_str_in_output_keeps_the_match(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        _, text = _baseline(llm, checkpoint, "keep it", 16)
        stop = text[3:6]
        at = text.index(stop)

        out = llm.generate("keep it", _greedy(16, stop=(stop,), include_stop_str_in_output=True))

        assert not isinstance(out, list)
        assert out.outputs[0].text == text[: at + len(stop)]
        assert out.outputs[0].stop_reason == stop

    def test_a_stop_string_that_never_appears_changes_nothing(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        tokens, text = _baseline(llm, checkpoint, "no match", 9)

        out = llm.generate("no match", _greedy(9, stop=("\x00\x01",)))

        assert not isinstance(out, list)
        assert out.outputs[0].text == text
        assert out.outputs[0].token_ids == tuple(tokens)
        assert out.outputs[0].finish_reason is FinishReason.LENGTH

    def test_min_tokens_holds_a_stop_string_back(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        _, text = _baseline(llm, checkpoint, "min tokens", 10)

        out = llm.generate("min tokens", _greedy(10, min_tokens=10, stop=(text[2:5],)))

        assert not isinstance(out, list)
        assert out.outputs[0].text == text
        assert out.outputs[0].finish_reason is FinishReason.LENGTH

    def test_of_two_stop_strings_the_one_found_first_in_the_text_wins(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        _, text = _baseline(llm, checkpoint, "two", 16)
        early, late = text[2:4], text[9:12]

        out = llm.generate("two", _greedy(16, stop=(late, early)))

        assert not isinstance(out, list)
        first = min((text.index(early), early), (text.index(late), late))
        assert out.outputs[0].text == text[: first[0]]
        assert out.outputs[0].stop_reason == first[1]


class TestRefusals:
    @pytest.mark.parametrize(
        "sampling",
        [
            SamplingParams(temperature=0.8, n=2),
            SamplingParams(logprobs=1),
            SamplingParams(repetition_penalty=1.2),
            SamplingParams(presence_penalty=0.5),
            SamplingParams(logit_bias={3: 1.0}),
            SamplingParams(allowed_token_ids=frozenset({1, 2})),
        ],
        ids=["n", "logprobs", "repetition", "presence", "logit_bias", "allowed_ids"],
    )
    def test_an_unsupported_feature_is_refused_before_anything_is_queued(
        self, llm: LLM, sampling: SamplingParams
    ) -> None:
        with pytest.raises(UnsupportedFeatureError, match="does not support"):
            llm.generate("hello", sampling)

        assert not llm.engine.has_unfinished
        assert llm.engine.scheduler.num_waiting == 0

    def test_a_prompt_longer_than_the_context_is_invalid(self, llm: LLM) -> None:
        with pytest.raises(InvalidRequestError):
            llm.generate("x" * (MAX_POSITIONS + 5), _greedy(4))

    def test_max_tokens_beyond_the_context_is_invalid(self, llm: LLM) -> None:
        with pytest.raises(InvalidRequestError, match="model context"):
            llm.generate("hi", SamplingParams(max_tokens=MAX_POSITIONS + 1))

    def test_a_token_id_the_model_does_not_have_is_invalid(self, llm: LLM) -> None:
        with pytest.raises(InvalidRequestError):
            llm.generate((5000,), _greedy(2))

    def test_a_bad_prompt_in_a_batch_leaves_nothing_running(self, llm: LLM) -> None:
        with pytest.raises(InvalidRequestError):
            llm.generate(["fine", "x" * 500, "also fine"], _greedy(3))

        assert not llm.engine.has_unfinished
        assert llm.engine.kv.stats().num_requests == 0
        llm.engine.kv.check_integrity()
        # And the engine still works.
        assert not isinstance(llm.generate("after", _greedy(2)), list)

    def test_an_empty_conversation_is_invalid(self, llm: LLM) -> None:
        with pytest.raises(InvalidRequestError):
            llm.chat([])

    def test_stop_strings_need_a_tokenizer(self, llm: LLM) -> None:
        from ayaka.engine.input import require_supported

        with pytest.raises(UnsupportedFeatureError, match="without a tokenizer"):
            require_supported(SamplingParams(stop=("x",)), can_decode_text=False)

        require_supported(SamplingParams(stop=("x",)))  # with one it is fine


class TestFailures:
    def test_a_step_error_ends_its_requests_and_the_engine_goes_on(self, llm: LLM) -> None:
        runner = llm.engine._runner  # pyright: ignore[reportPrivateUsage]
        real = runner.execute
        calls = {"n": 0}

        def fail_once(step: Any, drafts: Any = None) -> StepOutput:
            calls["n"] += 1
            if calls["n"] == 1:
                return StepOutput(
                    step_id=step.step_id,
                    request_ids=step.request_ids,
                    error=ErrorInfo("backend_error", "injected"),
                )
            return real(step, drafts)

        runner.execute = fail_once  # type: ignore[method-assign]  # inject a failure

        failed = llm.generate("boom", _greedy(3))
        fine = llm.generate("boom", _greedy(3))

        assert not isinstance(failed, list)
        assert failed.outputs[0].finish_reason is FinishReason.ERROR
        assert failed.error is not None
        assert failed.error.code == "backend_error"
        assert not isinstance(fine, list)
        assert fine.outputs[0].finish_reason is FinishReason.LENGTH
        llm.engine.kv.check_integrity()

    def test_an_unplanned_exception_kills_the_engine_for_good(self, llm: LLM) -> None:
        runner = llm.engine._runner  # pyright: ignore[reportPrivateUsage]

        def explode(step: Any, drafts: Any = None) -> StepOutput:
            raise RuntimeError("kernel on fire")

        runner.execute = explode  # type: ignore[method-assign]  # inject a failure

        with pytest.raises(EngineDeadError, match="kernel on fire"):
            llm.generate("die", _greedy(3))
        with pytest.raises(EngineDeadError):
            llm.generate("again", _greedy(3))
        with pytest.raises(EngineDeadError):
            llm.engine.step()

    def test_a_shut_down_engine_refuses_work(self, checkpoint: TinyCheckpoint) -> None:
        llm = LLM.from_config(tiny_engine_config(checkpoint.directory), tokenizer=tiny_tokenizer())
        llm.close()
        llm.close()  # twice is fine

        with pytest.raises(EngineError, match="shut down"):
            llm.generate("late", _greedy(2))


class TestEngineCore:
    def test_abort_ends_a_running_request_with_cancelled_and_frees_its_cache(
        self, llm: LLM
    ) -> None:
        engine = llm.engine
        engine.add_request(
            GenerateRequest(request_id=RequestId("c"), prompt="cancel me", sampling=_greedy(50))
        )
        assert engine.step() == []  # the prefill emits one token but a non-streaming request waits

        assert engine.abort(RequestId("c"))
        [out] = engine.step()

        assert out.outputs[0].finish_reason is FinishReason.CANCELLED
        assert len(out.outputs[0].token_ids) >= 1
        assert not engine.has_unfinished
        assert engine.kv.stats().num_requests == 0
        assert not engine.abort(RequestId("c"))

    def test_streaming_chunks_add_up_to_the_whole_result(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        tokens, text = _baseline(llm, checkpoint, "stream", 9)
        llm.engine.add_request(
            GenerateRequest(request_id=RequestId("st"), prompt="stream", sampling=_greedy(9)),
            stream=True,
        )

        chunks = []
        while llm.engine.has_unfinished:
            chunks.extend(llm.engine.step())

        assert [c.finished for c in chunks] == [False] * (len(chunks) - 1) + [True]
        assert "".join(c.outputs[0].text for c in chunks) == text
        assert [t for c in chunks for t in c.outputs[0].token_ids] == tokens
        assert chunks[-1].usage.completion_tokens == 9

    def test_a_streaming_stop_string_never_leaks_the_held_back_tail(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        _, text = _baseline(llm, checkpoint, "leak", 16)
        stop = text[6:10]
        llm.engine.add_request(
            GenerateRequest(
                request_id=RequestId("lk"), prompt="leak", sampling=_greedy(16, stop=(stop,))
            ),
            stream=True,
        )

        chunks = []
        while llm.engine.has_unfinished:
            chunks.extend(llm.engine.step())

        assert "".join(c.outputs[0].text for c in chunks) == text[: text.index(stop)]
        assert chunks[-1].outputs[0].stop_reason == stop

    def test_a_duplicate_request_id_is_refused_and_the_first_still_finishes(self, llm: LLM) -> None:
        engine = llm.engine
        first = GenerateRequest(request_id=RequestId("dup"), prompt="one", sampling=_greedy(3))
        engine.add_request(first)

        with pytest.raises(InvalidRequestError, match="already in use"):
            engine.add_request(first)

        outputs = []
        while engine.has_unfinished:
            outputs.extend(engine.step())
        assert [o.request_id for o in outputs] == ["dup"]


class TestBuild:
    def test_the_plan_and_the_cache_agree_with_the_config(
        self, llm: LLM, checkpoint: TinyCheckpoint
    ) -> None:
        engine = llm.engine

        assert engine.model_config == checkpoint.config
        assert engine.plan.kv_blocks == (129,)
        assert engine.kv.stats().usable_blocks == (128,)
        assert engine.scheduler.max_running == 4

    def test_on_the_cpu_the_cache_is_sized_for_the_requests_that_run_not_for_all_of_ram(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        config = tiny_engine_config(checkpoint.directory, num_blocks=None, max_running=None)

        with LLM.from_config(config, tokenizer=tiny_tokenizer()) as llm:
            [usable] = llm.engine.kv.stats().usable_blocks

            assert llm.engine.plan.kv_blocks[0] > usable  # the plan would have used more
            assert llm.engine.scheduler.max_running == 8
            assert usable == 8 * (MAX_POSITIONS // 4)

    def test_a_missing_model_directory_is_a_model_load_error(self, tmp_path: Path) -> None:
        config = tiny_engine_config(tmp_path / "nowhere")

        with pytest.raises(ModelLoadError):
            LLM.from_config(config, tokenizer=tiny_tokenizer())

    def test_a_memory_ratio_too_small_for_the_weights_is_a_capacity_error(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        config = tiny_engine_config(checkpoint.directory, num_blocks=None)
        config = type(config)(
            source=config.source,
            device=config.device,
            cache=CacheConfig(block_size=4, memory_ratio=1e-9),
            scheduler=config.scheduler,
        )

        with pytest.raises(CapacityError, match="short"):
            LLM.from_config(config, tokenizer=tiny_tokenizer())

    def test_a_dtype_the_gpu_cannot_compute_in_is_refused_before_loading(
        self, checkpoint: TinyCheckpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        turing = DeviceInfo(
            kind="cuda",
            index=0,
            name="Turing",
            total_bytes=8 * 2**30,
            free_bytes=8 * 2**30,
            compute_capability=(7, 5),
        )
        host = DeviceInfo(kind="cpu", index=0, name="cpu", total_bytes=2**34, free_bytes=2**33)
        monkeypatch.setattr(device_info, "probe_devices", lambda: (turing, host))
        config = replace(
            tiny_engine_config(checkpoint.directory),
            device=DeviceConfig(device="cuda:0", dtype=DType.BF16),
        )

        with pytest.raises(ConfigError) as caught:
            LLM.from_config(config, tokenizer=tiny_tokenizer())

        assert caught.value.code == "DEVICE_CAPABILITY_MISSING"
        assert "sm_75 has no bf16 arithmetic" in str(caught.value)

    def test_cuda_graphs_are_refused_until_capture_is_implemented(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        config = replace(
            tiny_engine_config(checkpoint.directory),
            device=DeviceConfig(device="cuda:0", cuda_graph=True),
        )

        with pytest.raises(ConfigError) as caught:
            LLM.from_config(config, tokenizer=tiny_tokenizer())

        assert caught.value.code == "CUDA_GRAPH_UNSUPPORTED"

    def test_a_kv_reserve_larger_than_the_model_context_is_refused(
        self, checkpoint: TinyCheckpoint
    ) -> None:
        config = tiny_engine_config(checkpoint.directory)
        config = replace(
            config,
            cache=replace(config.cache, kv_reserve_tokens=MAX_POSITIONS + 1),
        )

        with pytest.raises(ConfigError) as caught:
            LLM.from_config(config, tokenizer=tiny_tokenizer())

        assert caught.value.code == "RESERVE_EXCEEDS_CONTEXT"

    def test_a_tokenizer_that_cannot_load_is_a_model_load_error(
        self, checkpoint: TinyCheckpoint, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(cls: Any, config: Any) -> None:
            raise OSError("no tokenizer.json here")

        monkeypatch.setattr(DefaultTokenizerFactory, "from_config", classmethod(broken))

        with pytest.raises(ModelLoadError, match=r"no tokenizer\.json here"):
            LLM.from_config(tiny_engine_config(checkpoint.directory))


def test_importing_the_engine_does_not_load_torch() -> None:
    code = (
        "import sys\n"
        "import ayaka, ayaka.engine\n"
        "from ayaka.engine.build import build_engine\n"
        "from ayaka.engine.llm import LLM\n"
        "from ayaka.engine.core import EngineCore\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
