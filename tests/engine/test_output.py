"""The output processor: detokenizing, stop strings and the rules about what may be emitted.

No model is involved: the tests feed it ``RequestEvent`` objects the way the scheduler would, over
the byte-level fake tokenizer, whose ids 0-255 are bytes, so a character of several bytes arrives
split across tokens, which is the case the incremental decoder and the stop matcher exist for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.engine.output import OutputProcessor
from ayaka.scheduler.request import RequestEvent
from tokenizer_fakes import EOS, make_service

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ayaka.core.outputs import RequestOutput
    from ayaka.tokenizers.service import TokenizerService

pytestmark = pytest.mark.unit

RID = RequestId("r")


class _Aborts:
    """Records the aborts the processor asks the scheduler for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, FinishReason]] = []

    def __call__(self, request_id: RequestId, reason: FinishReason) -> bool:
        self.calls.append((request_id, reason))
        return True


@pytest.fixture
def service() -> Iterator[TokenizerService]:
    with make_service() as it:
        yield it


def _processor(
    service: TokenizerService, aborts: _Aborts, *, stream: bool = False, **sampling: object
) -> OutputProcessor:
    processor = OutputProcessor(service, abort=aborts)
    processor.register(
        TokenizedRequest(
            request_id=RID,
            prompt_token_ids=(104, 105),
            sampling=SamplingParams(temperature=0.0, **sampling),  # type: ignore[arg-type]
        ),
        stream=stream,
    )
    return processor


def _event(
    *token_ids: int, finish: FinishReason | None = None, stop: int | None = None
) -> RequestEvent:
    return RequestEvent(request_id=RID, token_ids=token_ids, finish_reason=finish, stop_reason=stop)


def _bytes(text: str) -> tuple[int, ...]:
    return tuple(text.encode())


def _text(outputs: list[RequestOutput]) -> str:
    return "".join(o.outputs[0].text for o in outputs)


class TestDetokenizing:
    def test_a_character_split_across_tokens_is_emitted_whole_with_its_last_byte(
        self, service: TokenizerService
    ) -> None:
        processor = _processor(service, _Aborts(), stream=True)
        e_acute = "é".encode()  # two bytes, two tokens

        first = processor.process([_event(e_acute[0])])
        second = processor.process([_event(e_acute[1])])

        assert _text(first) == ""  # half a character is never emitted
        assert _text(second) == "é"

    def test_a_non_streaming_request_gives_one_output_with_everything(
        self, service: TokenizerService
    ) -> None:
        processor = _processor(service, _Aborts())

        assert processor.process([_event(*_bytes("he"))]) == []
        assert processor.process([_event(*_bytes("llo"))]) == []
        [out] = processor.process([_event(finish=FinishReason.LENGTH)])

        assert out.finished
        assert out.outputs[0].text == "hello"
        assert out.outputs[0].token_ids == _bytes("hello")
        assert out.outputs[0].finish_reason is FinishReason.LENGTH
        assert out.usage.prompt_tokens == 2
        assert out.usage.completion_tokens == 5
        assert processor.num_active == 0

    def test_special_tokens_are_not_in_the_text_but_stay_in_the_ids(
        self, service: TokenizerService
    ) -> None:
        processor = _processor(service, _Aborts())

        [out] = processor.process([_event(*_bytes("end"), EOS, finish=FinishReason.STOP, stop=EOS)])

        assert out.outputs[0].text == "end"
        assert out.outputs[0].token_ids == (*_bytes("end"), EOS)
        assert out.outputs[0].stop_reason == EOS

    def test_a_dangling_half_character_is_flushed_when_the_request_ends(
        self, service: TokenizerService
    ) -> None:
        processor = _processor(service, _Aborts())

        [out] = processor.process([_event("é".encode()[0], finish=FinishReason.LENGTH)])

        assert out.finished  # the decoder reports the broken character, it does not hang
        assert out.outputs[0].text in ("", "�")


class TestStopStrings:
    def test_a_stop_string_split_across_tokens_matches_cuts_and_aborts(
        self, service: TokenizerService
    ) -> None:
        aborts = _Aborts()
        processor = _processor(service, aborts, stop=("lo w",))

        assert processor.process([_event(*_bytes("hel"))]) == []
        assert processor.process([_event(*_bytes("lo"))]) == []  # "lo" could start "lo w"
        [out] = processor.process([_event(*_bytes(" w"))])

        assert out.finished
        assert out.outputs[0].text == "hel"
        assert out.outputs[0].finish_reason is FinishReason.STOP
        assert out.outputs[0].stop_reason == "lo w"
        assert out.outputs[0].token_ids == _bytes("hello w")
        assert aborts.calls == [(RID, FinishReason.STOP)]
        assert processor.num_active == 0

    def test_tokens_after_the_one_that_completes_the_match_are_dropped(
        self, service: TokenizerService
    ) -> None:
        processor = _processor(service, _Aborts(), stop=("cd",))

        [out] = processor.process([_event(*_bytes("abcdef"))])

        assert out.outputs[0].text == "ab"
        assert out.outputs[0].token_ids == _bytes("abcd")
        assert out.usage.completion_tokens == 4

    def test_the_stop_string_is_kept_on_request(self, service: TokenizerService) -> None:
        processor = _processor(service, _Aborts(), stop=("cd",), include_stop_str_in_output=True)

        [out] = processor.process([_event(*_bytes("abcdef"))])

        assert out.outputs[0].text == "abcd"

    def test_a_stop_string_made_of_a_multibyte_character_across_two_tokens(
        self, service: TokenizerService
    ) -> None:
        aborts = _Aborts()
        processor = _processor(service, aborts, stop=("é",))
        e_acute = "é".encode()

        assert processor.process([_event(*_bytes("caf"), e_acute[0])]) == []
        [out] = processor.process([_event(e_acute[1])])

        assert out.outputs[0].text == "caf"
        assert out.outputs[0].stop_reason == "é"
        assert aborts.calls == [(RID, FinishReason.STOP)]

    def test_streaming_holds_back_only_what_might_still_become_the_stop_string(
        self, service: TokenizerService
    ) -> None:
        processor = _processor(service, _Aborts(), stream=True, stop=("STOP",))

        chunks = [
            *processor.process([_event(*_bytes("go S"))]),
            *processor.process([_event(*_bytes("T"))]),
            *processor.process([_event(*_bytes("O"))]),
            *processor.process([_event(*_bytes("P"))]),
        ]

        assert _text(chunks) == "go "
        assert [c.finished for c in chunks][-1] is True

    def test_a_held_back_tail_is_released_when_the_request_ends_without_a_match(
        self, service: TokenizerService
    ) -> None:
        processor = _processor(service, _Aborts(), stream=True, stop=("STOP",))

        chunks = [
            *processor.process([_event(*_bytes("end ST"))]),
            *processor.process([_event(finish=FinishReason.LENGTH)]),
        ]

        assert _text(chunks) == "end ST"
        assert chunks[-1].outputs[0].finish_reason is FinishReason.LENGTH

    def test_a_stop_string_in_the_flushed_tail_still_counts(
        self, service: TokenizerService
    ) -> None:
        aborts = _Aborts()
        processor = _processor(service, aborts, stop=("zz",))

        [out] = processor.process([_event(*_bytes("azz"), finish=FinishReason.LENGTH)])

        assert out.outputs[0].text == "a"
        assert out.outputs[0].finish_reason is FinishReason.STOP
        # The scheduler had already ended the request, so there is nothing to abort.
        assert aborts.calls == []

    def test_min_tokens_keeps_the_matcher_off_until_enough_tokens_exist(
        self, service: TokenizerService
    ) -> None:
        processor = _processor(service, _Aborts(), stop=("ab",), min_tokens=4)

        assert processor.process([_event(*_bytes("ab"))]) == []  # two tokens: matching is off
        [out] = processor.process([_event(*_bytes("xab"))])  # the fifth token is allowed to stop

        assert out.outputs[0].text == "abx"
        assert out.outputs[0].stop_reason == "ab"


class TestFailures:
    def test_a_token_the_tokenizer_does_not_have_fails_that_request_only(
        self, service: TokenizerService
    ) -> None:
        aborts = _Aborts()
        processor = _processor(service, aborts)

        [out] = processor.process([_event(*_bytes("ok"), 299)])  # 299 is not a token

        assert out.finished
        assert out.outputs[0].finish_reason is FinishReason.ERROR
        assert out.error is not None
        assert out.error.code == "detokenization_failed"
        assert aborts.calls == [(RID, FinishReason.ERROR)]
        assert processor.num_active == 0

    def test_the_end_event_of_a_request_already_finished_here_is_ignored(
        self, service: TokenizerService
    ) -> None:
        processor = _processor(service, _Aborts(), stop=("cd",))
        processor.process([_event(*_bytes("abcd"))])

        assert processor.process([_event(finish=FinishReason.STOP)]) == []

    def test_an_error_carried_by_the_event_reaches_the_output(
        self, service: TokenizerService
    ) -> None:
        from ayaka.core.errors import ErrorInfo

        processor = _processor(service, _Aborts())
        event = RequestEvent(
            request_id=RID,
            finish_reason=FinishReason.TIMEOUT,
            error=ErrorInfo("request_timeout", "too slow", retryable=True),
        )

        [out] = processor.process([event])

        assert out.outputs[0].finish_reason is FinishReason.TIMEOUT
        assert out.error is not None
        assert out.error.code == "request_timeout"

    def test_fail_all_ends_every_unfinished_request(self, service: TokenizerService) -> None:
        from ayaka.core.errors import ErrorInfo

        processor = _processor(service, _Aborts())
        processor.process([_event(*_bytes("hi"))])

        [out] = processor.fail_all(ErrorInfo("engine_dead", "gone"))

        assert out.outputs[0].finish_reason is FinishReason.ERROR
        assert out.outputs[0].text == "hi"
        assert out.error is not None
        assert out.error.code == "engine_dead"
        assert processor.num_active == 0
