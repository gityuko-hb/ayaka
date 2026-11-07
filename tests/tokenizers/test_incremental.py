from __future__ import annotations

from threading import RLock

import pytest

from ayaka.configs.tokenizer import DetokenizeParams
from ayaka.tokenizers.decoding.byte import ByteIncrementalDetokenizer, verify_byte_table
from ayaka.tokenizers.decoding.incremental import (
    FastIncrementalDetokenizer,
    IncrementalDetokenizer,
    SlowIncrementalDetokenizer,
)
from tokenizer_fakes import EOS, FakeTokenizer


def ids(text: str) -> list[int]:
    return list(text.encode())


def collect(detok: IncrementalDetokenizer, tokens: list[int]) -> str:
    """Feed one token at a time, then finish; return everything emitted."""
    out = ""
    for token in tokens:
        out += detok.update([token]).delta
    return out + detok.finish().delta


PARAMS = DetokenizeParams(accumulate_text=True)


@pytest.mark.unit
class TestBackendSelection:
    def test_auto_prefers_verified_bytes(self, verified_fake: FakeTokenizer) -> None:
        assert isinstance(
            IncrementalDetokenizer.create(verified_fake, PARAMS), ByteIncrementalDetokenizer
        )

    def test_auto_uses_stream_when_bytes_unverified(self, fake: FakeTokenizer) -> None:
        assert isinstance(IncrementalDetokenizer.create(fake, PARAMS), FastIncrementalDetokenizer)

    def test_auto_falls_back_to_slow_without_stream(self) -> None:
        detok = IncrementalDetokenizer.create(FakeTokenizer(stream=False), PARAMS)

        assert isinstance(detok, SlowIncrementalDetokenizer)

    def test_byte_path_is_skipped_when_special_tokens_are_kept(
        self, verified_fake: FakeTokenizer
    ) -> None:
        params = DetokenizeParams(skip_special_tokens=False)

        assert isinstance(
            IncrementalDetokenizer.create(verified_fake, params), FastIncrementalDetokenizer
        )

    def test_explicit_bytes_never_falls_back(self, fake: FakeTokenizer) -> None:
        with pytest.raises(ValueError, match="bytes backend"):
            IncrementalDetokenizer.create(fake, PARAMS, backend="bytes")

    def test_explicit_stream_never_falls_back(self) -> None:
        with pytest.raises(ValueError, match="DecodeStream"):
            IncrementalDetokenizer.create(FakeTokenizer(stream=False), PARAMS, backend="stream")

    def test_window_selects_the_slow_reference_path(self, fake: FakeTokenizer) -> None:
        assert isinstance(
            IncrementalDetokenizer.create(fake, PARAMS, backend="window"),
            SlowIncrementalDetokenizer,
        )

    def test_unknown_backend_is_rejected(self, fake: FakeTokenizer) -> None:
        with pytest.raises(ValueError, match="unknown detokenizer backend"):
            IncrementalDetokenizer.create(fake, PARAMS, backend="nope")  # type: ignore[arg-type]

    def test_shared_lock_is_used(self, fake: FakeTokenizer) -> None:
        lock = RLock()

        assert IncrementalDetokenizer.create(fake, PARAMS, lock=lock)._lock is lock


@pytest.mark.unit
class TestEveryPathDecodesTheSame:
    @pytest.mark.parametrize("backend", ["bytes", "stream", "window"])
    @pytest.mark.parametrize("text", ["hello", "héllo wörld", "日本語 🚀 mixed"])
    def test_matches_the_reference_decode(
        self, verified_fake: FakeTokenizer, backend: str, text: str
    ) -> None:
        detok = IncrementalDetokenizer.create(
            verified_fake,
            PARAMS,
            backend=backend,  # type: ignore[arg-type]
        )

        assert collect(detok, ids(text)) == text
        assert detok.output_text == text
        assert detok.n_chars == len(text)

    @pytest.mark.parametrize("backend", ["bytes", "stream", "window"])
    def test_partial_character_stalls_until_complete(
        self, verified_fake: FakeTokenizer, backend: str
    ) -> None:
        detok = IncrementalDetokenizer.create(
            verified_fake,
            PARAMS,
            backend=backend,  # type: ignore[arg-type]
        )
        first, second = "é".encode()

        stalled = detok.update([first])
        done = detok.update([second])

        assert stalled.delta == ""
        assert stalled.stalled
        assert done.delta == "é"
        assert not done.stalled

    @pytest.mark.parametrize("backend", ["bytes", "stream", "window"])
    def test_special_tokens_are_skipped(self, verified_fake: FakeTokenizer, backend: str) -> None:
        detok = IncrementalDetokenizer.create(
            verified_fake,
            PARAMS,
            backend=backend,  # type: ignore[arg-type]
        )

        assert collect(detok, [*ids("a"), EOS, *ids("b")]) == "ab"


@pytest.mark.unit
class TestStopAndLifecycle:
    def test_stop_ends_decoding_and_reports_consumed_tokens(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, DetokenizeParams(stop=("END",)))

        update = detok.update(ids("abcENDxyz"))

        assert update.delta == "abc"
        assert update.stop_matched == "END"
        assert update.consumed_tokens == 6

    def test_terminal_after_stop(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, DetokenizeParams(stop=("x",)))
        detok.update(ids("ax"))

        with pytest.raises(RuntimeError, match="terminal"):
            detok.update(ids("more"))
        assert detok.finish().delta == ""

    def test_finish_flushes_text_held_for_a_possible_stop(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, DetokenizeParams(stop=("END",)))

        first = detok.update(ids("abE"))

        assert first.delta == "a"
        assert detok.finish().delta == "bE"

    def test_finish_is_idempotent(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, PARAMS)
        detok.update(ids("hi"))
        detok.finish()

        assert detok.finish().delta == ""

    def test_update_after_finish_is_refused(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, PARAMS)
        detok.finish()

        with pytest.raises(RuntimeError, match="terminal"):
            detok.update(ids("x"))

    def test_include_stop_string(self, fake: FakeTokenizer) -> None:
        params = DetokenizeParams(stop=("END",), include_stop_str_in_output=True)

        update = IncrementalDetokenizer.create(fake, params).update(ids("aENDb"))

        assert update.delta == "aEND"

    def test_stop_matching_may_start_late_but_never_stop(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, DetokenizeParams(stop=("END",)))
        detok.update(ids("EN"), check_stops=False)
        detok.update(ids("a"), check_stops=True)

        with pytest.raises(ValueError, match="cannot be disabled"):
            detok.update(ids("b"), check_stops=False)

    def test_unmatched_stop_text_before_min_tokens_is_not_cut(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, DetokenizeParams(stop=("END",)))

        update = detok.update(ids("END"), check_stops=False)

        assert update.delta == "END"
        assert update.stop_matched is None

    def test_unknown_token_id_is_rejected_and_not_terminal(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, PARAMS)

        with pytest.raises(ValueError, match="unknown token id"):
            detok.update([999])
        assert detok.update(ids("ok")).delta == "ok"

    def test_output_text_requires_accumulation(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, DetokenizeParams())

        with pytest.raises(RuntimeError, match="accumulate_text"):
            _ = detok.output_text

    def test_prompt_is_not_part_of_the_output(self, fake: FakeTokenizer) -> None:
        detok = IncrementalDetokenizer.create(fake, PARAMS, prompt_token_ids=ids("prompt "))

        assert collect(detok, ids("answer")) == "answer"

    def test_byte_path_rejects_prompt_ending_mid_character(
        self, verified_fake: FakeTokenizer
    ) -> None:
        with pytest.raises(ValueError, match="inside a UTF-8 character"):
            IncrementalDetokenizer.create(
                verified_fake, PARAMS, prompt_token_ids=[ids("é")[0]], backend="bytes"
            )


@pytest.mark.unit
class TestVerifyByteTable:
    def test_accepts_a_correct_table(self, fake: FakeTokenizer) -> None:
        assert verify_byte_table(fake, ["hello", "héllo"])

    def test_rejects_a_table_that_disagrees_with_decode(self) -> None:
        class Broken(FakeTokenizer):
            def token_bytes_table(self) -> tuple[bytes, ...]:
                table = list(super().token_bytes_table())
                table[ord("a")] = b"b"
                return tuple(table)

        assert not verify_byte_table(Broken(), ["banana"], n_random=0)

    def test_rejects_an_empty_table(self) -> None:
        class Empty(FakeTokenizer):
            def token_bytes_table(self) -> tuple[bytes, ...]:
                return ()

        assert not verify_byte_table(Empty(), ["x"])
