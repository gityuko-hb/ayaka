"""Append-only incremental detokenization with explicit stop and finish state.

One detokenizer serves one request. It turns the token ids the model generates into
text that is only ever appended, never rewritten, and fails loudly instead of
corrupting output when a decoder would have to rewrite text it already emitted::

    IncrementalDetokenizer            (this module: stop matching, state, locking)
        |  create() picks the first supported path, never silently falling back
        +-- ByteIncrementalDetokenizer     decoding.byte      verified byte table
        +-- FastIncrementalDetokenizer     native DecodeStream, checked against the
        |                                  reference decode at finish()
        +-- SlowIncrementalDetokenizer     re-decodes the whole output per token

    token ids --> _decode_next --> raw text --> StopChecker --> DetokUpdate.delta

Cost model: byte and stream paths do constant work per token. The slow path is
quadratic in output length and exists for correctness, not speed. The stream path
runs one reference decode over prompt plus output when the request finishes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import TYPE_CHECKING

from ayaka.configs.tokenizer import DetokBackend, DetokenizeParams, DetokUpdate
from ayaka.tokenizers.decoding.stop import StopChecker
from ayaka.tokenizers.ports import ByteTokenizerLike, DecodeStreamLike, TokenizerLike
from ayaka.utils.validation import require_bool, require_int

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FastIncrementalDetokenizer",
    "IncrementalDetokenizer",
    "SlowIncrementalDetokenizer",
]

_REPLACEMENT = "�"
_BACKENDS: tuple[str, ...] = ("auto", "bytes", "stream", "window")


class IncrementalDetokenizer(ABC):
    """Base class: stop matching, lifecycle and locking around a token decoder.

    Subclasses implement only how one token becomes text
    (:meth:`_decode_next`), which ids are legal (:meth:`_validate_token`) and how
    to flush the decoder (:meth:`_decode_tail`). Use :meth:`create` to get the right
    subclass for a tokenizer.

    Attributes:
        _params: The :class:`~ayaka.configs.tokenizer.DetokenizeParams` of the request.
        _stop: Stop-string matcher fed with every piece of decoded text.
        _text: Emitted pieces, kept only when ``params.accumulate_text`` is set.
        _n_chars: Characters emitted so far.
        _finished: ``finish()`` ran or a decode error ended the stream.
        _stopped: A stop string matched; no further tokens are accepted.
        _checking_stops: ``None`` before the first update, then whether stop matching
            is on. It may be switched on later but never off again.
        _lock: Serializes calls; shared with the tokenizer service when given.
    """

    def __init__(self, params: DetokenizeParams, *, lock: RLock | None = None) -> None:
        if not isinstance(params, DetokenizeParams):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("params must be DetokenizeParams")
        self._params = params
        self._stop = StopChecker(params.stop, params.include_stop_str_in_output)
        self._text: list[str] = []
        self._n_chars = 0
        self._finished = False
        self._stopped = False
        self._checking_stops: bool | None = None
        self._lock = lock if lock is not None else RLock()

    @staticmethod
    def create(
        tokenizer: TokenizerLike,
        params: DetokenizeParams,
        *,
        prompt_token_ids: Sequence[int] = (),
        backend: DetokBackend = "auto",
        lock: RLock | None = None,
    ) -> IncrementalDetokenizer:
        """Pick a supported decoding path; an explicit choice never falls back.

        ``auto`` tries bytes, then stream, then the slow reference path. ``bytes``
        and ``stream`` raise ``ValueError`` when unavailable.

        Args:
            tokenizer: The adapter to decode with.
            params: Per-request decoding options.
            prompt_token_ids: Prompt ids, so the output continues from them.
            backend: Requested strategy (see ``DetokBackend``).
            lock: Lock to share with the encoder that uses the same adapter.

        Raises:
            ValueError: If ``backend`` is unknown or an explicit one is unsupported.
        """
        if backend not in _BACKENDS:
            raise ValueError(f"unknown detokenizer backend: {backend}")
        if not isinstance(params, DetokenizeParams):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("params must be DetokenizeParams")
        # Byte tables carry no policy for spaces around visible special tokens.
        if (
            backend in ("auto", "bytes")
            and params.skip_special_tokens
            and isinstance(tokenizer, ByteTokenizerLike)
            and tokenizer.byte_path_verified
        ):
            from ayaka.tokenizers.decoding.byte import ByteIncrementalDetokenizer

            return ByteIncrementalDetokenizer(
                tokenizer, params, prompt_token_ids=prompt_token_ids, lock=lock
            )
        if backend == "bytes":
            raise ValueError("bytes backend requires a verified byte table and skip_special_tokens")
        if backend in ("auto", "stream"):
            stream = tokenizer.new_decode_stream(skip_special_tokens=params.skip_special_tokens)
            if stream is not None:
                return FastIncrementalDetokenizer(
                    stream,
                    params,
                    tokenizer=tokenizer,
                    prompt_token_ids=prompt_token_ids,
                    lock=lock,
                )
            if backend == "stream":
                raise ValueError("tokenizer does not provide a DecodeStream")
        return SlowIncrementalDetokenizer(
            tokenizer, params, prompt_token_ids=prompt_token_ids, lock=lock
        )

    def _record(self, text: str) -> str:
        self._n_chars += len(text)
        if text and self._params.accumulate_text:
            self._text.append(text)
        return text

    def update(self, new_token_ids: Sequence[int], *, check_stops: bool = True) -> DetokUpdate:
        """Consume tokens until the first stop string; refuse updates after the end.

        Args:
            new_token_ids: Newly generated ids, in order.
            check_stops: ``False`` for tokens before ``min_tokens``. It may be
                turned on later but never off again once matching has started,
                because no stop string may span that boundary.

        Returns:
            What was decoded; ``consumed_tokens`` is smaller than the input when a
            stop matched part-way.

        Raises:
            RuntimeError: If the detokenizer already finished or stopped.
            ValueError: On an unknown id or an illegal ``check_stops`` switch.
        """
        with self._lock:
            return self._update(new_token_ids, check_stops=check_stops)

    def _update(self, new_token_ids: Sequence[int], *, check_stops: bool) -> DetokUpdate:
        if self._finished or self._stopped:
            raise RuntimeError("detokenizer is terminal")
        require_bool(check_stops, "check_stops")
        ids = tuple(new_token_ids)
        for token in ids:
            require_int(token, "token id")
            self._validate_token(token)
        if ids:
            if not check_stops and self._checking_stops is True:
                raise ValueError("stop matching cannot be disabled after it starts")
            self._checking_stops = check_stops
        emitted: list[str] = []
        matched: str | None = None
        consumed = 0
        for token in ids:
            try:
                raw = self._decode_next(token)
            except Exception:
                self._finished = True
                raise
            consumed += 1
            text, matched = self._stop.feed(raw) if check_stops else (raw, None)
            emitted.append(self._record(text))
            if matched is not None:
                self._stopped = True
                break
        delta = "".join(emitted)
        return DetokUpdate(
            delta=delta,
            stop_matched=matched,
            stalled=bool(consumed) and not delta,
            consumed_tokens=consumed,
        )

    def finish(self) -> DetokUpdate:
        """Flush the decoder and the stop buffer exactly once; repeats return empty."""
        with self._lock:
            return self._finish()

    def _finish(self) -> DetokUpdate:
        if self._finished:
            return DetokUpdate()
        self._finished = True
        if self._stopped:
            return DetokUpdate()
        tail = self._decode_tail()
        checking = self._checking_stops is not False
        emit, matched = self._stop.feed(tail) if checking else (tail, None)
        # A non-empty decoder tail can still sit entirely inside the stop buffer.
        emit += self._stop.flush()
        self._stopped = matched is not None
        return DetokUpdate(delta=self._record(emit), stop_matched=matched)

    @property
    def output_text(self) -> str:
        """All emitted text. Requires ``accumulate_text=True`` in the params."""
        with self._lock:
            if not self._params.accumulate_text:
                raise RuntimeError("output_text requires accumulate_text=True")
            return "".join(self._text)

    @property
    def n_chars(self) -> int:
        """Characters emitted so far."""
        with self._lock:
            return self._n_chars

    @abstractmethod
    def _validate_token(self, token_id: int) -> None:
        """Raise ``ValueError`` if ``token_id`` is not decodable."""

    @abstractmethod
    def _decode_next(self, token_id: int) -> str:
        """Decode one token to the text it contributes (possibly empty)."""

    def _decode_tail(self) -> str:
        """Text still buffered inside the decoder at the end of the stream."""
        return ""


class _ReferenceDecoder(IncrementalDetokenizer):
    """Shared base for decoders that check themselves against ``tokenizer.decode``.

    Attributes:
        _tok: The adapter used for reference decoding.
        _valid_ids: Legal token ids, shared with the adapter (not copied).
        _ids: Prompt ids followed by every generated id.
        _prefix: Reference decode of the prompt; output must extend it.
        _raw_text: Text already accounted for against the reference.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        params: DetokenizeParams,
        *,
        prompt_token_ids: Sequence[int] = (),
        lock: RLock | None = None,
    ) -> None:
        super().__init__(params, lock=lock)
        self._tok = tokenizer
        self._valid_ids = tokenizer.valid_token_ids
        self._ids = list(prompt_token_ids)
        for token in self._ids:
            require_int(token, "prompt token id")
            self._validate_token(token)
        self._prefix = self._decode()
        self._raw_text = ""

    def _validate_token(self, token_id: int) -> None:
        if token_id not in self._valid_ids:
            raise ValueError(f"unknown token id: {token_id}")

    def _decode(self) -> str:
        return self._tok.decode(
            self._ids,
            skip_special_tokens=self._params.skip_special_tokens,
            spaces_between_special_tokens=self._params.spaces_between_special_tokens,
        )

    def _reference_output(self) -> str:
        full = self._decode()
        if not full.startswith(self._prefix):
            raise ValueError("decoder rewrote the prompt boundary; cannot stream append-only")
        return full[len(self._prefix) :]

    def _decode_tail(self) -> str:
        reference = self._reference_output()
        if not reference.startswith(self._raw_text):
            raise ValueError("decoder rewrote emitted text; cannot stream append-only")
        return reference[len(self._raw_text) :]


class FastIncrementalDetokenizer(_ReferenceDecoder):
    """Native ``DecodeStream`` per token, verified against the reference at the end.

    There is no implicit reset: resetting without replaying the ids would lose the
    decoder's context, so a stream fault propagates and the caller must fail the
    request if it already emitted text.

    Attributes:
        _stream: The native incremental decoder.
        _pieces: Every piece the stream returned, compared with the reference at finish.
    """

    def __init__(
        self,
        stream: DecodeStreamLike,
        params: DetokenizeParams,
        *,
        tokenizer: TokenizerLike,
        prompt_token_ids: Sequence[int] = (),
        lock: RLock | None = None,
    ) -> None:
        super().__init__(tokenizer, params, prompt_token_ids=prompt_token_ids, lock=lock)
        self._stream = stream
        self._pieces: list[str] = []
        for token in self._ids:
            self._stream.step(token)

    def _decode_next(self, token_id: int) -> str:
        self._ids.append(token_id)
        text = self._stream.step(token_id) or ""
        self._pieces.append(text)
        return text

    def _decode_tail(self) -> str:
        self._raw_text = "".join(self._pieces)
        return super()._decode_tail()


class SlowIncrementalDetokenizer(_ReferenceDecoder):
    """Correctness-first fallback: re-decodes everything and buffers incomplete UTF-8."""

    def _decode_next(self, token_id: int) -> str:
        self._ids.append(token_id)
        reference = self._reference_output()
        if reference.endswith(_REPLACEMENT):
            return ""  # probably a character split across tokens; wait for the rest
        if not reference.startswith(self._raw_text):
            raise ValueError("decoder rewrote emitted text; cannot stream append-only")
        delta = reference[len(self._raw_text) :]
        self._raw_text = reference
        return delta
