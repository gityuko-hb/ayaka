"""Tokenizing in separate processes, for CPU-heavy encode and stateful decode.

Threads share the GIL and the tokenizer; processes do not. This pool starts a few
single-worker *spawn* processes, each holding its own :class:`TokenizerService`::

    caller process                      worker process k  (one TokenizerService each)
    --------------                      ---------------------------------------------
    ProcessTokenizerPool.submit  ----->  _encode(value, options)  -> EncodeResult
    pool.new_detokenizer()       ----->  _decoder("create")       keeps the decoder here
      ProcessDetokenizer.update  ----->  _decoder("update")       same worker, every time
      ProcessDetokenizer.finish  ----->  _decoder("finish")       decoder dropped

A request's decoder is pinned to one worker for its whole life, because its state
(partial characters, stop buffer) cannot move between processes. Admission is bounded
by job count and by payload bytes; a cancelled future keeps its charge until the child
actually stops using it.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ayaka.configs.tokenizer import DetokenizeParams, DetokUpdate, EncodeResult, TokenizerConfig
from ayaka.tokenizers.contracts import (
    ChatInput,
    EncodeInput,
    EncodeOptions,
    encode_input_weight,
)
from ayaka.tokenizers.errors import TokenizerOverloaded
from ayaka.tokenizers.service import TokenizerService
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayaka.tokenizers.decoding.incremental import IncrementalDetokenizer

__all__ = ["ProcessDetokenizer", "ProcessTokenizerPool"]

_DEFAULT_OPTIONS = EncodeOptions()
_ID_BYTES = 8  # budget charged per media placeholder token

# State of a worker process; only touched inside the child.
_service: TokenizerService | None = None
_decoders: dict[str, IncrementalDetokenizer] = {}


def _initialize(
    config: TokenizerConfig,
    max_model_len: int,
    model_vocab_size: int,
    factory: Callable[[], TokenizerService] | None,
) -> None:
    """Build this worker's service (runs once, in the child)."""
    global _service
    _service = (
        factory()
        if factory is not None
        else TokenizerService(
            config, max_model_len=max_model_len, model_vocab_size=model_vocab_size
        )
    )


def _require_service() -> TokenizerService:
    if _service is None:
        raise RuntimeError("tokenizer worker is not initialized")
    return _service


def _encode(value: EncodeInput, options: EncodeOptions) -> EncodeResult:
    return _require_service().encode(value, options)


def _encode_media(
    value: ChatInput,
    options: EncodeOptions,
    markers: tuple[tuple[str, int], ...],
    placeholder: int,
) -> tuple[EncodeResult, tuple[int, ...]]:
    from ayaka.tokenizers.encoding.media import encode_media_chat

    return encode_media_chat(_require_service(), value, options, markers, placeholder)


def _decoder(operation: str, identity: str, *args: Any) -> DetokUpdate | None:
    """Create, drive or drop the decoder named ``identity`` (runs in the child)."""
    if operation == "create":
        params, prompt = args
        _decoders[identity] = _require_service().new_detokenizer(params, prompt_token_ids=prompt)
        return None
    if operation == "forget":
        _decoders.pop(identity, None)
        return None
    decoder = _decoders[identity]
    if operation == "update":
        tokens, check_stops = args
        return decoder.update(tokens, check_stops=check_stops)
    if operation == "finish":
        try:
            return decoder.finish()
        finally:
            del _decoders[identity]
    raise ValueError("unknown decoder operation")


class ProcessDetokenizer:
    """One request's decoder, pinned to one worker until it finishes or is closed.

    Implements :class:`~ayaka.tokenizers.ports.DetokenizerLike`, so the engine treats
    it like an in-process detokenizer.

    Attributes:
        _pool: The pool that owns the worker.
        _worker: Index of the pinned worker.
        _identity: Key of this decoder inside the worker.
        _finished: ``finish()`` or ``close()`` already ran.
    """

    def __init__(self, pool: ProcessTokenizerPool, worker: int, identity: str) -> None:
        self._pool = pool
        self._worker = worker
        self._identity = identity
        self._finished = False

    def update(self, new_token_ids: Sequence[int], *, check_stops: bool = True) -> DetokUpdate:
        """Decode new ids in the worker and return the resulting text update."""
        if self._finished:
            raise RuntimeError("detokenizer is terminal")
        result = self._pool.submit_call(
            self._worker, _decoder, "update", self._identity, tuple(new_token_ids), check_stops
        ).result()
        assert isinstance(result, DetokUpdate)  # noqa: S101  # "update" always returns one
        return result

    def finish(self) -> DetokUpdate:
        """Flush the worker's decoder and drop it; repeats return an empty update."""
        if self._finished:
            return DetokUpdate()
        result = self._pool.submit_call(self._worker, _decoder, "finish", self._identity).result()
        self._finished = True
        assert isinstance(result, DetokUpdate)  # noqa: S101  # "finish" always returns one
        return result

    def close(self) -> None:
        """Drop the worker's decoder without flushing (for abandoned requests)."""
        if not self._finished:
            self._pool.submit_call(self._worker, _decoder, "forget", self._identity).result()
            self._finished = True


class ProcessTokenizerPool:
    """Separate processes for encoding and for stateful incremental decoding.

    Attributes:
        max_model_len: Context length given to every worker.
        model_vocab_size: Model id capacity given to every worker.
        eos_token_ids: Stop ids the caller supplied for pre-sample masking.
    """

    def __init__(
        self,
        config: TokenizerConfig,
        *,
        max_model_len: int,
        model_vocab_size: int,
        workers: int = 2,
        max_pending: int = 64,
        eos_token_ids: tuple[int, ...] = (),
        factory: Callable[[], TokenizerService] | None = None,
    ) -> None:
        """Start ``workers`` single-process executors.

        Args:
            config: Tokenizer settings; ``encode_pool_workers`` must be ``0``
                because each child runs its tokenizer inline.
            max_model_len: Context length of the model.
            model_vocab_size: Number of ids the model can embed.
            workers: Number of worker processes (``>= 1``).
            max_pending: Maximum jobs queued plus running (``>= 1``).
            eos_token_ids: Stop ids to expose as :attr:`eos_token_ids`.
            factory: Optional importable zero-argument callable returning the
                child's service (for example to load offline).

        Raises:
            ValueError: If ``config.encode_pool_workers`` is non-zero.
        """
        require_int(workers, "workers", minimum=1)
        require_int(max_pending, "max_pending", minimum=1)
        if config.encode_pool_workers:
            raise ValueError("process workers require inline child tokenizers")
        self.max_model_len = max_model_len
        self.model_vocab_size = model_vocab_size
        self.eos_token_ids = eos_token_ids
        self._capacity = threading.BoundedSemaphore(max_pending)
        self._lock = threading.Lock()
        self._next = 0
        self._pending_bytes = 0
        self._max_pending_bytes = config.encode_max_pending_bytes
        self._closed = False
        self._workers = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize,
                initargs=(config, max_model_len, model_vocab_size, factory),
            )
            for _ in range(workers)
        ]

    def _pick_worker(self) -> int:
        with self._lock:
            worker = self._next % len(self._workers)
            self._next += 1
            return worker

    def submit_call[T](
        self,
        worker: int,
        function: Callable[..., T],
        *args: Any,
        payload_bytes: int = 0,
    ) -> Future[T]:
        """Run ``function(*args)`` on a specific worker, charged against the budgets.

        Raises:
            TokenizerOverloaded: If the job or byte budget is exhausted.
            RuntimeError: If the pool is closed.
        """
        if not self._capacity.acquire(blocking=False):
            raise TokenizerOverloaded("process tokenizer capacity exhausted")
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("process tokenizer pool is closed")
                if self._pending_bytes + payload_bytes > self._max_pending_bytes:
                    raise TokenizerOverloaded("process tokenizer byte capacity exhausted")
                future = self._workers[worker].submit(function, *args)
                self._pending_bytes += payload_bytes
        except BaseException:
            self._capacity.release()
            raise

        def release(_: Future[T]) -> None:
            with self._lock:
                self._pending_bytes -= payload_bytes
            self._capacity.release()

        future.add_done_callback(release)
        return future

    def submit(
        self, value: EncodeInput, options: EncodeOptions = _DEFAULT_OPTIONS
    ) -> Future[EncodeResult]:
        """Encode ``value`` on the next worker (round robin)."""
        weight = encode_input_weight(value) + len(options.cache_namespace.encode())
        return self.submit_call(self._pick_worker(), _encode, value, options, payload_bytes=weight)

    def encode(self, value: EncodeInput, options: EncodeOptions = _DEFAULT_OPTIONS) -> EncodeResult:
        """Blocking form of :meth:`submit`."""
        return self.submit(value, options).result()

    async def encode_async(
        self, value: EncodeInput, options: EncodeOptions = _DEFAULT_OPTIONS
    ) -> EncodeResult:
        """Awaitable form of :meth:`submit`."""
        return await asyncio.wrap_future(self.submit(value, options))

    def encode_multimodal(
        self,
        value: ChatInput,
        options: EncodeOptions,
        markers: tuple[tuple[str, int], ...],
        placeholder: int,
    ) -> tuple[EncodeResult, tuple[int, ...]]:
        """Encode a chat with media placeholders on a worker (see ``encoding.media``)."""
        weight = (
            encode_input_weight(value)
            + len(options.cache_namespace.encode())
            + sum(_ID_BYTES * count for _, count in markers)
        )
        return self.submit_call(
            self._pick_worker(),
            _encode_media,
            value,
            options,
            markers,
            placeholder,
            payload_bytes=weight,
        ).result()

    def new_detokenizer(
        self, params: DetokenizeParams, *, prompt_token_ids: Sequence[int] = ()
    ) -> ProcessDetokenizer:
        """Create a decoder pinned to the next worker."""
        worker, identity = self._pick_worker(), uuid4().hex
        self.submit_call(
            worker, _decoder, "create", identity, params, tuple(prompt_token_ids)
        ).result()
        return ProcessDetokenizer(self, worker, identity)

    def close(self) -> None:
        """Stop admission, cancel queued jobs and wait for the workers to exit."""
        with self._lock:
            self._closed = True
        for worker in self._workers:
            worker.shutdown(wait=True, cancel_futures=True)
