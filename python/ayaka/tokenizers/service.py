"""Bounded preprocessing service: admission, batching, caching and validation.

It owns no scheduler, GPU state or request publication. Its whole job is to turn an
:data:`~ayaka.tokenizers.contracts.EncodeInput` into a validated
:class:`~ayaka.configs.tokenizer.EncodeResult`, or to reject it quickly::

    submit() --> admission --> queue --> dispatcher --> worker pool --> _run_batch
                 (count and     (workers > 0 only;       |                |
                  byte budget;   coalesces for            |                v
                  TokenizerOverloaded  encode_batch_window_ms)        _encode_batch
                  when full)                                               |
                                 guard + render chat --> session cache --> adapter encode
                                                                           |
                                                       _result: id and length checks,
                                                       truncation --> EncodeResult

Chat content is untrusted: before a template renders it, every string in it goes
through a :class:`~ayaka.tokenizers.security.SpecialTokenGuard` built from the loaded
tokenizer's special tokens and the configured ``special_token_policy``. Plain text and
already-rendered prompts are not guarded.

Threading rules:

* Zero workers means synchronous: the call runs inline and returns a finished future.
  Positive workers coalesce jobs submitted within a bounded window.
* Adapter calls and cache access are serialized by one lock, to protect slow or
  custom tokenizers. Detokenizers made by :meth:`TokenizerService.new_detokenizer`
  do not take it.
* Cancellation cannot interrupt a native encode: its result is discarded and its
  admission charge stays held until the native call returns.

Checked ids are not checked again. An encode submitted with ``vouch=True`` remembers the tuple
of ids it returned (:class:`~ayaka.tokenizers.vouched.VouchedIds`);
:meth:`TokenizerService.take_vouched` hands that same tuple back with only the length rule
applied, so a server that encoded a prompt does not pay for checking every id again when the engine
admits it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from ayaka.configs.tokenizer import DetokenizeParams, EncodeResult, TokenizerConfig
from ayaka.core.errors import CancelledError, ContextLengthError
from ayaka.tokenizers.contracts import (
    ChatInput,
    EncodeInput,
    EncodeOptions,
    LoadedTokenizer,
    RenderedInput,
    TemplateInput,
    TextInput,
    TokenIdsInput,
    encode_input_weight,
)
from ayaka.tokenizers.decoding.incremental import IncrementalDetokenizer
from ayaka.tokenizers.errors import TokenizerOverloaded
from ayaka.tokenizers.factory import DefaultTokenizerFactory, validate_loading_config
from ayaka.tokenizers.ports import RenderedEncoderLike, TokenizerLike
from ayaka.tokenizers.security import SpecialTokenGuard, SpecialTokenPolicy
from ayaka.tokenizers.vouched import VouchedIds
from ayaka.utils.cancellation import CancellationToken
from ayaka.utils.validation import plain_token_ids, require_int

if TYPE_CHECKING:
    from types import TracebackType

__all__ = ["TokenizerService"]

_DEFAULT_OPTIONS = EncodeOptions()
_LOGGER = logging.getLogger(__name__)
_ID_BYTES = 8  # cache charge per token id

type _CacheKey = tuple[str, bool, bool, str]
"""(text, add_special_tokens, rendered, cache_namespace)"""


class _WorkerState(threading.local):
    """Per-thread flag: is this thread inside an encode completion?"""

    in_worker: bool = False


@dataclass(slots=True)
class _Job:
    """One admitted request.

    Attributes:
        value: The input to encode.
        options: Length and cache options for this request.
        token: Cancellation token shared with the caller.
        future: Completed with the result or an exception.
        weight: Bytes charged against the admission budget.
    """

    value: EncodeInput
    options: EncodeOptions
    token: CancellationToken
    future: Future[EncodeResult]
    weight: int
    vouch: bool = False


class TokenizerService:
    """Admission-controlled, batching, caching front end to one tokenizer.

    Attributes:
        config: The tokenizer settings (budgets, batching window, cache sizes).
        max_model_len: Context length; prompt plus reserved output must fit.
        model_vocab_size: Embedding and logit capacity; every id must be below it.
        loaded: The adapter and its capabilities.
        tokenizer: Shortcut for ``loaded.tokenizer`` (``None`` in token-ID-only mode).
    """

    def __init__(
        self,
        config: TokenizerConfig,
        *,
        max_model_len: int,
        model_vocab_size: int,
        loaded: LoadedTokenizer | None = None,
    ) -> None:
        """Validate the configuration against the loaded adapter and start workers.

        Args:
            config: Tokenizer settings.
            max_model_len: Context length of the model (``>= 1``).
            model_vocab_size: Number of ids the model can embed (``>= 1``).
            loaded: A ready adapter bundle (useful for tests and process workers);
                loaded from ``config`` when omitted.

        Raises:
            ValueError: If the loaded adapter contradicts the configuration or its
                vocabulary does not fit the model.
        """
        require_int(max_model_len, "max_model_len", minimum=1)
        require_int(model_vocab_size, "model_vocab_size", minimum=1)
        validate_loading_config(config)
        self.config = config
        self.max_model_len = max_model_len
        self.model_vocab_size = model_vocab_size
        self.loaded = loaded if loaded is not None else DefaultTokenizerFactory.from_config(config)
        if not isinstance(self.loaded, LoadedTokenizer):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("loaded must be LoadedTokenizer")
        self._check_consistency()

        self.tokenizer = self.loaded.tokenizer
        self._rendered_encoder = (
            self.tokenizer if isinstance(self.tokenizer, RenderedEncoderLike) else None
        )
        self._valid_ids: frozenset[int] | None = None
        self._guard: SpecialTokenGuard | None = None
        self._fingerprint = "disabled"
        if self.tokenizer is not None:
            self._valid_ids = self.tokenizer.valid_token_ids
            if self._valid_ids and (
                min(self._valid_ids) < 0 or max(self._valid_ids) >= model_vocab_size
            ):
                raise ValueError("tokenizer vocabulary exceeds model embedding/logit capacity")
            self._fingerprint = self.tokenizer.fingerprint()
            self._guard = SpecialTokenGuard(
                self.tokenizer.all_special_tokens, SpecialTokenPolicy(config.special_token_policy)
            )

        self._cv = threading.Condition()
        self._adapter_lock = threading.RLock()
        self._queue: deque[_Job] = deque()
        self._jobs: dict[int, _Job] = {}
        self._pending = 0
        self._pending_bytes = 0
        self._closed = False
        self._cache: OrderedDict[_CacheKey, tuple[int, ...]] = OrderedDict()
        self._cache_bytes = 0
        self._vouched = VouchedIds(
            max_entries=config.encode_max_pending,
            max_ids=max(1, config.encode_max_pending_bytes // _ID_BYTES),
        )
        self._local = _WorkerState()
        self._pool: ThreadPoolExecutor | None = None
        self._dispatcher: threading.Thread | None = None
        if config.encode_pool_workers:
            self._pool = ThreadPoolExecutor(
                max_workers=config.encode_pool_workers, thread_name_prefix="ayaka-encode"
            )
            self._dispatcher = threading.Thread(
                target=self._dispatch, args=(self._pool,), daemon=True, name="ayaka-encode-dispatch"
            )
            self._dispatcher.start()

    def _check_consistency(self) -> None:
        config, caps = self.config, self.loaded.capabilities
        if config.skip_tokenizer_init != (caps.backend == "disabled"):
            raise ValueError("skip_tokenizer_init and loaded backend disagree")
        explicit = {"hf": "hf", "slow": "slow", "tiktoken": "tiktoken"}.get(config.mode)
        if explicit is not None and not config.skip_tokenizer_init and explicit != caps.backend:
            raise ValueError("requested and loaded backend disagree")
        if config.detokenize_backend == "bytes" and not caps.byte_decode:
            raise ValueError("loaded adapter cannot use byte decoding")
        if config.detokenize_backend == "stream" and not caps.decode_stream:
            raise ValueError("loaded adapter cannot use stream decoding")

    def _require_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is disabled")
        return self.tokenizer

    # -- introspection ---------------------------------------------------------

    @property
    def pending(self) -> tuple[int, int]:
        """Charged request count and logical payload bytes, including running work."""
        with self._cv:
            return self._pending, self._pending_bytes

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """Sorted stop-token ids for pre-sample masking; empty when disabled."""
        if self.tokenizer is None:
            return ()
        return tuple(sorted(self.tokenizer.eos_token_ids))

    @property
    def chat_template(self) -> str | None:
        """The model's declared chat template, if the loaded adapter carries one."""
        template = getattr(self.tokenizer, "chat_template", None)
        return template if isinstance(template, str) else None

    # -- submission ------------------------------------------------------------

    def submit(
        self,
        value: EncodeInput,
        options: EncodeOptions = _DEFAULT_OPTIONS,
        *,
        cancellation: CancellationToken | None = None,
        vouch: bool = False,
    ) -> Future[EncodeResult]:
        """Admit ``value`` or reject it immediately; zero workers also run it inline.

        Args:
            value: What to encode.
            options: Output reservation, truncation and cache namespace.
            cancellation: Token to cancel the request; one is created if omitted.
            vouch: Remember the returned ids, so :meth:`take_vouched` accepts them without
                checking each one again.

        Returns:
            A future holding the :class:`EncodeResult` or the failure.

        Raises:
            TokenizerOverloaded: If the count or byte budget is exhausted.
            ValueError: If the input kind is unsupported by the loaded tokenizer, or
                the output reservation leaves no room for a prompt.
            RuntimeError: If the service is closed.
            CancelledError: If ``cancellation`` was already cancelled.
        """
        if not isinstance(options, EncodeOptions):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("options must be EncodeOptions")
        if cancellation is not None and not isinstance(cancellation, CancellationToken):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("cancellation must be CancellationToken")
        token = cancellation if cancellation is not None else CancellationToken()
        token.raise_if_cancelled()
        weight = encode_input_weight(value) + len(options.cache_namespace.encode())
        caps = self.loaded.capabilities
        if self.tokenizer is None and not isinstance(value, TokenIdsInput):
            raise ValueError("tokenizer disabled: only token-ID input is available")
        if not isinstance(value, TokenIdsInput) and not caps.text_encode:
            raise ValueError("loaded tokenizer does not support text encoding")
        if isinstance(value, ChatInput | TemplateInput) and not caps.chat:
            raise ValueError("loaded tokenizer does not support chat templates")
        if options.max_output_tokens >= self.max_model_len:
            raise ValueError("output reservation leaves no room for a prompt")
        job = _Job(value, options, token, Future(), weight, vouch)
        with self._cv:
            if self._closed:
                raise RuntimeError("tokenizer service is closed")
            if (
                self._pending >= self.config.encode_max_pending
                or self._pending_bytes + weight > self.config.encode_max_pending_bytes
            ):
                raise TokenizerOverloaded("tokenizer input capacity exhausted")
            self._pending += 1
            self._pending_bytes += weight
            self._jobs[id(job)] = job
            if self._pool is not None:
                self._queue.append(job)
                self._cv.notify_all()
        if self._pool is None:
            self._run_batch([job])
        return job.future

    def encode(
        self,
        value: EncodeInput,
        options: EncodeOptions = _DEFAULT_OPTIONS,
        *,
        cancellation: CancellationToken | None = None,
        vouch: bool = False,
    ) -> EncodeResult:
        """Blocking convenience over :meth:`submit`.

        Raises:
            RuntimeError: If called from an encode completion callback, where
                blocking would deadlock the worker pool.
        """
        if self._pool is not None and self._local.in_worker:
            raise RuntimeError("blocking encode must not run from a completion callback")
        return self.submit(value, options, cancellation=cancellation, vouch=vouch).result()

    async def encode_async(
        self,
        value: EncodeInput,
        options: EncodeOptions = _DEFAULT_OPTIONS,
        *,
        cancellation: CancellationToken | None = None,
        vouch: bool = False,
    ) -> EncodeResult:
        """Await a background encode without blocking the event loop.

        Cancelling the awaiting task cancels the underlying work too.

        Raises:
            ValueError: If the service has no worker threads.
        """
        if self._pool is None:
            raise ValueError("encode_async requires encode_pool_workers > 0")
        token = cancellation if cancellation is not None else CancellationToken()
        try:
            return await asyncio.wrap_future(
                self.submit(value, options, cancellation=token, vouch=vouch)
            )
        except asyncio.CancelledError:
            token.cancel("async encode cancelled")
            raise

    def render_chat(self, value: ChatInput | TemplateInput) -> str:
        """Render a chat input to its prompt text without tokenizing it.

        Serialized with encoding, so it is safe to call from any thread.
        """
        with self._adapter_lock:
            return self._render(value)[0]

    # -- dispatch and execution ------------------------------------------------

    def _dispatch(self, pool: ThreadPoolExecutor) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: bool(self._queue) or self._closed)
                if self._closed and not self._queue:
                    return
                deadline = time.monotonic() + self.config.encode_batch_window_ms / 1000
                while not self._closed and len(self._queue) < self.config.encode_max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(remaining)
                batch = [
                    self._queue.popleft()
                    for _ in range(min(len(self._queue), self.config.encode_max_batch))
                ]
            try:
                pool.submit(self._run_batch, batch)
            except Exception as exc:
                for job in batch:
                    job.future.set_running_or_notify_cancel()
                    self._finish_job(job, exc)

    def _finish_job(self, job: _Job, result: EncodeResult | BaseException) -> None:
        with self._cv:
            self._pending -= 1
            self._pending_bytes -= job.weight
            del self._jobs[id(job)]
            self._cv.notify_all()
        previous = self._local.in_worker
        self._local.in_worker = True
        try:
            if not job.future.cancelled():
                if job.token.is_cancelled:
                    job.future.set_exception(CancelledError("encode cancelled"))
                elif isinstance(result, BaseException):
                    job.future.set_exception(result)
                else:
                    job.future.set_result(result)
        except BaseException:
            # Future runs user callbacks inline after committing its result. A fatal
            # callback must not strand the other jobs in this batch.
            _LOGGER.exception("encode completion callback failed")
        finally:
            self._local.in_worker = previous

    def _run_batch(self, batch: list[_Job]) -> None:
        previous = self._local.in_worker
        self._local.in_worker = True
        active: list[_Job] = []
        for job in batch:
            if job.future.set_running_or_notify_cancel():
                active.append(job)
            else:
                self._finish_job(job, CancelledError("encode cancelled"))
        results: list[EncodeResult | BaseException]
        try:
            with self._adapter_lock:
                results = self._encode_batch(active)
        except BaseException as exc:
            results = [exc] * len(active)
        try:
            for job, result in zip(active, results, strict=True):
                self._finish_job(job, result)
        finally:
            self._local.in_worker = previous

    def _render(
        self, value: TextInput | RenderedInput | ChatInput | TemplateInput
    ) -> tuple[str, bool, bool]:
        """Return ``(text, add_special_tokens, rendered)`` for a text-bearing input."""
        if isinstance(value, TextInput):
            return value.text, value.add_special_tokens, False
        if isinstance(value, RenderedInput):
            return value.text, False, True
        tokenizer = self._require_tokenizer()
        if isinstance(value, TemplateInput):
            # Every string in the history and tool definitions is client-supplied, role included.
            text = tokenizer.apply_chat_template(
                self._sanitize(json.loads(value.messages_json)),
                tools=self._sanitize(json.loads(value.tools_json)),
            )
        else:
            # The template text itself (``chat_template``) is code, not content: never sanitized.
            text = tokenizer.apply_chat_template(
                [{"role": m.role, "content": self._sanitize(m.content)} for m in value.messages],
                add_generation_prompt=value.add_generation_prompt,
                chat_template=value.chat_template,
                continue_final_message=value.continue_final_message,
            )
        return text, False, True

    def _sanitize(self, content: Any) -> Any:
        """Apply the special-token policy to client-supplied chat content."""
        return content if self._guard is None else self._guard.sanitize(content)

    def _encode_texts(self, texts: list[str], *, special: bool, rendered: bool) -> list[list[int]]:
        """Encode texts; rendered prompts keep their control tokens where the adapter needs it."""
        if rendered and self._rendered_encoder is not None:
            return self._rendered_encoder.encode_rendered_batch(texts)
        return self._require_tokenizer().encode_batch(texts, add_special_tokens=special)

    # -- session cache ---------------------------------------------------------

    @staticmethod
    def _cache_charge(key: _CacheKey, ids: tuple[int, ...]) -> int:
        return len(key[0].encode()) + len(key[3].encode()) + _ID_BYTES * len(ids)

    def _cache_put(self, key: _CacheKey, ids: tuple[int, ...]) -> None:
        size = self._cache_charge(key, ids)
        if size > self.config.session_cache_max_bytes:
            return
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_bytes -= self._cache_charge(key, old)
        self._cache[key] = ids
        self._cache_bytes += size
        while (
            len(self._cache) > self.config.session_cache_capacity
            or self._cache_bytes > self.config.session_cache_max_bytes
        ):
            removed, value = self._cache.popitem(last=False)
            self._cache_bytes -= self._cache_charge(removed, value)

    # -- encoding --------------------------------------------------------------

    def _encode_batch(self, jobs: list[_Job]) -> list[EncodeResult | BaseException]:
        results: list[EncodeResult | BaseException] = [
            RuntimeError("unprocessed encode") for _ in jobs
        ]
        groups: dict[tuple[bool, bool], list[tuple[int, str, _CacheKey]]] = {}
        rendered_bytes = 0
        for i, job in enumerate(jobs):
            try:
                job.token.raise_if_cancelled()
                if isinstance(job.value, TokenIdsInput):
                    results[i] = self._result(job.value.token_ids, job.options, reused=False)
                    continue
                text, special, rendered = self._render(job.value)
                size = len(text.encode())
                if rendered_bytes + size > self.config.encode_max_pending_bytes:
                    raise TokenizerOverloaded("rendered batch exceeds input budget")
                rendered_bytes += size
                # Without a dedicated rendered path, rendered and plain text encode
                # identically, so they may share a batch and a cache entry.
                path = rendered and self._rendered_encoder is not None
                key: _CacheKey = (text, special, path, job.options.cache_namespace)
                cached = self._cache.get(key) if self.config.enable_session_cache else None
                if cached is None:
                    groups.setdefault((special, path), []).append((i, text, key))
                    continue
                if self.config.verify_session_cache:
                    actual = tuple(self._encode_texts([text], special=special, rendered=path)[0])
                    if actual != cached:
                        raise ValueError("encode-cache verification mismatch")
                self._cache.move_to_end(key)
                results[i] = self._result(cached, job.options, reused=True)
            except Exception as exc:
                results[i] = exc
        for (special, path), group in groups.items():
            try:
                encoded = self._encode_texts(
                    [t for _, t, _ in group], special=special, rendered=path
                )
                if len(encoded) != len(group):
                    raise ValueError("encode_batch cardinality mismatch")
            except Exception as exc:
                for i, _, _ in group:
                    results[i] = exc
                continue
            for (i, _, key), ids in zip(group, encoded, strict=True):
                try:
                    jobs[i].token.raise_if_cancelled()
                    owned = tuple(ids)
                    results[i] = self._result(owned, jobs[i].options, reused=False)
                    if self.config.enable_session_cache:
                        self._cache_put(key, owned)
                except Exception as exc:
                    results[i] = exc
        for job, result in zip(jobs, results, strict=True):
            if job.vouch and isinstance(result, EncodeResult):
                self._vouched.vouch(result.token_ids)
        return results

    def _result(
        self, ids: tuple[int, ...], options: EncodeOptions, *, reused: bool
    ) -> EncodeResult:
        """Validate ids against model and tokenizer, apply the length limit, build the result."""
        if not self._known_ids(ids):
            for token in ids:
                require_int(token, "encoded token id")
                if token >= self.model_vocab_size or (
                    self._valid_ids is not None and token not in self._valid_ids
                ):
                    raise ValueError(f"token ID is not supported by model/tokenizer: {token}")
        fitted = self._fit(ids, options)
        return EncodeResult(
            fitted,
            len(fitted) if reused else 0,
            fitted is not ids,  # only truncation makes a new tuple
            extra=(("tokenizer_fingerprint", self._fingerprint),),
        )

    def _known_ids(self, ids: tuple[int, ...]) -> bool:
        """Whether every id is a plain int the model and the tokenizer have, at C speed.

        ``False`` only means the item-by-item check has to decide (and name the bad id).
        """
        if not plain_token_ids(ids):
            return False
        if not ids:
            return True
        if max(ids) >= self.model_vocab_size:
            return False
        return self._valid_ids is None or self._valid_ids.issuperset(ids)

    def _fit(self, ids: tuple[int, ...], options: EncodeOptions) -> tuple[int, ...]:
        """Apply the length limit: the ids unchanged, truncated, or an error.

        Raises:
            ContextLengthError: If the prompt and the output reservation exceed the context.
            ValueError: If the prompt exceeds ``max_input_tokens``, or nothing is left.
        """
        room = self.max_model_len - options.max_output_tokens
        limit = room if options.max_input_tokens is None else min(room, options.max_input_tokens)
        if len(ids) > limit:
            if not options.truncate:
                if len(ids) > room:
                    raise ContextLengthError(
                        prompt_tokens=len(ids),
                        output_tokens=options.max_output_tokens,
                        max_context_tokens=self.max_model_len,
                    )
                raise ValueError(f"the prompt has {len(ids)} tokens; max_input_tokens is {limit}")
            ids = ids[-limit:] if self.config.truncation_side == "left" else ids[:limit]
        if not ids:
            raise ValueError("empty encoded prompt")
        return ids

    def take_vouched(
        self, ids: tuple[int, ...], options: EncodeOptions = _DEFAULT_OPTIONS
    ) -> tuple[int, ...] | None:
        """Accept ids this service returned from an encode with ``vouch=True``, without
        checking each id again; the length rule of ``options`` still applies.

        Each vouching encode is good for one call. The ids must be the very tuple the encode
        returned: an equal tuple built elsewhere was not checked here.

        Returns:
            The ids, cut to fit if ``options`` asks for truncation; ``None`` if this service did
            not vouch for them (or no longer remembers it), and the caller checks them itself.

        Raises:
            ContextLengthError: If the prompt and the output reservation exceed the context.
            ValueError: As the length rule of :meth:`submit`.
        """
        if not self._vouched.take(ids):
            return None
        if options.max_output_tokens >= self.max_model_len:
            raise ValueError("output reservation leaves no room for a prompt")
        return self._fit(ids, options)

    # -- decoding --------------------------------------------------------------

    def new_detokenizer(
        self, params: DetokenizeParams, *, prompt_token_ids: tuple[int, ...] = ()
    ) -> IncrementalDetokenizer:
        """Create a per-request detokenizer with its own state and lock.

        It does not take the adapter lock that serializes encoding: decoding reads only the
        adapter's read-only tables and the tokenizer's thread-safe decode, so a long prompt
        being encoded on another thread never holds up token delivery. Only the end of the
        prompt is replayed, so the cost does not grow with the prompt.

        Raises:
            ValueError: If the tokenizer is disabled or cannot decode.
            RuntimeError: If the service is closed.
        """
        if self.tokenizer is None or not self.loaded.capabilities.text_decode:
            raise ValueError("tokenizer disabled: text output is unavailable")
        with self._cv:
            if self._closed:
                raise RuntimeError("tokenizer service is closed")
        return IncrementalDetokenizer.create(
            self.tokenizer,
            params,
            prompt_token_ids=prompt_token_ids,
            backend=self.config.detokenize_backend,
        )

    # -- lifecycle -------------------------------------------------------------

    def close(self, *, cancel_pending: bool = True) -> None:
        """Stop admission and wait for native work to finish; safe to call repeatedly.

        Args:
            cancel_pending: Cancel jobs that have not started running.

        Raises:
            RuntimeError: If called from an encode completion callback.
        """
        if self._local.in_worker:
            raise RuntimeError("close must not run from an encode completion callback")
        with self._cv:
            self._closed = True
            cancelled = [
                job for job in self._jobs.values() if cancel_pending and not job.future.running()
            ]
            self._cv.notify_all()
        for job in cancelled:
            job.token.cancel("tokenizer service closed")
        if self._dispatcher is not None:
            self._dispatcher.join()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        with self._cv:
            self._cv.wait_for(lambda: self._pending == 0)
        with self._adapter_lock:
            self._cache.clear()
            self._cache_bytes = 0
        self._vouched.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
