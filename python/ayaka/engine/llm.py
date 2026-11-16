"""The synchronous face of the engine: give it prompts, get generated text back.

::

    llm = LLM.from_config(EngineConfig(source=ModelSourceConfig("path/to/model")))
    answer = llm.generate("The capital of France is", SamplingParams(max_tokens=16))
    print(answer.outputs[0].text)

:class:`LLM` runs the engine loop on the calling thread until every request it was given has
finished, so a call returns complete results. A list of prompts is batched: the scheduler runs them
together, and the results come back in the order of the prompts, whatever order they finish in.
"""

from __future__ import annotations

import contextlib
from collections import deque
from typing import TYPE_CHECKING, Self, cast

from ayaka.core.errors import EngineError, InvalidRequestError, OverloadedError
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.tokenizers.contracts import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence
    from types import TracebackType

    from ayaka.configs.engine import EngineConfig
    from ayaka.core.outputs import RequestOutput
    from ayaka.engine.core import EngineCore
    from ayaka.tokenizers.contracts import LoadedTokenizer

__all__ = ["LLM"]

type Prompt = str | tuple[int, ...] | GenerateRequest
"""One prompt: text, a tuple of token ids, or a full request that carries its own parameters."""


def _is_one_prompt(prompts: object) -> bool:
    """Whether ``prompts`` is a single prompt rather than a sequence of them.

    A tuple of ints is one prompt of token ids; a tuple of anything else is a sequence.
    """
    if isinstance(prompts, (str, GenerateRequest)):
        return True
    if isinstance(prompts, tuple):
        items = cast("tuple[object, ...]", prompts)
        return bool(items) and all(isinstance(token, int) for token in items)
    return False


def _turns(messages: Sequence[ChatMessage | Mapping[str, str]]) -> list[ChatMessage]:
    """The messages as ``ChatMessage`` values; a mapping needs ``role`` and ``content``."""
    turns: list[ChatMessage] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            turns.append(message)
            continue
        try:
            turns.append(ChatMessage(role=message["role"], content=message["content"]))  # type: ignore[arg-type]  # ChatMessage checks the role
        except KeyError as exc:
            raise InvalidRequestError(f"a chat message needs role and content: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(f"invalid chat message: {exc}") from exc
    return turns


class LLM:
    """Generates text for prompts, one batch at a time, on the calling thread.

    Several generators may be open on one instance: each receives only the outputs of
    its own request, and a step taken by one generator also delivers the outputs of the
    others to their queues. The instance is not thread-safe; one thread owns it.

    Args:
        engine: A built engine; ``build_engine`` makes one, and :meth:`from_config` does both.
    """

    def __init__(self, engine: EngineCore) -> None:
        self._engine = engine
        self._counter = 0
        self._pending: dict[RequestId, deque[RequestOutput]] = {}

    @classmethod
    def from_config(cls, config: EngineConfig, *, tokenizer: LoadedTokenizer | None = None) -> LLM:
        """Build an engine from a configuration and wrap it.

        Args:
            config: What to load and how to run it.
            tokenizer: A tokenizer to use instead of loading the one the configuration names.

        Raises:
            ModelLoadError, ConfigError, CapabilityError, CapacityError: As ``build_engine``.
        """
        from ayaka.engine.build import build_engine

        return cls(build_engine(config, tokenizer=tokenizer))

    @property
    def engine(self) -> EngineCore:
        """The engine this object drives."""
        return self._engine

    def generate(
        self,
        prompts: Prompt | Sequence[Prompt],
        sampling: SamplingParams | Sequence[SamplingParams | None] | None = None,
    ) -> RequestOutput | list[RequestOutput]:
        """Generate a completion for each prompt and wait for all of them.

        Args:
            prompts: One prompt, or a sequence of prompts. A prompt is text, a tuple of token
                ids, or a :class:`~ayaka.core.request.GenerateRequest` that has its own id and
                sampling parameters.
            sampling: The parameters for text and token prompts: one value for all of them, or one
                per prompt (an entry may be ``None``). ``None`` means the defaults of
                ``SamplingParams``. A ``GenerateRequest`` brings its own, so ``sampling`` may not
                be set for it.

        Returns:
            One :class:`~ayaka.core.outputs.RequestOutput` for a single prompt, a list in the
            order of the prompts for a sequence. A request that failed is not raised: its output
            has ``finish_reason`` ``ERROR`` (or ``TIMEOUT``) and an ``error``.

        Raises:
            InvalidRequestError, UnsupportedFeatureError: If a prompt cannot be served; no prompt
                of the call is left running.
            EngineDeadError: If the engine died.
        """
        single = _is_one_prompt(prompts)
        items = cast("Sequence[Prompt]", [prompts] if single else prompts)
        requests = self._requests(items, sampling)
        results = self._run(requests)
        ordered = [results[request.request_id] for request in requests]
        return ordered[0] if single else ordered

    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        sampling: SamplingParams | None = None,
    ) -> RequestOutput:
        """Answer a conversation, rendered with the model's chat template.

        Args:
            messages: The turns, each a :class:`~ayaka.tokenizers.contracts.ChatMessage` or a
                mapping with ``role`` and ``content``.
            sampling: Parameters; ``None`` means the defaults.

        Raises:
            InvalidRequestError: If the conversation is empty or malformed, or does not fit.
            UnsupportedFeatureError: If the parameters ask for something the engine lacks.
            EngineDeadError: If the engine died.
        """
        request_id = self._next_id()
        engine = self._engine
        engine.add_chat(request_id, _turns(messages), sampling)
        return self._drain({request_id})[request_id]

    def stream(
        self, prompt: Prompt, sampling: SamplingParams | None = None
    ) -> Generator[RequestOutput]:
        """Generate for one prompt and yield an output per step that has something new.

        Each output carries only the new text and token ids; the last has ``finished`` true, the
        finish reason and the usage. Closing the generator early cancels the request.

        Args:
            prompt: Text, token ids, or a ``GenerateRequest`` (which then brings its own sampling).
            sampling: Parameters for text and token prompts; ``None`` means the defaults.

        Raises:
            InvalidRequestError, UnsupportedFeatureError: If the prompt cannot be served; raised
                on the first ``next``.
            EngineDeadError: If the engine died.
        """
        [request] = self._requests([prompt], sampling)
        self._engine.add_request(request, stream=True)
        return self._follow(request.request_id)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        sampling: SamplingParams | None = None,
    ) -> Generator[RequestOutput]:
        """Like :meth:`stream`, for a conversation rendered with the model's chat template."""
        request_id = self._next_id()
        self._engine.add_chat(request_id, _turns(messages), sampling, stream=True)
        return self._follow(request_id)

    def _follow(self, request_id: RequestId) -> Generator[RequestOutput]:
        finished = False
        try:
            while not finished:
                queue = self._pending.get(request_id)
                if queue:
                    output = queue.popleft()
                    finished = output.finished
                    yield output
                    continue
                if not self._engine.has_unfinished:
                    # Every request ended and this one never produced a final output. Raising
                    # is safer than looping on an idle engine forever.
                    raise EngineError(f"request {request_id!r} ended without a final output")
                self._advance()
        finally:
            if not finished:
                self._cancel({request_id})
            self._forget(request_id)

    def close(self) -> None:
        """Shut the engine down; later calls raise ``EngineError``."""
        self._engine.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- internals -------------------------------------------------------------------------

    def _next_id(self) -> RequestId:
        self._counter += 1
        return RequestId(f"ayaka-{self._counter}")

    def _requests(
        self,
        items: Sequence[Prompt],
        sampling: SamplingParams | Sequence[SamplingParams | None] | None,
    ) -> list[GenerateRequest]:
        if isinstance(sampling, SamplingParams) or sampling is None:
            per_prompt: Sequence[SamplingParams | None] = [sampling] * len(items)
        else:
            per_prompt = list(sampling)
            if len(per_prompt) != len(items):
                raise ValueError(f"{len(per_prompt)} sampling values for {len(items)} prompts")
        requests: list[GenerateRequest] = []
        for item, params in zip(items, per_prompt, strict=True):
            if isinstance(item, GenerateRequest):
                if params is not None:
                    raise ValueError("a GenerateRequest carries its own sampling parameters")
                requests.append(item)
            elif isinstance(item, (str, tuple)):  # pyright: ignore[reportUnnecessaryIsInstance]
                requests.append(
                    GenerateRequest(
                        request_id=self._next_id(),
                        prompt=item,
                        sampling=params if params is not None else SamplingParams(),
                    )
                )
            else:
                raise TypeError(f"a prompt is text, token ids or a GenerateRequest, not {item!r}")
        return requests

    def _run(self, requests: list[GenerateRequest]) -> dict[RequestId, RequestOutput]:
        """Feed the requests to the engine as it has room and run it until all are finished."""
        engine = self._engine
        waiting = deque(requests)
        admitted: set[RequestId] = set()
        results: dict[RequestId, RequestOutput] = {}
        try:
            while waiting or len(results) < len(admitted):
                while waiting:
                    try:
                        engine.add_request(waiting[0])
                    except OverloadedError:
                        if not engine.has_unfinished:
                            raise
                        break
                    admitted.add(waiting.popleft().request_id)
                self._advance()
                self._collect(admitted, results)
                if len(results) < len(admitted) and not waiting and not engine.has_unfinished:
                    raise EngineError("the engine stopped before every admitted request ended")
        except BaseException:
            self._cancel(admitted - results.keys())
            raise
        finally:
            for request_id in admitted:
                self._forget(request_id)
        return results

    def _drain(self, request_ids: set[RequestId]) -> dict[RequestId, RequestOutput]:
        results: dict[RequestId, RequestOutput] = {}
        try:
            while len(results) < len(request_ids):
                self._advance()
                self._collect(request_ids, results)
                if len(results) < len(request_ids) and not self._engine.has_unfinished:
                    raise EngineError("the engine stopped before every request ended")
        except BaseException:
            self._cancel(request_ids - results.keys())
            raise
        finally:
            for request_id in request_ids:
                self._forget(request_id)
        return results

    def _advance(self) -> None:
        """Run one engine step and queue each output under the request that produced it."""
        for output in self._engine.step():
            self._pending.setdefault(output.request_id, deque()).append(output)

    def _collect(
        self,
        request_ids: Iterable[RequestId],
        results: dict[RequestId, RequestOutput],
    ) -> None:
        """Move the final outputs of ``request_ids`` from their queues into ``results``."""
        for request_id in request_ids:
            queue = self._pending.get(request_id)
            if queue is None:
                continue
            while queue:
                output = queue.popleft()
                if output.finished:
                    results[request_id] = output

    def _forget(self, request_id: RequestId) -> None:
        """Drop the output queue of a request this object no longer owns."""
        self._pending.pop(request_id, None)

    def _cancel(self, request_ids: set[RequestId]) -> None:
        """Cancel requests this call left behind and let the engine finish them, if it can."""
        engine = self._engine
        # A dead or closed engine has nothing left to clean up.
        with contextlib.suppress(EngineError):
            for request_id in request_ids:
                engine.abort(request_id)
            while engine.has_unfinished:
                self._advance()
