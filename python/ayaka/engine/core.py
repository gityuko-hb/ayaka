"""The engine loop: one thread, one step at a time, from requests in to outputs out.

:class:`EngineCore` owns the scheduler, the model runner, the KV cache manager and the two
processors, and runs the loop the scheduler and the runner were designed around::

    lookahead = runner.plan_lookahead()
    step = scheduler.schedule(lookahead)          # None: nothing to do
    output = runner.execute(step)
    scheduler.update(output)
    outputs = output_processor.process(scheduler.take_events())

It is not thread safe and takes no locks: the caller that owns the engine calls it from one thread.
:class:`~ayaka.engine.llm.LLM` is the synchronous caller; an asynchronous one would put a queue in
front of it.

Failure follows two rules. A request that fails, by a backend error in its step, a deadline or a
decoding problem, ends with an error in its own output and the engine goes on, as long as the KV
bookkeeping still passes its integrity check. Anything else, an exception nobody planned for or a
broken cache, makes the engine dead: it raises :class:`~ayaka.core.errors.EngineDeadError` from the
call that found out and from every call after it, rather than keep running on state it cannot check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.core.errors import EngineDeadError, EngineError
from ayaka.core.types import FinishReason

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.configs.model import ModelConfig
    from ayaka.core.outputs import RequestOutput
    from ayaka.core.params import SamplingParams
    from ayaka.core.request import GenerateRequest, TokenizedRequest
    from ayaka.core.types import RequestId
    from ayaka.engine.input import InputProcessor
    from ayaka.engine.output import OutputProcessor
    from ayaka.engine.profile import StartupReport
    from ayaka.execution.runner import ModelRunner
    from ayaka.kvcache.manager import KVCacheManager
    from ayaka.memory.budget import BudgetPlan
    from ayaka.scheduler.scheduler import Scheduler
    from ayaka.tokenizers.contracts import ChatMessage
    from ayaka.tokenizers.service import TokenizerService

__all__ = ["EngineCore"]


class EngineCore:
    """Admits requests, runs steps and reports outputs.

    ``build_engine`` makes one; the arguments are the parts it wired together.

    Args:
        scheduler: Decides each step.
        runner: Executes each step on the model.
        kv: The cache manager the scheduler uses; checked after a step fails.
        input_processor: Validates and tokenizes requests.
        output_processor: Detokenizes and applies the stop rules.
        tokenizer: The tokenizer service, closed by :meth:`shutdown`.
        model_config: The architecture of the loaded model.
        plan: The memory plan the cache was sized from.
        startup: What the start-up run measured and reserved on a GPU; ``None`` on the CPU.
    """

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        runner: ModelRunner,
        kv: KVCacheManager,
        input_processor: InputProcessor,
        output_processor: OutputProcessor,
        tokenizer: TokenizerService,
        model_config: ModelConfig,
        plan: BudgetPlan,
        startup: StartupReport | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._runner = runner
        self._kv = kv
        self._input = input_processor
        self._output = output_processor
        self._tokenizer = tokenizer
        self.model_config = model_config
        self.plan = plan
        self.startup = startup
        self._dead: EngineDeadError | None = None
        self._closed = False

    @property
    def tokenizer(self) -> TokenizerService:
        """The engine's tokenizer service."""
        return self._tokenizer

    @property
    def scheduler(self) -> Scheduler:
        """The scheduler, for reading its counters; requests go through the engine."""
        return self._scheduler

    @property
    def runner(self) -> ModelRunner:
        """The model runner, for reading what it captured; steps go through the engine."""
        return self._runner

    @property
    def kv(self) -> KVCacheManager:
        """The cache manager, for reading its statistics and checking its integrity."""
        return self._kv

    @property
    def has_unfinished(self) -> bool:
        """Whether any request is waiting, running or has output still to be delivered."""
        return self._scheduler.has_unfinished or self._output.num_active > 0

    # -- requests --------------------------------------------------------------------------

    def add_request(self, request: GenerateRequest, *, stream: bool = False) -> None:
        """Validate, tokenize and queue a request.

        Args:
            request: The request.
            stream: Deliver one output per step with only the new text and tokens, instead of one
                output with everything when the request ends.

        Raises:
            UnsupportedFeatureError: If the request needs a feature the engine does not have.
            InvalidRequestError: If it does not fit the model or its id is in use.
            OverloadedError: If the tokenizer or the waiting queue is full.
            EngineDeadError: If the engine died.
            EngineError: If the engine is shut down.
        """
        self._require_running()
        self._admit(self._input.process(request), stream)

    def add_chat(
        self,
        request_id: RequestId,
        messages: Sequence[ChatMessage],
        sampling: SamplingParams | None = None,
        *,
        stream: bool = False,
    ) -> None:
        """Render a conversation with the model's chat template and queue it.

        Raises:
            As :meth:`add_request`, and ``InvalidRequestError`` for an empty conversation.
        """
        self._require_running()
        self._admit(self._input.process_chat(request_id, messages, sampling), stream)

    def abort(self, request_id: RequestId) -> bool:
        """Cancel a request. Its final output, with ``FinishReason.CANCELLED``, comes from the
        next :meth:`step`.

        Returns:
            ``False`` if the request is unknown or already finished.
        """
        self._require_running()
        return self._scheduler.abort(request_id, FinishReason.CANCELLED)

    def _admit(self, request: TokenizedRequest, stream: bool) -> None:
        # The scheduler checks the id first: registering a duplicate would replace the output
        # state of the request that already has it.
        self._scheduler.add_request(request)
        try:
            self._output.register(request, stream=stream)
        except BaseException:
            self._scheduler.abort(request.request_id, FinishReason.CANCELLED)
            raise

    # -- the loop --------------------------------------------------------------------------

    def step(self) -> list[RequestOutput]:
        """Run one cycle of the loop and return the outputs it produced.

        An idle cycle, with nothing to run, still returns the outputs of requests that ended
        without a step: cancelled, timed out, or cut by a stop string. A step that ends requests
        with an error returns their outputs and leaves the engine running.

        Raises:
            EngineDeadError: If the engine is dead or dies in this cycle.
            EngineError: If the engine is shut down.
        """
        self._require_running()
        try:
            return self._cycle()
        except Exception as exc:
            raise self._die(exc) from exc

    def _cycle(self) -> list[RequestOutput]:
        scheduler = self._scheduler
        step = scheduler.schedule(self._runner.plan_lookahead())
        outputs = self._deliver()
        if step is None:
            if not outputs and self.has_unfinished:
                raise EngineError(
                    f"the scheduler has nothing to run but {self._output.num_active} "
                    "requests are unfinished"
                )
            return outputs
        answer = self._runner.execute(step)
        scheduler.update(answer)
        outputs += self._deliver()
        if answer.error is not None:
            self._kv.check_integrity()
        return outputs

    def _deliver(self) -> list[RequestOutput]:
        """Turn the scheduler's events into outputs, until the processors stop reacting.

        A stop string aborts its request, which queues one more event; looping lets that event be
        taken and ignored in the same cycle instead of surfacing a cycle later.
        """
        outputs: list[RequestOutput] = []
        while events := self._scheduler.take_events():
            outputs += self._output.process(events)
        return outputs

    # -- life cycle ------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop accepting work and close the tokenizer; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._tokenizer.close()

    def _require_running(self) -> None:
        if self._dead is not None:
            raise self._dead
        if self._closed:
            raise EngineError("the engine is shut down")

    def _die(self, cause: Exception) -> EngineDeadError:
        error = EngineDeadError(f"the engine stopped after an unexpected error: {cause}")
        self._dead = error
        return error
