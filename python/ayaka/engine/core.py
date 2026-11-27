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
front of it. The one exception is :meth:`EngineCore.prepare`, which validates and tokenizes a
request without touching engine state and may run on any thread, so that work whose cost grows with
the prompt stays off the thread that runs the steps; :meth:`EngineCore.admit` then queues the result
on the engine's own thread.

A request with ``n > 1`` is tokenized once and queued as ``n`` sibling requests, ``<id>:0`` to
``<id>:<n-1>``, that share the prompt, the seed and every other parameter and differ only in their
sequence index, which picks an independent random stream. With the prefix cache the siblings wait
for the first one to prefill the prompt and then attach it. The outputs carry the parent id, and
aborting the parent aborts every sibling.

After every step the scheduler's counts and the time the runner took go to the stat loggers, when
the engine has them; an idle cycle records nothing.

The runner's penalty counts follow the requests: a request that uses a penalty gets its counts
when it is queued, every token it emits is counted before the next step, and the counts are
dropped when it ends.

Failure follows two rules. A request that fails, by a backend error in its step, a deadline or a
decoding problem, ends with an error in its own output and the engine goes on, as long as the KV
bookkeeping still passes its integrity check. Anything else, an exception nobody planned for or a
broken cache, makes the engine dead: it raises :class:`~ayaka.core.errors.EngineDeadError` from the
call that found out and from every call after it, rather than keep running on state it cannot check.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

from ayaka.core.errors import EngineDeadError, EngineError, InvalidRequestError
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.engine.seam import EngineInfo
from ayaka.metrics.stats import IterationStats
from ayaka.sampling.rng import resolve_seed

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayaka.configs.model import ModelConfig
    from ayaka.core.outputs import RequestOutput
    from ayaka.core.request import GenerateRequest
    from ayaka.core.types import TokenId
    from ayaka.engine.input import InputProcessor
    from ayaka.engine.output import OutputProcessor
    from ayaka.engine.profile import StartupReport
    from ayaka.engine.recorder import SessionRecorder
    from ayaka.execution.runner import ModelRunner
    from ayaka.kvcache.manager import KVCacheManager
    from ayaka.memory.budget import BudgetPlan
    from ayaka.metrics.manager import StatLoggerManager
    from ayaka.scheduler.scheduler import Scheduler
    from ayaka.scheduler.stats import SchedulerStats
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
        stats: The stat loggers each step is recorded to, or ``None``.
        recorder: Writes every finished request as token ids (``--record``), or ``None``; it is
            closed by :meth:`shutdown`.
        clock: A monotonic clock in seconds, for the step time.
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
        stats: StatLoggerManager | None = None,
        recorder: SessionRecorder | None = None,
        clock: Callable[[], float] = time.perf_counter,
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
        self._stats = stats
        self._clock = clock
        self._families: dict[RequestId, tuple[RequestId, ...]] = {}
        self._recorder = recorder
        self._last_host_offloads = 0
        self._last_host_promotions = 0
        self._dead: EngineDeadError | None = None
        self._closed = False

    @property
    def tokenizer(self) -> TokenizerService:
        """The engine's tokenizer service."""
        return self._tokenizer

    @property
    def info(self) -> EngineInfo:
        """What a server needs to know about this engine."""
        longest = self._tokenizer.max_model_len
        return EngineInfo(
            max_model_len=longest,
            max_context_tokens=min(longest, self._kv.max_context_tokens),
            prefix_caching=self._kv.prefix_caching,
        )

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
    def stats(self) -> StatLoggerManager | None:
        """The stat loggers each step is recorded to."""
        return self._stats

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

    def prepare(self, request: GenerateRequest) -> TokenizedRequest:
        """Validate and tokenize a request for :meth:`admit`; safe to call from any thread.

        Raises:
            UnsupportedFeatureError, InvalidRequestError, OverloadedError: As
                :meth:`add_request`.
            EngineDeadError: If the engine died.
            EngineError: If the engine is shut down.
        """
        self._require_running()
        return self._input.process(request)

    def admit(self, request: TokenizedRequest, *, stream: bool = False) -> None:
        """Queue a request :meth:`prepare` made; on the engine's thread.

        Raises:
            InvalidRequestError: If its id is in use or it does not fit the KV cache.
            OverloadedError: If the waiting queue is full.
            EngineDeadError: If the engine died.
            EngineError: If the engine is shut down.
        """
        self._require_running()
        self._admit(request, stream)

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
        siblings = self._families.get(request_id)
        if siblings is None:
            return self._scheduler.abort(request_id, FinishReason.CANCELLED)
        found = [self._scheduler.abort(sibling, FinishReason.CANCELLED) for sibling in siblings]
        return any(found)

    def _admit(self, request: TokenizedRequest, stream: bool) -> None:
        if request.sampling.n > 1:
            self._admit_family(request, stream)
            return
        if request.request_id in self._families:
            raise InvalidRequestError(f"request id {request.request_id!r} is already in use")
        # The scheduler checks the id first: registering a duplicate would replace the output
        # state of the request that already has it.
        self._scheduler.add_request(request)
        try:
            self._output.register(request, stream=stream)
            self._runner.penalties.admit(
                request.request_id, request.prompt_token_ids, request.sampling
            )
            self._admit_constraint(request)
        except BaseException:
            self._scheduler.abort(request.request_id, FinishReason.CANCELLED)
            raise
        if self._recorder is not None:
            self._recorder.admitted(request)

    def _admit_constraint(self, request: TokenizedRequest) -> None:
        """Start the request's output constraint from its compiled grammar, if it has one."""
        if request.constraint_id is not None and request.structured is not None:
            matcher = self._input.compiler.matcher(request.constraint_id, request.structured)
            self._runner.constraints.admit(request.request_id, matcher)

    def _admit_family(self, request: TokenizedRequest, stream: bool) -> None:
        """Queue the ``n`` sequences of a request as siblings; all of them or none."""
        parent = request.request_id
        if parent in self._families or self._scheduler.get(parent) is not None:
            raise InvalidRequestError(f"request id {parent!r} is already in use")
        sampling = replace(request.sampling, n=1, seed=resolve_seed(parent, request.sampling.seed))
        children = tuple(
            replace(
                request,
                request_id=RequestId(f"{parent}:{index}"),
                sampling=sampling,
                sequence_index=index,
            )
            for index in range(request.sampling.n)
        )
        # Waiting for the first sibling only pays when the others can attach its prompt.
        wait = self._kv.prefix_caching and request.prefix_cache
        added: list[RequestId] = []
        try:
            for child in children:
                blocked_on = children[0].request_id if wait and added else None
                self._scheduler.add_request(child, blocked_on=blocked_on)
                added.append(child.request_id)
            self._output.register_family(parent, children, stream=stream)
            for child in children:
                self._runner.penalties.admit(
                    child.request_id, child.prompt_token_ids, child.sampling
                )
                self._admit_constraint(child)
        except BaseException:
            for child_id in added:
                self._scheduler.abort(child_id, FinishReason.CANCELLED)
            self._output.discard(children[0].request_id)
            raise
        self._families[parent] = tuple(added)
        if self._recorder is not None:
            self._recorder.admitted(request)  # the family's outputs carry the parent's id

    # -- the loop --------------------------------------------------------------------------

    def prewarm(
        self,
        prompt_token_ids: Sequence[TokenId],
        *,
        salt: str | None = None,
        pin: bool = False,
    ) -> int:
        """Prefill and cache full blocks of ``prompt_token_ids``, optionally pinning in VRAM.

        Returns:
            Number of full prefix blocks cached (and pinned when ``pin=True``).
        """
        self._require_running()
        if not self._kv.prefix_caching or len(prompt_token_ids) < self._kv.block_size:
            return 0
        req_id = RequestId(f"__prewarm__:{time.monotonic_ns()}")
        request = TokenizedRequest(
            request_id=req_id,
            prompt_token_ids=tuple(prompt_token_ids),
            sampling=SamplingParams(temperature=0.0, max_tokens=1),
            cache_salt=salt,
            prefix_cache=True,
        )
        self._admit(request, stream=False)
        while self._scheduler.get(req_id) is not None:
            self.step()
        if pin:
            return self._kv.pin_prefix(prompt_token_ids, salt)
        return self._kv.cached_prefix_blocks(prompt_token_ids, salt)

    def resize_kv_pool(self, new_num_blocks: int | Sequence[int]) -> tuple[int, ...]:
        """Atomically resize the engine's KV pool and rebind the model runner.

        Raises:
            EngineError: If a step is currently in flight or the engine is shut down/dead.
            CapacityError: If the requested size cannot fit active and pinned blocks.
            OutOfMemoryError: If reallocating storage fails (previous pool stays intact).
        """
        self._require_running()
        if self._scheduler.in_flight:
            raise EngineError("cannot resize KV pool while a step is in flight")
        targets = (
            (new_num_blocks,) * len(self._kv.groups)
            if isinstance(new_num_blocks, int)
            else tuple(new_num_blocks)
        )
        new_pools = self._kv.resize(targets)
        if new_pools:
            self._runner.rebind_kv_pools(new_pools)
        return self._kv.usable_counts

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
        started = self._clock()
        answer = self._runner.execute(step)
        seconds = self._clock() - started
        if answer.error is not None and answer.error.code == "out_of_memory":
            scheduler.recover_from_oom(answer.error)
            self._record_iteration(scheduler.take_stats(), seconds)
            outputs += self._deliver()
            self._kv.check_integrity()
            return outputs
        scheduler.update(answer)
        self._record_iteration(scheduler.take_stats(), seconds)
        outputs += self._deliver()
        if answer.error is not None:
            self._kv.check_integrity()
        return outputs

    def _record_iteration(self, counts: SchedulerStats | None, seconds: float) -> None:
        if self._stats is None or counts is None:
            return
        host_stored = offloads_delta = promotions_delta = 0
        host_store = self._kv.host_store
        if host_store is not None:
            hs = host_store.stats()
            host_stored = hs.stored_blocks
            offloads_delta = max(0, hs.offloads - self._last_host_offloads)
            promotions_delta = max(0, hs.promotions - self._last_host_promotions)
            self._last_host_offloads = hs.offloads
            self._last_host_promotions = hs.promotions
        self._stats.record(
            _iteration(
                counts,
                seconds,
                host_stored=host_stored,
                host_offloads=offloads_delta,
                host_promotions=promotions_delta,
            )
        )
        self._stats.maybe_log()

    def _deliver(self) -> list[RequestOutput]:
        """Turn the scheduler's events into outputs, until the processors stop reacting.

        A stop string aborts its request, which queues one more event; looping lets that event be
        taken and ignored in the same cycle instead of surfacing a cycle later.
        """
        outputs: list[RequestOutput] = []
        penalties = self._runner.penalties
        constraints = self._runner.constraints
        while events := self._scheduler.take_events():
            if len(penalties) or len(constraints):
                for event in events:
                    if event.finished:
                        penalties.drop(event.request_id)
                        constraints.drop(event.request_id)
                    elif event.token_ids:
                        penalties.commit(event.request_id, event.token_ids)
                        constraints.accept(event.request_id, event.token_ids)
            outputs += self._output.process(events)
        if self._families:
            for output in outputs:
                if output.finished:
                    self._families.pop(output.request_id, None)
        if self._recorder is not None:
            for output in outputs:
                self._recorder.observed(output)
        return outputs

    # -- life cycle ------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop accepting work and close the tokenizer; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._recorder is not None:
            self._recorder.close()
        host_store = self._kv.host_store
        if host_store is not None:
            host_store.close()
        self._input.compiler.close()
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


def _iteration(
    counts: SchedulerStats,
    seconds: float,
    *,
    host_stored: int = 0,
    host_offloads: int = 0,
    host_promotions: int = 0,
) -> IterationStats:
    """The loggers' record of a step: the scheduler's counts and the runner's time."""
    return IterationStats(
        step_id=counts.step_id,
        num_running=counts.num_running,
        num_waiting=counts.num_waiting,
        num_preempted=counts.num_preempted,
        num_scheduled_tokens=counts.num_scheduled_tokens,
        num_new_tokens=counts.num_new_tokens,
        num_finished=counts.num_finished,
        num_aborted=counts.num_aborted,
        num_failed=counts.num_failed,
        num_settled=counts.num_settled,
        step_seconds=seconds,
        pressure_level=counts.pressure_level,
        num_oom_recoveries=counts.num_oom_recoveries,
        host_kv_stored_blocks=host_stored,
        host_kv_offloads=host_offloads,
        host_kv_promotions=host_promotions,
    )
