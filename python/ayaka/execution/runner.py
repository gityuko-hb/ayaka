"""Runs one step on the model: build the inputs, forward, sample, answer the scheduler.

:class:`ModelRunner` is the executor half of the engine loop::

    step = scheduler.schedule(runner.plan_lookahead())
    output = runner.execute(step)
    scheduler.update(output)

It owns the :class:`~ayaka.execution.inputs.InputBuilder`, the sampler and the cache tensors, and it
holds no request state: everything it needs is in the ``StepInput``.

A request may have several rows in a step. A prefill chunk has one per chunk token and samples only
the last row of the chunk that ends the prompt. A decode request has ``1 + k`` rows, the pending
token and ``k`` draft tokens, and samples all of them. Row ``j`` of the window draws the token
that follows its input, with the random number of its own position, so the token a row draws does
not depend on the window it sits in. A draft is accepted while it equals the token the row before
it drew; the request then emits the tokens up to and including the first draw that no draft
matched. This is exactly what plain decoding would have emitted, so a verified step may only be
faster, never different. No draft proposer exists yet, so the scheduler is never asked for
look-ahead rows and :meth:`ModelRunner.plan_lookahead` returns none; the shape of the contract is
what the tests keep.

With :meth:`ModelRunner.capture_graphs`, plain decode steps that fit a captured bucket replay a
CUDA graph (:class:`~ayaka.execution.graph.GraphRunner`) instead of running eagerly; every other
step runs as before. Both paths draw the same tokens.

A request that asks for masks, a logit bias, ``min_tokens``, a penalty or logprobs gets them
from :mod:`ayaka.sampling.logits` in its step; such a step runs eagerly, never in a graph. The
penalty counts are the one per-request state of sampling: the runner keeps them in
:attr:`ModelRunner.penalties`, and the engine adds, grows and drops them as requests are admitted,
emit tokens and end.

This module imports torch. Import it through ``ayaka.execution``, which loads it on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ayaka.core.errors import BackendError
from ayaka.execution.graph import GraphRunner
from ayaka.execution.inputs import InputBuilder
from ayaka.layers.context import ForwardContext
from ayaka.sampling.logits import LogitRow, build_controls, needs_controls
from ayaka.sampling.penalties import PenaltyStates
from ayaka.sampling.sampler import Sampler
from ayaka.scheduler.step import StepOutput
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from ayaka.core.outputs import TokenLogprob
    from ayaka.core.types import RequestId, TokenId
    from ayaka.execution.graph import CapturedBucket
    from ayaka.execution.inputs import ModelInputs
    from ayaka.kvcache.storage.pool import KVPool
    from ayaka.kvcache.table import BlockTable
    from ayaka.models.base import CausalLM
    from ayaka.ops.kvcache import KVBuffers
    from ayaka.scheduler.step import StepInput

__all__ = ["ModelRunner"]


class ModelRunner:
    """Executes :class:`~ayaka.scheduler.step.StepInput` steps on one model and one cache.

    Args:
        model: The model, with its weights loaded; its ``ops`` are the backend the step uses.
        pools: The cache tensors, one pool per KV group. Exactly one group runs today.
        table: The host block table the scheduler writes; the runner mirrors it on the device.
        max_rows: Most rows (tokens) one step can have: at least the scheduler's prefill budget
            and its running requests times one plus the look-ahead.
        device: The device of the model and the cache.
        eos_token_ids: The model's end-of-sequence ids, which ``min_tokens`` bans until enough
            tokens exist.

    Raises:
        EngineError: If the model has more than one KV group.
    """

    @staticmethod
    def required_rows(*, token_budget: int, max_running: int, lookahead: int = 0) -> int:
        """Rows the fixed input buffers must hold for the largest step.

        A prefill step carries at most ``token_budget`` rows; a decode step carries one
        row per running request plus, with a draft proposer, ``lookahead`` draft rows.

        Args:
            token_budget: Largest number of tokens a prefill step may carry.
            max_running: Most requests that run at once.
            lookahead: Draft rows per request the runner may ask for; ``0`` today.

        Raises:
            TypeError: If a value is not an ``int``.
            ValueError: If a value is below zero (one for the two limits).
        """
        return max(
            require_int(token_budget, "token_budget", minimum=1),
            require_int(max_running, "max_running", minimum=1)
            * (1 + require_int(lookahead, "lookahead")),
        )

    def __init__(
        self,
        model: CausalLM,
        pools: Sequence[KVPool],
        table: BlockTable,
        *,
        max_rows: int,
        device: torch.device | str,
        eos_token_ids: Collection[TokenId] = (),
    ) -> None:
        self._model = model
        self._sampler = Sampler(model.ops.sampling)
        self._device = torch.device(device)
        self._eos = frozenset(eos_token_ids)
        self._vocab = model.config.vocab_size
        self._penalties = PenaltyStates(self._vocab, self._device)
        self._buffers: dict[int, KVBuffers] = {}
        for pool in pools:
            for layer_id in pool.group.layer_ids:
                self._buffers[layer_id] = pool.buffers(layer_id)
        spec = pools[0].group.spec
        self._builder = InputBuilder(
            attention=model.ops.attention,
            table=table,
            block_size=spec.block_size,
            max_rows=max_rows,
            device=device,
        )
        self._graph: GraphRunner | None = None

    @property
    def builder(self) -> InputBuilder:
        """The input builder, whose fixed buffers tests inspect."""
        return self._builder

    @property
    def model(self) -> CausalLM:
        """The model this runner executes."""
        return self._model

    @property
    def kv_buffers(self) -> Mapping[int, KVBuffers]:
        """The cache tensors of each layer, by layer id."""
        return self._buffers

    @property
    def penalties(self) -> PenaltyStates:
        """The penalty counts of the requests that use a penalty."""
        return self._penalties

    @property
    def graph(self) -> GraphRunner | None:
        """The decode graphs, once :meth:`capture_graphs` has run."""
        return self._graph

    def capture_graphs(self, buckets: Sequence[int]) -> list[CapturedBucket]:
        """Capture decode steps of each bucket size; later decode steps that fit replay them.

        The model's ops must have been loaded with ``cuda_graph=True``.

        Raises:
            ConfigError: If the ops are not graph-safe, the model is not on a GPU, or no bucket
                fits the buffers.
        """
        graph = GraphRunner(self._model, self._builder, self._buffers, buckets=buckets)
        captured = graph.capture()
        self._graph = graph
        return captured

    def plan_lookahead(self) -> Mapping[RequestId, int]:
        """Ask for draft rows per request; none, because no draft proposer exists yet."""
        return {}

    def execute(
        self, step: StepInput, drafts: Sequence[Sequence[TokenId]] | None = None
    ) -> StepOutput:
        """Run one step and describe what it produced.

        A backend that refuses the step (a :class:`~ayaka.core.errors.BackendError`) makes the
        answer a step error, so the scheduler ends the step's requests with ``ERROR`` and the
        engine goes on; the cache of the step is not committed, so nothing it wrote is read.
        Any other exception propagates: the engine cannot know what state it left.

        Args:
            step: The step the scheduler handed out.
            drafts: Per request, the tokens of its draft rows. ``None`` lets the runner fill them.

        Returns:
            The step's answer; every request is releasable at once, because the call is synchronous.
        """
        try:
            with torch.inference_mode():
                if (
                    drafts is None
                    and self._graph is not None
                    and self._graph.bucket_for(step)
                    and not self._needs_controls(step)
                ):
                    drawn = self._graph.run(step)
                    return StepOutput(
                        step_id=step.step_id,
                        request_ids=step.request_ids,
                        tokens=tuple((token,) for token in drawn),
                        accepted=(0,) * step.num_requests,
                    )
                inputs = self._builder.build(step, drafts)
                sampled, logprobs = self._forward(step, inputs)
        except BackendError as exc:
            return StepOutput(
                step_id=step.step_id, request_ids=step.request_ids, error=exc.to_info()
            )
        return _answer(step, inputs, sampled, logprobs)

    def _forward(
        self, step: StepInput, inputs: ModelInputs
    ) -> tuple[list[TokenId], list[TokenLogprob | None]]:
        """Run the model over the rows and sample the logit rows.

        Returns:
            The drawn token ids, and per logit row its logprobs (empty if no row asked).
        """
        hidden = self._model(inputs.token_ids, self._context(inputs))
        count = int(inputs.logit_rows.shape[0])
        if count == 0:
            return [], []  # prefill chunks that all continue: the cache is written, none drawn
        logits = self._model.compute_logits(hidden.index_select(0, inputs.logit_rows))
        controls = build_controls(
            _logit_rows(step, inputs.logit_counts),
            vocab_size=int(logits.shape[-1]),
            eos_token_ids=self._eos,
            penalties=self._penalties,
            device=self._device,
        )
        logprobs = self._sampler.sample(logits, inputs.sampling, inputs.sampled, controls)
        return self._builder.read_sampled(count), logprobs

    def _needs_controls(self, step: StepInput) -> bool:
        """Whether a request of the step needs masks, bias, penalties or logprobs now."""
        has_eos = bool(self._eos)
        return any(
            needs_controls(params, step.output_length(index), has_eos=has_eos)
            for index, params in enumerate(step.sampling_params)
        )

    def _context(self, inputs: ModelInputs) -> ForwardContext:
        return ForwardContext(attention=inputs.attention, kv_buffers=self._buffers)


def _logit_rows(step: StepInput, logit_counts: Sequence[int]) -> list[LogitRow]:
    """One :class:`LogitRow` per logit row: row ``j`` of a request draws its token ``j``."""
    rows: list[LogitRow] = []
    for index, taken in enumerate(logit_counts):
        request_id = step.request_ids[index]
        params = step.sampling_params[index]
        produced = step.output_length(index)
        rows += [LogitRow(request_id, params, produced + j) for j in range(taken)]
    return rows


def _answer(
    step: StepInput,
    inputs: ModelInputs,
    sampled: list[TokenId],
    logprobs: Sequence[TokenLogprob | None] = (),
) -> StepOutput:
    """Turn the drawn tokens into per-request tokens, applying the draft acceptance rule."""
    tokens: list[tuple[TokenId, ...]] = []
    accepted: list[int] = []
    per_request: list[tuple[TokenLogprob, ...] | None] = []
    cursor = 0
    for index, taken in enumerate(inputs.logit_counts):
        drawn = sampled[cursor : cursor + taken]
        window = logprobs[cursor : cursor + taken] if logprobs else ()
        cursor += taken
        drafts = inputs.draft_tokens[index]
        count = 0
        while count < len(drafts) and drawn and drafts[count] == drawn[count]:
            count += 1
        # ``count`` drafts matched, so the draws up to and including the one after them stand.
        tokens.append(tuple(drawn[: count + 1]))
        accepted.append(count)
        kept = [entry for entry in window[: count + 1] if entry is not None]
        per_request.append(tuple(kept) if kept and len(kept) == len(drawn[: count + 1]) else None)
    return StepOutput(
        step_id=step.step_id,
        request_ids=step.request_ids,
        tokens=tuple(tokens),
        accepted=tuple(accepted),
        logprobs=tuple(per_request) if any(entry is not None for entry in per_request) else (),
    )
