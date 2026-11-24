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

A request that asks for allowed ids, a logit bias, ``min_tokens``, a penalty or logprobs gets them
from :mod:`ayaka.sampling.logits` in its step; such a step runs eagerly, never in a graph. The
structural mask of an active output constraint is the exception: the runner computes it on the host
before the replay and the graph applies it, so a constrained decode step replays like any other
(unless one of its rows also asks for something above). The penalty counts and the constraint
matchers are the per-request state of sampling: the runner keeps them in
:attr:`ModelRunner.penalties` and :attr:`ModelRunner.constraints`, and the engine adds, feeds and
drops them as requests are admitted, emit tokens and end.

This module imports torch. Import it through ``ayaka.execution``, which loads it on first use.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Final, cast

import torch

from ayaka.core.errors import BackendError, EngineError
from ayaka.core.types import ForwardMode
from ayaka.execution.answers import answer_step, answer_verify, logit_rows
from ayaka.execution.drafts import (
    observe_decode,
    observe_step,
    prepare_step_drafts,
)
from ayaka.execution.graph import GraphRunner
from ayaka.execution.inputs import InputBuilder
from ayaka.execution.speculation import SpeculationState
from ayaka.kvcache.storage.pool import KVPool, LatentKVPool
from ayaka.layers.context import ForwardContext, StateCheckpointCapture
from ayaka.sampling.logits import build_controls, needs_controls
from ayaka.sampling.penalties import PenaltyStates
from ayaka.sampling.sampler import Sampler
from ayaka.sampling.structured.runtime import ConstraintRuntime
from ayaka.scheduler.step import StepOutput
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ayaka.configs.speculative import SpeculativeConfig
    from ayaka.core.multimodal import ImageInput
    from ayaka.core.outputs import TokenLogprob
    from ayaka.core.params import SamplingParams
    from ayaka.core.types import RequestId, TokenId
    from ayaka.execution.graph import CapturedBucket
    from ayaka.execution.staged import ModelInputs
    from ayaka.kvcache.state_slots import StateCheckpointStore
    from ayaka.kvcache.storage.pool import RecurrentStatePool
    from ayaka.kvcache.table import BlockTable
    from ayaka.models.base import CausalLM
    from ayaka.ops.kvcache import KVBuffers, LatentKVBuffers
    from ayaka.ops.linear_attention import RecurrentStateBuffers
    from ayaka.scheduler.step import StepInput
    from ayaka.speculative.policy import DraftController
    from ayaka.speculative.proposer import Proposer

__all__ = ["ModelRunner"]

_MAX_VISION_CACHE: Final = 16

_LOG = logging.getLogger(__name__)


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
        pools: Sequence[KVPool | LatentKVPool],
        table: BlockTable,
        *,
        max_rows: int,
        device: torch.device | str,
        eos_token_ids: Collection[TokenId] = (),
        state_pool: RecurrentStatePool | None = None,
        state_checkpoints: StateCheckpointStore | None = None,
        speculative: SpeculativeConfig | None = None,
        proposer: Proposer | None = None,
        controller: DraftController | None = None,
    ) -> None:
        self._model = model
        self._sampler = Sampler(model.ops.sampling)
        self._device = torch.device(device)
        self._eos = frozenset(eos_token_ids)
        self._vocab = model.config.vocab_size
        self._penalties = PenaltyStates(self._vocab, self._device)
        self._constraints = ConstraintRuntime()
        self._buffers: dict[int, KVBuffers] = {}
        self._latent_buffers: dict[int, LatentKVBuffers] = {}
        self._group_layers: tuple[tuple[int, ...], ...] = tuple(
            pool.group.layer_ids for pool in pools
        )
        for pool in pools:
            if isinstance(pool, LatentKVPool):
                for layer_id in pool.group.layer_ids:
                    self._latent_buffers[layer_id] = pool.buffers(layer_id)
            else:
                for layer_id in pool.group.layer_ids:
                    self._buffers[layer_id] = pool.buffers(layer_id)
        self._state_pool = state_pool
        self._state_checkpoints = state_checkpoints
        self._state_buffers: dict[int, RecurrentStateBuffers] = {}
        if state_pool is not None:
            for layer_id in state_pool.group.layer_ids:
                self._state_buffers[layer_id] = state_pool.buffers(layer_id)
        spec = pools[0].group.spec
        self._builder = InputBuilder(
            attention=model.ops.attention,
            table=table,
            block_size=spec.block_size,
            max_rows=max_rows,
            device=device,
            has_linear_attention=state_pool is not None,
        )
        self._graph: GraphRunner | None = None
        self._vision_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._spec = SpeculationState(
            self._constraints, speculative=speculative, proposer=proposer, controller=controller
        )
        self._extra_graph_bytes = 0

    @property
    def extra_graph_bytes(self) -> int:
        """CUDA graph bytes captured by auxiliary proposers (a proposer's own draft graphs)."""
        return self._extra_graph_bytes

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
    def latent_kv_buffers(self) -> Mapping[int, LatentKVBuffers]:
        """Compressed MLA latent KV cache tensors by layer id."""
        return self._latent_buffers

    @property
    def state_buffers(self) -> Mapping[int, RecurrentStateBuffers]:
        """Recurrent linear-attention state tensors by layer id."""
        return self._state_buffers

    @property
    def group_layers(self) -> tuple[tuple[int, ...], ...]:
        """Layer ids assigned to each KV or latent KV group."""
        return self._group_layers

    @property
    def penalties(self) -> PenaltyStates:
        """The penalty counts of the requests that use a penalty."""
        return self._penalties

    @property
    def constraints(self) -> ConstraintRuntime:
        """The running output constraints of the requests that have one."""
        return self._constraints

    @property
    def graph(self) -> GraphRunner | None:
        """The decode graphs, once :meth:`capture_graphs` has run."""
        return self._graph

    def capture_graphs(
        self, buckets: Sequence[int], *, verify_buckets: Sequence[int] = ()
    ) -> list[CapturedBucket]:
        """Capture decode steps (and verify row buckets when given) for graph replay.

        The model's ops must have been loaded with ``cuda_graph=True``.

        Raises:
            ConfigError: If the ops are not graph-safe, the model is not on a GPU, or no bucket
                fits the buffers.
        """
        graph = GraphRunner(
            self._model,
            self._builder,
            self._buffers,
            buckets=buckets,
            latent_kv_buffers=self._latent_buffers,
            state_buffers=self._state_buffers,
            group_layers=self._group_layers,
            device=self._device,
            verify_buckets=verify_buckets,
            save_hidden=self._spec.needs_hidden,
            feature_layer_ids=frozenset(self._spec.feature_order),
        )
        captured = graph.capture()
        self._graph = graph
        self._extra_graph_bytes = 0
        controller = self._spec.controller
        if verify_buckets and controller is not None:
            if graph.max_verify_rows > 0:
                controller.max_allowed_rows = graph.max_verify_rows
            else:
                _LOG.warning(
                    "no verify CUDA graph bucket could be captured; verify steps will run eagerly"
                )
        return captured

    def release(self) -> None:
        """Drop captured graphs, cache tensor views, and state pools."""
        if self._graph is not None:
            self._graph.release()
            self._graph = None
        self._buffers.clear()
        self._latent_buffers.clear()
        self._state_buffers.clear()
        self._state_pool = None
        self._state_checkpoints = None
        self._vision_cache.clear()
        self._model.release_vision()

    @staticmethod
    def clear_device_cache() -> None:
        """Reclaim unreferenced host/device memory after dropping an engine runner."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def proposer(self) -> Proposer | None:
        """The active speculative proposer, if any."""
        return self._spec.proposer

    @property
    def controller(self) -> DraftController | None:
        """The speculative draft-length controller, if any."""
        return self._spec.controller

    def set_speculative(
        self,
        proposer: Proposer | None,
        controller: DraftController | None = None,
        speculative: SpeculativeConfig | None = None,
    ) -> None:
        """Install or replace the speculative proposer and controller (primarily for tests)."""
        self._spec.replace(proposer, controller, speculative)

    def admit_speculative(
        self,
        request_id: RequestId,
        prompt: Sequence[TokenId],
        *,
        sampling: SamplingParams,
        has_constraint: bool = False,
        prediction: Sequence[TokenId] = (),
    ) -> None:
        """Register ``request_id`` with the speculative proposer and record eligibility."""
        self._spec.admit(
            request_id,
            prompt,
            sampling=sampling,
            has_constraint=has_constraint,
            prediction=prediction,
        )

    def commit_speculative(self, request_id: RequestId, tokens: Sequence[TokenId]) -> None:
        """Feed committed tokens to the proposer; unverified drafts never enter."""
        self._spec.commit(request_id, tokens)

    def release_speculative(self, request_id: RequestId) -> None:
        """Drop per-request speculative state when a request finishes or is aborted."""
        self._spec.release(request_id)

    def release_all_speculative(self) -> None:
        """Release every active request's speculative state (e.g., on shutdown)."""
        self._spec.release_all()

    def plan_lookahead(self) -> Mapping[RequestId, int]:
        """Ask the speculative proposer and controller for draft rows per active request."""
        return self._spec.plan()

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
        started = time.perf_counter()
        resolved_drafts, used_proposals, draft_valid, disabled = prepare_step_drafts(
            step,
            drafts,
            plans=self._spec.planned,
            pending_disabled=self._spec.pending_disabled,
            vocab=self._vocab,
        )
        try:
            with torch.inference_mode():
                if (
                    drafts is None
                    and not (step.images and self._model.uses_multimodal_decode_offsets())
                    and step.mode is ForwardMode.DECODE
                    and self._graph is not None
                    and self._graph.bucket_for(step)
                    and not self._needs_controls(step)
                ):
                    drawn = self._graph.run(step, self._structural_masks(step))
                    elapsed = time.perf_counter() - started
                    observe_decode(self._spec.controller, step, elapsed)
                    output = StepOutput(
                        step_id=step.step_id,
                        request_ids=step.request_ids,
                        tokens=tuple((token,) for token in drawn),
                        accepted=(0,) * step.num_requests,
                        spec_disabled=disabled,
                    )
                    if self._spec.needs_hidden or self._spec.feature_order:
                        self._notify_proposer(
                            step, output, self._graph.hidden_view(step.num_rows), from_graph=True
                        )
                    return output
                if (
                    drafts is None
                    and step.mode is ForwardMode.VERIFY
                    and resolved_drafts is not None
                    and self._graph is not None
                    and self._graph.bucket_for(step)
                    and not self._needs_controls(step)
                    and not self._structural_masks(step)
                ):
                    staged, sampled = self._graph.run_verify(step, resolved_drafts)
                    elapsed = time.perf_counter() - started
                    output = answer_verify(
                        step,
                        staged.logit_counts,
                        staged.draft_tokens,
                        sampled,
                        draft_valid=draft_valid,
                        vocab_size=self._vocab,
                        used_proposals=used_proposals,
                        disabled=disabled,
                    )
                    self._notify_proposer(
                        step, output, self._graph.hidden_view(step.num_rows), from_graph=True
                    )
                    observe_step(self._spec.controller, step, output, used_proposals, elapsed)
                    return output
                inputs = self._builder.build(step, resolved_drafts)
                sampled, logprobs, step_hidden = self._forward(step, inputs)
        except BackendError as exc:
            if self._spec.feature_sink is not None:
                self._spec.feature_sink.clear()
            return StepOutput(
                step_id=step.step_id, request_ids=step.request_ids, error=exc.to_info()
            )
        elapsed = time.perf_counter() - started
        output = answer_step(
            step,
            inputs,
            sampled,
            logprobs,
            draft_valid=draft_valid,
            vocab_size=self._vocab,
            used_proposals=used_proposals,
            disabled=disabled,
        )
        self._notify_proposer(step, output, step_hidden, from_graph=False)
        observe_step(self._spec.controller, step, output, used_proposals, elapsed)
        return output

    def _prepare_prefill_states(self, step: StepInput) -> None:
        """Zero fresh state slots or restore checkpoints from prefix hits before prefill."""
        if self._state_pool is None or step.mode is not ForwardMode.PREFILL:
            return
        restores = step.state_restore_digests
        for idx, slot in enumerate(step.state_slots):
            if slot == 0:
                continue
            digest = restores[idx] if restores else None
            if digest is not None:
                if self._state_checkpoints is None:
                    raise EngineError("step requests a state restore with no checkpoint store")
                payload = self._state_checkpoints.get(digest)
                if payload is None:
                    raise EngineError(
                        f"missing recurrent state checkpoint for prefix digest {digest.hex()}"
                    )
                ssm, conv = cast("tuple[torch.Tensor, torch.Tensor]", payload)
                self._state_pool.restore_slot(slot, (ssm, conv))
            elif step.start_positions[idx] == 0:
                self._state_pool.zero_slot(slot)

    def _save_prefill_checkpoints(
        self, step: StepInput, capture: StateCheckpointCapture | None
    ) -> None:
        """Persist captured per-layer recurrent states into the bounded checkpoint store."""
        if (
            capture is None
            or self._state_pool is None
            or self._state_checkpoints is None
            or not capture.snapshots
        ):
            return
        layer_ids = self._state_pool.group.layer_ids
        seen: set[bytes] = set()
        for targets in step.state_checkpoint_targets:
            for _, digest in targets:
                if digest in seen:
                    continue
                seen.add(digest)
                if not all((layer_id, digest) in capture.snapshots for layer_id in layer_ids):
                    continue
                ssm_stack = torch.stack(
                    [capture.snapshots[(layer_id, digest)][0] for layer_id in layer_ids], dim=0
                )
                conv_stack = torch.stack(
                    [capture.snapshots[(layer_id, digest)][1] for layer_id in layer_ids], dim=0
                )
                nbytes = int(
                    ssm_stack.numel() * ssm_stack.element_size()
                    + conv_stack.numel() * conv_stack.element_size()
                )
                self._state_checkpoints.put(digest, (ssm_stack, conv_stack), nbytes)

    def _forward(
        self, step: StepInput, inputs: ModelInputs
    ) -> tuple[list[TokenId], list[TokenLogprob | None], torch.Tensor]:
        """Run the model over the rows and sample the logit rows.

        Returns:
            The drawn token ids, per logit row its logprobs, and final post-norm hidden states.
        """
        self._prepare_prefill_states(step)
        capture = (
            StateCheckpointCapture(boundaries=step.state_checkpoint_targets, snapshots={})
            if (
                step.mode is ForwardMode.PREFILL
                and self._state_pool is not None
                and self._state_checkpoints is not None
                and step.state_checkpoint_targets
                and any(step.state_checkpoint_targets)
            )
            else None
        )
        embeds, mrope_pos, rope_offsets, bidir_spans = self._prepare_multimodal(step, inputs)
        hidden = self._model(
            inputs.token_ids,
            self._context(
                inputs,
                capture,
                input_embeds=embeds,
                mrope_positions=mrope_pos,
                rope_position_offsets=rope_offsets,
                bidirectional_spans=bidir_spans,
            ),
        )
        self._save_prefill_checkpoints(step, capture)
        count = int(inputs.logit_rows.shape[0])
        if count == 0:
            return [], [], hidden
        logits = self._model.compute_logits(hidden.index_select(0, inputs.logit_rows))
        draft_by_req = {
            step.request_ids[index]: inputs.draft_tokens[index]
            for index in range(step.num_requests)
            if inputs.draft_tokens[index]
        }
        controls = build_controls(
            logit_rows(step, inputs.logit_counts),
            vocab_size=int(logits.shape[-1]),
            eos_token_ids=self._eos,
            penalties=self._penalties,
            device=self._device,
            constraints=self._constraints,
            draft_tokens=draft_by_req,
        )
        logprobs = self._sampler.sample(logits, inputs.sampling, inputs.sampled, controls)
        return self._builder.read_sampled(count), logprobs, hidden

    def _notify_proposer(
        self,
        step: StepInput,
        output: StepOutput,
        hidden: torch.Tensor | None,
        *,
        from_graph: bool,
    ) -> None:
        self._spec.notify(step, output, hidden, self._graph if from_graph else None)

    def _needs_controls(self, step: StepInput) -> bool:
        """Whether a request of the step needs more than a structural mask: allowed ids, bias,
        penalties, logprobs or a ``min_tokens`` hold-back, now."""
        has_eos = bool(self._eos)
        return any(
            needs_controls(params, step.output_length(index), has_eos=has_eos)
            for index, params in enumerate(step.sampling_params)
        )

    def _structural_masks(self, step: StepInput) -> dict[int, NDArray[np.uint32]]:
        """The allowed tokens of the requests of a decode step whose constraint is active.

        By index in the step, which is also the row: a decode step has one row per request.
        """
        constraints = self._constraints
        masks: dict[int, NDArray[np.uint32]] = {}
        if not len(constraints):
            return masks
        for index, request_id in enumerate(step.request_ids):
            mask = constraints.mask(request_id)
            if mask is not None:
                masks[index] = mask
        return masks

    def _prepare_multimodal(
        self, step: StepInput, inputs: ModelInputs
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[tuple[int, int], ...],
    ]:
        if not step.images:
            return None, None, None, ()
        mrope_pos, rope_offsets = self._model.multimodal_position_ids(
            step.prompt_lengths,
            step.image_spans,
            step.start_positions,
            step.rows_per_request,
            is_prefill=(step.mode is ForwardMode.PREFILL),
            device=self._device,
        )
        if step.mode is not ForwardMode.PREFILL:
            return None, mrope_pos, rope_offsets, ()

        missing: list[tuple[str, ImageInput]] = []
        seen: set[str] = set()
        for req_idx, (start, rows) in enumerate(
            zip(step.start_positions, step.rows_per_request, strict=True)
        ):
            for span in step.image_spans[req_idx]:
                if min(start + rows, span.token_offset + span.token_count) <= max(
                    start, span.token_offset
                ):
                    continue
                img = step.images[req_idx][span.image_index]
                digest = hashlib.sha256(img.data).hexdigest()
                if digest not in self._vision_cache and digest not in seen:
                    seen.add(digest)
                    missing.append((digest, img))

        if missing:
            encoded = self._model.encode_images([item[1] for item in missing])
            for (digest, _), feat in zip(missing, encoded, strict=True):
                self._vision_cache[digest] = feat
                self._vision_cache.move_to_end(digest)
                while len(self._vision_cache) > _MAX_VISION_CACHE:
                    self._vision_cache.popitem(last=False)

        embeds = self._model.embed_input_ids(inputs.token_ids).clone()
        bidir_spans: list[tuple[int, int]] = []
        use_bidir = self._model.uses_bidirectional_vision()
        batch_offset = 0
        for req_idx, (start, rows) in enumerate(
            zip(step.start_positions, step.rows_per_request, strict=True)
        ):
            for span in step.image_spans[req_idx]:
                lo = max(start, span.token_offset)
                hi = min(start + rows, span.token_offset + span.token_count)
                if hi <= lo:
                    continue
                img = step.images[req_idx][span.image_index]
                digest = hashlib.sha256(img.data).hexdigest()
                feat = self._vision_cache[digest]
                self._vision_cache.move_to_end(digest)
                if feat.shape[0] != span.token_count:
                    raise EngineError(
                        f"vision feature length {feat.shape[0]} != span {span.token_count}"
                    )
                row_lo = batch_offset + (lo - start)
                row_hi = batch_offset + (hi - start)
                feat_lo = lo - span.token_offset
                feat_hi = hi - span.token_offset
                embeds[row_lo:row_hi].copy_(feat[feat_lo:feat_hi].to(embeds.device, embeds.dtype))
                if use_bidir:
                    bidir_spans.append((row_lo, row_hi))
            batch_offset += rows
        return embeds, mrope_pos, rope_offsets, tuple(bidir_spans)

    def _context(
        self,
        inputs: ModelInputs,
        capture: StateCheckpointCapture | None = None,
        *,
        input_embeds: torch.Tensor | None = None,
        mrope_positions: torch.Tensor | None = None,
        rope_position_offsets: torch.Tensor | None = None,
        bidirectional_spans: tuple[tuple[int, int], ...] = (),
    ) -> ForwardContext:
        layer_attention = {
            layer_id: meta
            for meta, layer_ids in zip(inputs.group_attentions, self._group_layers, strict=False)
            for layer_id in layer_ids
        }
        return ForwardContext(
            attention=inputs.attention,
            kv_buffers=self._buffers,
            layer_attention=layer_attention,
            latent_kv_buffers=self._latent_buffers,
            state_buffers=self._state_buffers,
            linear_attention=inputs.linear_attention,
            checkpoint_capture=capture,
            input_embeds=input_embeds,
            mrope_positions=mrope_positions,
            rope_position_offsets=rope_position_offsets,
            bidirectional_spans=bidirectional_spans,
            feature_sink=self._spec.feature_sink,
        )
