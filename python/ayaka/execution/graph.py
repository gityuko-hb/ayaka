"""Decode steps through CUDA graphs: capture once per bucket, replay every step.

At small batch sizes most of a decode step is the time Python spends launching
kernels. :class:`GraphRunner` captures the whole step (the model's forward, the
logits and the sampler) once per *bucket* of requests and replays it, so a step
costs one launch.

::

    capture(bucket)                               at start-up, largest bucket first
        fixed metadata from ops.attention.graph_metadata, every request a padding request
        two eager runs (lazy kernel set-up and first allocations happen outside the graph)
        with the garbage collector paused: capture forward + logits + sampler, in one memory pool
        four variants per bucket: greedy (argmax only) or sampling (uniform + sample), each
        with or without a token mask applied to the logits before the draw

    run(step, masks)                              a decode step of at most the largest bucket
        InputBuilder.stage_decode: tokens and sampling parameters into the fixed buffers
        InputBuilder.stage_mask: the token masks of the constrained rows, one copy, when any
        ops.attention.update_graph_metadata: lengths, slots and block tables, outside the graph
        replay the smallest bucket that holds the step, in the variant the step needs
        read the sampled tokens back: the step's one device-to-host synchronisation

Only plain decode steps replay: one row per request, no draft rows. Prefill and
verify steps run eagerly (``ModelRunner`` decides), and so does a step with a row that
needs more than a token mask: bias, penalties, logprobs, allowed ids, ``min_tokens``. The
mask is the structural mask of an output constraint, computed on the host before the replay
(it depends on the token the last step drew); the graph only applies it. Every op family the
graph calls must come from a backend that declares it graph-safe; ``load_backend`` checks that
when it is built with ``cuda_graph=True``.

This module imports torch. Import it through ``ayaka.execution``, which loads it on first use.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from ayaka.core.errors import ConfigError, EngineError
from ayaka.core.types import ForwardMode
from ayaka.layers.context import ForwardContext
from ayaka.sampling.logits import LogitControls
from ayaka.sampling.sampler import Sampler
from ayaka.utils.gc_control import paused_gc

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ayaka.core.types import TokenId
    from ayaka.execution.inputs import InputBuilder, StagedDecode
    from ayaka.models.base import CausalLM
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import KVBuffers
    from ayaka.scheduler.step import StepInput

__all__ = ["CapturedBucket", "GraphRunner"]

logger = logging.getLogger(__name__)

#: Eager runs before a capture: the first may set kernels up or allocate, the second must not.
_WARMUP_RUNS = 2


@dataclass(frozen=True, slots=True)
class CapturedBucket:
    """What capturing one bucket cost.

    Attributes:
        bucket: Requests the graphs run.
        bytes: Memory the allocator reserved for the bucket's graphs and inputs (every variant).
            It is the allocator's growth, what the memory ledger is checked against, not the drop
            of the driver's free memory, which under WSL2 can miss an allocation for a while.
        seconds: Wall time of the warm-up runs and every capture.
    """

    bucket: int
    bytes: int
    seconds: float


class GraphRunner:
    """Captures and replays decode steps of one model on one cache.

    Args:
        model: The model; its ``ops`` must have been loaded with ``cuda_graph=True``.
        builder: The runner's input builder, whose fixed buffers the graphs read and write.
        kv_buffers: The cache tensors of each layer, by layer id.
        buckets: Requests per graph, ascending; each at most the builder's capacity.

    Raises:
        ConfigError: ``GRAPH_NEEDS_GRAPH_SAFE_OPS`` if the ops were not loaded for graphs,
            ``GRAPH_NEEDS_CUDA`` if the model is not on a GPU, ``GRAPH_BUCKETS_INVALID`` if no
            bucket fits the builder.
    """

    def __init__(
        self,
        model: CausalLM,
        builder: InputBuilder,
        kv_buffers: Mapping[int, KVBuffers],
        *,
        buckets: Sequence[int],
    ) -> None:
        if not model.ops.graph_safe:
            raise ConfigError(
                "device.cuda_graph",
                "GRAPH_NEEDS_GRAPH_SAFE_OPS",
                "the ops backend was not loaded with cuda_graph=True",
            )
        device = next(iter(kv_buffers.values())).k.device
        if device.type != "cuda":
            raise ConfigError(
                "device.cuda_graph", "GRAPH_NEEDS_CUDA", f"CUDA graphs need a GPU, got {device}"
            )
        capacity = min(builder.max_rows, builder.max_requests)
        usable = tuple(sorted({bucket for bucket in buckets if 1 <= bucket <= capacity}))
        if not usable:
            raise ConfigError(
                "device.graph_buckets",
                "GRAPH_BUCKETS_INVALID",
                f"no bucket of {list(buckets)} fits the {capacity} requests a step can hold",
            )
        self._model = model
        self._builder = builder
        self._kv_buffers = dict(kv_buffers)
        self._device = device
        self._buckets = usable
        self._sampler = Sampler(model.ops.sampling)
        self._words = (model.config.vocab_size + 31) // 32  # words of a token mask
        self._meta: dict[int, AttentionMetadata] = {}
        # (bucket, greedy, masked)
        self._graphs: dict[tuple[int, bool, bool], torch.cuda.CUDAGraph] = {}
        self._pool: Any = None
        self.replays = 0

    @property
    def buckets(self) -> tuple[int, ...]:
        """The bucket sizes this runner captures, ascending."""
        return self._buckets

    @property
    def captured(self) -> bool:
        """Whether :meth:`capture` has run."""
        return bool(self._graphs)

    def bucket_for(self, step: StepInput) -> int | None:
        """The smallest bucket that can replay ``step``, or ``None`` if it must run eagerly."""
        if not self._graphs or step.mode is not ForwardMode.DECODE:
            return None
        if step.num_rows != step.num_requests:  # draft rows
            return None
        for bucket in self._buckets:
            if bucket >= step.num_requests:
                return bucket
        return None

    def capture(self) -> list[CapturedBucket]:
        """Capture every bucket, largest first, in one shared memory pool.

        Returns:
            What each bucket cost, in ascending bucket order.
        """
        attention = self._model.ops.attention
        builder = self._builder
        results: list[CapturedBucket] = []
        self._pool = torch.cuda.graph_pool_handle()
        for bucket in reversed(self._buckets):
            started = time.monotonic()
            torch.cuda.synchronize(self._device)
            before = torch.cuda.memory_reserved(self._device)
            meta = attention.graph_metadata(
                bucket, builder.max_blocks, builder.block_size, self._device
            )
            self._meta[bucket] = meta
            for greedy in (True, False):
                for masked in (False, True):
                    self._graphs[bucket, greedy, masked] = self._capture_one(
                        bucket, greedy, masked, meta
                    )
            torch.cuda.synchronize(self._device)
            after = torch.cuda.memory_reserved(self._device)
            results.append(
                CapturedBucket(bucket, max(after - before, 0), time.monotonic() - started)
            )
        results.reverse()
        logger.info(
            "captured CUDA graphs for decode buckets %s: %d bytes in %.1fs",
            list(self._buckets),
            sum(item.bytes for item in results),
            sum(item.seconds for item in results),
        )
        return results

    def _capture_one(self, bucket: int, greedy: bool, masked: bool, meta: AttentionMetadata) -> Any:
        tokens, batch, sampled = self._builder.graph_views(bucket)
        batch = replace(batch, all_greedy=greedy)
        context = ForwardContext(attention=meta, kv_buffers=self._kv_buffers)
        # The same controls an eager step with only a structural mask builds, so the two draw
        # from the same float32 logits.
        controls = (
            LogitControls(
                bitmask=self._builder.graph_mask(bucket, self._words),
                bias=None,
                penalties=None,
                logprobs=(None,) * bucket,
            )
            if masked
            else None
        )

        def body() -> None:
            hidden = self._model(tokens, context)
            logits = self._model.compute_logits(hidden)
            self._sampler.sample(logits, batch, sampled, controls)

        self._reset_inputs(bucket)
        side = torch.cuda.Stream(self._device)
        side.wait_stream(torch.cuda.current_stream(self._device))
        with torch.inference_mode(), torch.cuda.stream(side):
            for _ in range(_WARMUP_RUNS):
                body()
        torch.cuda.current_stream(self._device).wait_stream(side)
        torch.cuda.synchronize(self._device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), paused_gc(), torch.cuda.graph(graph, pool=self._pool):
            body()
        return graph

    def _reset_inputs(self, bucket: int) -> None:
        """Fill a bucket's inputs with a valid all-padding step: what the capture runs on."""
        builder = self._builder
        lengths = torch.zeros(bucket, dtype=torch.int32)
        rows = torch.zeros((bucket, builder.max_blocks), dtype=torch.int32)
        self._model.ops.attention.update_graph_metadata(self._meta[bucket], lengths, rows)
        builder.reset_graph_inputs(bucket)

    def launch(
        self, step: StepInput, masks: Mapping[int, NDArray[np.uint32]] | None = None
    ) -> StagedDecode:
        """Write ``step`` into the graph's inputs and queue its replay; wait for nothing.

        Everything here is host work and asynchronous copies: the device is never read, so the
        step's only synchronisation is :meth:`run`'s read of the sampled tokens.

        Args:
            step: A decode step.
            masks: The allowed tokens, as ``uint32`` words, of the requests that have a
                constraint, by index in the step. Any entry replays a graph that applies the
                masks; every request without one allows every token.

        Raises:
            EngineError: If no captured bucket can replay the step (check :meth:`bucket_for`).
            BackendError: If the backend refuses the step's metadata.
        """
        bucket = self.bucket_for(step)
        if bucket is None:
            raise EngineError(f"step {step.step_id} cannot be replayed by a decode graph")
        staged = self._builder.stage_decode(step, bucket)
        masked = bool(masks)
        if masks:
            self._builder.stage_mask(masks, bucket)
        self._model.ops.attention.update_graph_metadata(
            self._meta[bucket], staged.kv_lengths, staged.block_rows
        )
        self._graphs[bucket, staged.all_greedy, masked].replay()
        self.replays += 1
        return staged

    def run(
        self, step: StepInput, masks: Mapping[int, NDArray[np.uint32]] | None = None
    ) -> list[TokenId]:
        """Replay ``step`` and return the token each of its requests drew, in step order.

        Raises:
            EngineError: If no captured bucket can replay the step (check :meth:`bucket_for`).
            BackendError: If the backend refuses the step's metadata.
        """
        staged = self.launch(step, masks)
        return self._builder.read_sampled(staged.count)
