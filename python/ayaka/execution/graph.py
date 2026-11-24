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
from ayaka.layers.context import FeatureSink, ForwardContext
from ayaka.sampling.logits import LogitControls
from ayaka.sampling.sampler import Sampler
from ayaka.utils.gc_control import paused_gc

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ayaka.core.types import TokenId
    from ayaka.execution.inputs import InputBuilder
    from ayaka.execution.staged import StagedDecode, StagedVerify
    from ayaka.models.base import CausalLM
    from ayaka.ops.attention import AttentionMetadata
    from ayaka.ops.kvcache import KVBuffers, LatentKVBuffers
    from ayaka.ops.linear_attention import RecurrentStateBuffers
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
        latent_kv_buffers: Mapping[int, LatentKVBuffers] | None = None,
        state_buffers: Mapping[int, RecurrentStateBuffers] | None = None,
        group_layers: Sequence[tuple[int, ...]] | None = None,
        device: torch.device | str | None = None,
        verify_buckets: Sequence[int] = (),
        save_hidden: bool = False,
        feature_layer_ids: frozenset[int] = frozenset(),
    ) -> None:
        if not model.ops.graph_safe:
            raise ConfigError(
                "device.cuda_graph",
                "GRAPH_NEEDS_GRAPH_SAFE_OPS",
                "the ops backend was not loaded with cuda_graph=True",
            )
        latent_map = dict(latent_kv_buffers or {})
        state_map = dict(state_buffers or {})
        if device is not None:
            resolved = torch.device(device)
        elif kv_buffers:
            resolved = next(iter(kv_buffers.values())).k.device
        elif latent_map:
            resolved = next(iter(latent_map.values())).ckv.device
        elif state_map:
            resolved = next(iter(state_map.values())).ssm_state.device
        else:
            resolved = next(model.parameters()).device
        if resolved.type != "cuda":
            raise ConfigError(
                "device.cuda_graph", "GRAPH_NEEDS_CUDA", f"CUDA graphs need a GPU, got {resolved}"
            )
        capacity = min(builder.max_rows, builder.max_requests)
        usable = tuple(sorted({bucket for bucket in buckets if 1 <= bucket <= capacity}))
        if not usable:
            raise ConfigError(
                "device.graph_buckets",
                "GRAPH_BUCKETS_INVALID",
                f"no bucket of {list(buckets)} fits the {capacity} requests a step can hold",
            )
        # A verify window is captured for one KV group of paged KV only: the staged verify step
        # carries one block table, and a recurrent state or a latent cache has no verify kernel.
        # Such models still verify draft windows eagerly (or refuse speculation, see
        # ``speculative.proposer``).
        verify_capturable = builder.num_groups == 1 and not latent_map and not state_map
        verify_usable = tuple(
            sorted({bucket for bucket in verify_buckets if 2 <= bucket <= builder.max_rows})
            if verify_capturable
            else ()
        )
        self._model = model
        self._builder = builder
        self._kv_buffers = dict(kv_buffers)
        self._latent_kv_buffers = latent_map
        self._state_buffers = state_map
        self._group_layers = tuple(group_layers) if group_layers is not None else ()
        self._device = resolved
        self._buckets = usable
        self._verify_buckets = verify_usable
        self._sampler = Sampler(model.ops.sampling)
        self._words = (model.config.vocab_size + 31) // 32  # words of a token mask
        self._meta: dict[int, tuple[AttentionMetadata, ...]] = {}
        self._verify_meta: dict[int, AttentionMetadata] = {}
        # (bucket, greedy, masked)
        self._graphs: dict[tuple[int, bool, bool], torch.cuda.CUDAGraph] = {}
        # (bucket, greedy) for VERIFY steps
        self._verify_graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}
        self._pool: Any = None
        max_cap = max([*usable, *verify_usable])
        hidden_size = model.config.hidden_size
        model_dtype = model.compute_dtype.torch_dtype
        self._hidden_buf: torch.Tensor | None = (
            torch.empty((max_cap, hidden_size), dtype=model_dtype, device=resolved)
            if save_hidden
            else None
        )
        self._feature_buffers: dict[int, torch.Tensor] = {
            lid: torch.empty((max_cap, hidden_size), dtype=model_dtype, device=resolved)
            for lid in feature_layer_ids
        }
        self._feature_sink: FeatureSink | None = (
            FeatureSink(layer_ids=feature_layer_ids, buffers=self._feature_buffers)
            if feature_layer_ids
            else None
        )
        self.replays = 0
        self.verify_replays = 0

    def hidden_view(self, num_rows: int) -> torch.Tensor | None:
        """View ``[num_rows, hidden_size]`` captured by the last decode or verify graph replay."""
        return None if self._hidden_buf is None else self._hidden_buf[:num_rows]

    def features_view(self, layer_ids: Sequence[int], num_rows: int) -> torch.Tensor | None:
        """Concatenated ``[num_rows, len(layer_ids) * hidden_size]`` features from last replay."""
        if not self._feature_buffers or not layer_ids:
            return None
        return torch.cat([self._feature_buffers[lid][:num_rows] for lid in layer_ids], dim=-1)

    @property
    def buckets(self) -> tuple[int, ...]:
        """The bucket sizes this runner captures, ascending."""
        return self._buckets

    @property
    def pool(self) -> Any:
        """Shared CUDA graph memory pool handle once :meth:`capture` has run."""
        return self._pool

    @property
    def verify_buckets(self) -> tuple[int, ...]:
        """The total-row bucket sizes captured for ``VERIFY`` steps, ascending."""
        return self._verify_buckets

    @property
    def max_verify_rows(self) -> int:
        """Largest verify step row count covered by captured graphs, or 0 if none."""
        return self._verify_buckets[-1] if self._verify_buckets else 0

    @property
    def captured(self) -> bool:
        """Whether :meth:`capture` has run."""
        return bool(self._graphs)

    def release(self) -> None:
        """Drop captured CUDA graphs, metadata, and buffer views to free device memory."""
        self._graphs.clear()
        self._meta.clear()
        self._verify_graphs.clear()
        self._verify_meta.clear()
        self._kv_buffers.clear()
        self._latent_kv_buffers.clear()
        self._state_buffers.clear()
        self._pool = None

    def bucket_for(self, step: StepInput) -> int | None:
        """The smallest bucket that can replay ``step``, or ``None`` if it must run eagerly."""
        if step.mode is ForwardMode.DECODE:
            if not self._graphs or step.num_rows != step.num_requests:
                return None
            for bucket in self._buckets:
                if bucket >= step.num_requests:
                    return bucket
            return None
        if step.mode is ForwardMode.VERIFY:
            if not self._verify_graphs:
                return None
            # When all max_requests slots are active and padding rows remain, we have no spare
            # padding request row in the block table; only exact-row match can replay then.
            for bucket in self._verify_buckets:
                if bucket >= step.num_rows and (
                    step.num_requests < self._builder.max_requests or bucket == step.num_rows
                ):
                    return bucket
            return None
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
            metas = tuple(
                attention.graph_metadata(
                    bucket, builder.max_blocks, builder.block_size, self._device
                )
                for _ in range(builder.num_groups)
            )
            self._meta[bucket] = metas
            for greedy in (True, False):
                for masked in (False, True):
                    self._graphs[bucket, greedy, masked] = self._capture_one(
                        bucket, greedy, masked, metas
                    )
            torch.cuda.synchronize(self._device)
            after = torch.cuda.memory_reserved(self._device)
            results.append(
                CapturedBucket(bucket, max(after - before, 0), time.monotonic() - started)
            )
        results.reverse()
        kept_verify: list[int] = []
        for vbucket in reversed(self._verify_buckets):
            started = time.monotonic()
            torch.cuda.synchronize(self._device)
            before = torch.cuda.memory_reserved(self._device)
            try:
                vmeta = attention.verify_graph_metadata(
                    vbucket,
                    builder.max_requests,
                    builder.max_blocks,
                    builder.block_size,
                    self._device,
                )
                self._verify_meta[vbucket] = vmeta
                for greedy in (True, False):
                    self._verify_graphs[vbucket, greedy] = self._capture_verify(
                        vbucket, greedy, vmeta
                    )
                torch.cuda.synchronize(self._device)
                after = torch.cuda.memory_reserved(self._device)
                kept_verify.append(vbucket)
                results.append(
                    CapturedBucket(vbucket, max(after - before, 0), time.monotonic() - started)
                )
            except torch.cuda.OutOfMemoryError as exc:
                self._verify_meta.pop(vbucket, None)
                self._verify_graphs.pop((vbucket, True), None)
                self._verify_graphs.pop((vbucket, False), None)
                torch.cuda.empty_cache()
                logger.warning(
                    "dropping verify CUDA graph bucket %d due to out-of-memory: %s",
                    vbucket,
                    exc,
                )
        self._verify_buckets = tuple(sorted(kept_verify))
        logger.info(
            "captured CUDA graphs for decode buckets %s and verify buckets %s: %d bytes in %.1fs",
            list(self._buckets),
            list(self._verify_buckets),
            sum(item.bytes for item in results),
            sum(item.seconds for item in results),
        )
        return results

    def _capture_verify(self, bucket: int, greedy: bool, meta: AttentionMetadata) -> Any:
        tokens, batch, sampled = self._builder.graph_views(bucket)
        batch = replace(batch, all_greedy=greedy)
        context = ForwardContext(
            attention=meta, kv_buffers=self._kv_buffers, feature_sink=self._feature_sink
        )

        def body() -> None:
            hidden = self._model(tokens, context)
            if self._hidden_buf is not None:
                self._hidden_buf[:bucket].copy_(hidden)
            logits = self._model.compute_logits(hidden)
            self._sampler.sample(logits, batch, sampled, None)

        self._reset_verify_inputs(bucket)
        side = _warmup_stream(self._device)
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

    def _reset_verify_inputs(self, bucket: int) -> None:
        builder = self._builder
        rows = torch.zeros(builder.max_requests, dtype=torch.int32)
        rows[0] = bucket
        lengths = torch.zeros(builder.max_requests, dtype=torch.int32)
        table = torch.zeros((builder.max_requests, builder.max_blocks), dtype=torch.int32)
        self._model.ops.attention.update_verify_graph_metadata(
            self._verify_meta[bucket], rows, lengths, table
        )
        builder.reset_graph_inputs(bucket)

    def _capture_one(
        self, bucket: int, greedy: bool, masked: bool, metas: tuple[AttentionMetadata, ...]
    ) -> Any:
        tokens, batch, sampled = self._builder.graph_views(bucket)
        batch = replace(batch, all_greedy=greedy)
        layer_attention = {
            layer_id: meta
            for meta, layer_ids in zip(metas, self._group_layers, strict=False)
            for layer_id in layer_ids
        }
        linear_meta = self._builder.graph_linear_metadata(bucket) if self._state_buffers else None
        context = ForwardContext(
            attention=metas[0],
            kv_buffers=self._kv_buffers,
            layer_attention=layer_attention,
            latent_kv_buffers=self._latent_kv_buffers,
            state_buffers=self._state_buffers,
            linear_attention=linear_meta,
            feature_sink=self._feature_sink,
        )
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
            if self._hidden_buf is not None:
                self._hidden_buf[:bucket].copy_(hidden)
            logits = self._model.compute_logits(hidden)
            self._sampler.sample(logits, batch, sampled, controls)

        self._reset_inputs(bucket)
        side = _warmup_stream(self._device)
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
        for meta in self._meta[bucket]:
            self._model.ops.attention.update_graph_metadata(meta, lengths, rows)
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
        group_rows = staged.group_block_rows or (staged.block_rows,)
        for meta, rows in zip(self._meta[bucket], group_rows, strict=True):
            self._model.ops.attention.update_graph_metadata(meta, staged.kv_lengths, rows)
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

    def launch_verify(self, step: StepInput, drafts: Sequence[Sequence[TokenId]]) -> StagedVerify:
        """Stage a ``VERIFY`` step into a captured total-row bucket and queue its replay."""
        bucket = self.bucket_for(step)
        if bucket is None or step.mode is not ForwardMode.VERIFY:
            raise EngineError(f"verify step {step.step_id} cannot be replayed by a verify graph")
        staged = self._builder.stage_verify(step, drafts, bucket)
        self._model.ops.attention.update_verify_graph_metadata(
            self._verify_meta[bucket],
            staged.rows_per_request,
            staged.kv_lengths,
            staged.block_rows,
        )
        self._verify_graphs[bucket, staged.all_greedy].replay()
        self.verify_replays += 1
        return staged

    def run_verify(
        self, step: StepInput, drafts: Sequence[Sequence[TokenId]]
    ) -> tuple[StagedVerify, list[TokenId]]:
        """Replay a ``VERIFY`` step and return ``(staged, sampled_tokens)`` in one D2H sync."""
        staged = self.launch_verify(step, drafts)
        return staged, self._builder.read_sampled(staged.num_rows)


#: The stream the warm-up runs of every capture on a device use, kept for the process.
_warmup_streams: dict[torch.device, torch.cuda.Stream] = {}


def _warmup_stream(device: torch.device) -> torch.cuda.Stream:
    """The one side stream the warm-up runs before every capture on ``device`` share.

    PyTorch gives every (cuBLAS handle, stream) pair a workspace from the caching allocator
    (8,519,680 bytes on an L4) and keeps it until the process exits, even after the stream is
    gone. A new stream per captured graph therefore left about 100 MiB behind every engine that
    captured graphs; one stream per device, like the capture stream PyTorch itself keeps, costs
    one workspace per device.
    """
    stream = _warmup_streams.get(device)
    if stream is None:
        stream = torch.cuda.Stream(device)
        _warmup_streams[device] = stream
    return stream
