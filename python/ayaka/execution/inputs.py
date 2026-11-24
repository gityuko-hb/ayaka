"""From a step description to the tensors a model step runs on, in buffers allocated once.

:class:`InputBuilder` turns a :class:`~ayaka.scheduler.step.StepInput` into everything one forward
needs: the token ids, the planned attention metadata, the rows whose logits are sampled and the
sampling parameters of those rows. Every input lives in a buffer that is allocated when the builder
is made, at the size of the largest step, and a step only writes the part it uses. Nothing is
allocated on the hot path, and the data pointer of each buffer is the same for the life of the
builder, which is what a CUDA graph will need and what ``tests/execution`` checks.

Each buffer is a pair: a host tensor, pinned when the device is a GPU, that Python fills through a
NumPy view of the same memory, and a device tensor that the step copies the used part into. On the
CPU the two are one tensor and no copy happens.

The block table is mirrored the same way. The scheduler reports, in ``StepInput.table_updates``,
which ranges of the host :class:`~ayaka.kvcache.table.BlockTable` changed; the builder copies only
those into the device mirror and gathers the rows of the step's requests from it. If a step fails
before its updates were applied, the next step copies the whole table, so a mirror never stays
stale.

This module imports torch. Import it through ``ayaka.execution``, which loads it on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ayaka.core.errors import EngineError
from ayaka.core.types import ForwardMode
from ayaka.execution.staged import ModelInputs, StagedDecode, StagedVerify
from ayaka.ops.linear_attention import LinearAttentionMetadata
from ayaka.sampling.batch import SamplingBatch
from ayaka.sampling.rng import resolve_seed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from ayaka.core.types import TokenId
    from ayaka.kvcache.table import BlockTable
    from ayaka.ops.attention import AttentionMetadata, AttentionOps
    from ayaka.scheduler.step import StepInput

__all__ = ["InputBuilder"]


class _Mirrored:
    """A fixed host buffer, a NumPy view of it, and the fixed device buffer it is copied to.

    On the CPU the host and device tensors are the same object, so a push is a no-op.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
        on_host = device.type == "cpu"
        self.host = torch.zeros(shape, dtype=dtype, pin_memory=not on_host)
        self.array: NDArray[Any] = self.host.numpy()
        self.device = self.host if on_host else torch.zeros(shape, dtype=dtype, device=device)

    def push(self, stop: int) -> None:
        """Copy the first ``stop`` entries along the first dimension to the device."""
        if self.device is not self.host and stop:
            self.device[:stop].copy_(self.host[:stop], non_blocking=True)

    def pull(self, stop: int) -> None:
        """Copy the first ``stop`` entries back to the host and wait for them."""
        if self.device is not self.host and stop:
            self.host[:stop].copy_(self.device[:stop])


class InputBuilder:
    """Builds :class:`ModelInputs` from steps without allocating per step.

    Args:
        attention: The attention ops of the backend; ``plan`` runs once per KV group per step.
        table: The host block table the scheduler writes.
        block_size: Tokens per cache block.
        max_rows: Most rows (tokens) one step can have.
        device: The device the model runs on.
        has_linear_attention: Whether the model uses recurrent state slots.
    """

    def __init__(
        self,
        *,
        attention: AttentionOps,
        table: BlockTable,
        block_size: int,
        max_rows: int,
        device: torch.device | str,
        has_linear_attention: bool = False,
    ) -> None:
        target = torch.device(device)
        self._target = target
        self._attention = attention
        self._table = table
        self._num_groups = table.num_groups
        self._block_size = block_size
        self._max_rows = max_rows
        self._max_requests = table.max_requests
        self._has_linear_attention = has_linear_attention
        self._stale = False

        rows, requests = max_rows, table.max_requests
        max_blocks = table.max_blocks_per_request
        self._tokens = _Mirrored((rows,), torch.int64, target)
        self._rows_per_request = _Mirrored((requests,), torch.int32, target)
        self._kv_lengths = _Mirrored((requests,), torch.int32, target)
        self._batch_rows = _Mirrored((requests,), torch.int64, target)
        self._mirrors = tuple(
            _Mirrored((requests, max_blocks), torch.int32, target) for _ in range(self._num_groups)
        )
        self._group_block_rows = tuple(
            torch.zeros((requests, max_blocks), dtype=torch.int32, device=target)
            for _ in range(self._num_groups)
        )
        self._state_slots = _Mirrored((requests,), torch.int32, target)
        self._cu_seqlens = _Mirrored((requests + 1,), torch.int32, target)
        self._logit_rows = _Mirrored((rows,), torch.int64, target)
        self._seeds = _Mirrored((rows,), torch.int64, target)
        self._positions = _Mirrored((rows,), torch.int64, target)
        self._sequences = _Mirrored((rows,), torch.int32, target)  # the output sequence (n > 1)
        self._temperature = _Mirrored((rows,), torch.float32, target)
        self._top_k = _Mirrored((rows,), torch.int32, target)
        self._top_p = _Mirrored((rows,), torch.float32, target)
        self._min_p = _Mirrored((rows,), torch.float32, target)
        self._sampled = _Mirrored((rows,), torch.int64, target)
        # Host staging of a graph step's lengths and block tables, handed to the backend.
        pinned = target.type != "cpu"
        self._graph_lengths = torch.zeros(requests, dtype=torch.int32, pin_memory=pinned)
        self._graph_rows = tuple(
            torch.zeros((requests, max_blocks), dtype=torch.int32, pin_memory=pinned)
            for _ in range(self._num_groups)
        )
        self._verify_rows = torch.zeros(requests, dtype=torch.int32, pin_memory=pinned)
        # The token masks the masked graph variants read, made by `graph_mask` (it knows the width).
        self._masks: _Mirrored | None = None

    @property
    def num_groups(self) -> int:
        """Number of paged KV or latent KV groups planned per step."""
        return self._num_groups

    @property
    def max_rows(self) -> int:
        """Most rows one step can have."""
        return self._max_rows

    def active_block_tables(self, step: StepInput) -> torch.Tensor:
        """Device ``[num_requests, max_blocks]`` block-table rows for ``step.batch_rows``."""
        count = step.num_requests
        self._batch_rows.array[:count] = step.batch_rows
        self._batch_rows.push(count)
        block_rows = self._group_block_rows[0][:count]
        torch.index_select(
            self._mirrors[0].device, 0, self._batch_rows.device[:count], out=block_rows
        )
        return block_rows

    @property
    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Every device buffer by name, for tests that check their addresses stay put."""
        buffers = {
            "tokens": self._tokens.device,
            "rows_per_request": self._rows_per_request.device,
            "kv_lengths": self._kv_lengths.device,
            "batch_rows": self._batch_rows.device,
            "block_table": self._mirrors[0].device,
            "block_rows": self._group_block_rows[0],
            "state_slots": self._state_slots.device,
            "cu_seqlens": self._cu_seqlens.device,
            "logit_rows": self._logit_rows.device,
            "seeds": self._seeds.device,
            "positions": self._positions.device,
            "sequences": self._sequences.device,
            "temperature": self._temperature.device,
            "top_k": self._top_k.device,
            "top_p": self._top_p.device,
            "min_p": self._min_p.device,
            "sampled": self._sampled.device,
        }
        if self._masks is not None:
            buffers["token_masks"] = self._masks.device
        return buffers

    def resync(self) -> None:
        """Copy the whole host block table across every group to the device mirror."""
        for group, mirror in enumerate(self._mirrors):
            mirror.array[:] = self._table.rows[group]
            mirror.push(self._max_requests)
        self._stale = False

    def build(
        self, step: StepInput, drafts: Sequence[Sequence[TokenId]] | None = None
    ) -> ModelInputs:
        """Write ``step`` into the buffers and plan its attention.

        Args:
            step: The step to run.
            drafts: Per request, the tokens of its draft rows (``step.draft_slots`` of them).
                A draft row without one repeats the request's last known token; verification
                makes a wrong guess cost time, never correctness.

        Raises:
            EngineError: If the step is larger than the buffers, a request's batch row is out of
                range, a draft list has the wrong length, or the table update names another group.
            BackendError: If the backend refuses the plan, for example because the block table
                mirror does not cover a request's length.
        """
        if step.num_requests > self._max_requests or step.num_rows > self._max_rows:
            raise EngineError(
                f"step {step.step_id} has {step.num_requests} requests and {step.num_rows} rows; "
                f"the buffers hold {self._max_requests} and {self._max_rows}"
            )
        try:
            return self._build(step, drafts)
        except BaseException:
            self._stale = True
            raise

    def _build(self, step: StepInput, drafts: Sequence[Sequence[TokenId]] | None) -> ModelInputs:
        self._apply_table_updates(step)
        count = step.num_requests
        slots = step.draft_slots
        if drafts is not None and len(drafts) != count:
            raise EngineError(f"{len(drafts)} draft lists for {count} requests")

        used_drafts: list[tuple[TokenId, ...]] = []
        offset = 0
        for index in range(count):
            known = step.tokens[index]
            wanted = slots[index]
            given = tuple(drafts[index]) if drafts is not None else ()
            if len(given) > wanted:
                raise EngineError(
                    f"request {step.request_ids[index]!r} has {wanted} draft rows, got {len(given)}"
                )
            filled = (*given, *([known[-1]] * (wanted - len(given))))
            used_drafts.append(filled)
            end = offset + step.rows_per_request[index]
            self._tokens.array[offset : offset + len(known)] = known
            self._tokens.array[offset + len(known) : end] = filled
            offset = end
        self._tokens.push(offset)

        batch_rows = step.batch_rows
        if max(batch_rows) >= self._max_requests or min(batch_rows) < 0:
            raise EngineError(f"step {step.step_id} uses a batch row outside the block table")
        self._rows_per_request.array[:count] = step.rows_per_request
        self._kv_lengths.array[:count] = [
            start + rows
            for start, rows in zip(step.start_positions, step.rows_per_request, strict=True)
        ]
        self._batch_rows.array[:count] = batch_rows
        for buffer in (self._rows_per_request, self._kv_lengths, self._batch_rows):
            buffer.push(count)

        group_attentions: list[AttentionMetadata] = []
        rows_dev = self._rows_per_request.device[:count]
        lens_dev = self._kv_lengths.device[:count]
        batch_dev = self._batch_rows.device[:count]
        for group in range(self._num_groups):
            block_rows = self._group_block_rows[group][:count]
            torch.index_select(self._mirrors[group].device, 0, batch_dev, out=block_rows)
            group_attentions.append(
                self._attention.plan(
                    step.mode,
                    rows_dev,
                    lens_dev,
                    block_rows,
                    self._block_size,
                )
            )

        linear_attention: LinearAttentionMetadata | None = None
        if step.state_slots or self._has_linear_attention:
            self._state_slots.array[:count] = step.state_slots or (0,) * count
            self._state_slots.push(count)
            self._cu_seqlens.array[0] = 0
            np.cumsum(
                step.rows_per_request, dtype=np.int32, out=self._cu_seqlens.array[1 : count + 1]
            )
            self._cu_seqlens.push(count + 1)
            linear_attention = LinearAttentionMetadata(
                is_prefill=step.mode is ForwardMode.PREFILL,
                cu_seqlens=self._cu_seqlens.device[: count + 1],
                state_slots=self._state_slots.device[:count],
                rows_per_request=rows_dev,
            )

        logit_counts = self._write_logit_rows(step)
        total = sum(logit_counts)
        return ModelInputs(
            token_ids=self._tokens.device[:offset],
            attention=group_attentions[0],
            logit_rows=self._logit_rows.device[:total],
            sampling=self._sampling_batch(step, logit_counts),
            sampled=self._sampled.device[:total],
            logit_counts=logit_counts,
            draft_tokens=tuple(used_drafts),
            group_attentions=tuple(group_attentions),
            linear_attention=linear_attention,
        )

    @property
    def max_requests(self) -> int:
        """Most requests one step can have: the rows of the block table."""
        return self._max_requests

    def reset_graph_inputs(self, bucket: int) -> None:
        """Fill the first ``bucket`` rows with a valid all-padding step, for a capture.

        A decode bucket counts requests, a verify bucket rows; the per-request buffers are
        filled for at most the requests a step can hold.
        """
        self._tokens.array[:bucket] = 0
        self._tokens.push(bucket)
        requests = min(bucket, self._max_requests)
        self._state_slots.array[:requests] = 0
        self._state_slots.push(requests)
        self._rows_per_request.array[:requests] = 1
        self._rows_per_request.push(requests)
        self._cu_seqlens.array[: requests + 1] = np.arange(requests + 1, dtype=np.int32)
        self._cu_seqlens.push(requests + 1)
        for buffer, neutral in (
            (self._seeds, 0),
            (self._positions, 0),
            (self._sequences, 0),
            (self._temperature, 1.0),
            (self._top_k, 0),
            (self._top_p, 1.0),
            (self._min_p, 0.0),
        ):
            buffer.array[:bucket] = neutral
            buffer.push(bucket)

    def graph_linear_metadata(self, bucket: int) -> LinearAttentionMetadata:
        """Return fixed device-buffer views for captured decode linear-attention steps."""
        self._rows_per_request.array[:bucket] = 1
        self._rows_per_request.push(bucket)
        self._cu_seqlens.array[: bucket + 1] = np.arange(bucket + 1, dtype=np.int32)
        self._cu_seqlens.push(bucket + 1)
        return LinearAttentionMetadata(
            is_prefill=False,
            cu_seqlens=self._cu_seqlens.device[: bucket + 1],
            state_slots=self._state_slots.device[:bucket],
            rows_per_request=self._rows_per_request.device[:bucket],
        )

    def stage_verify(
        self,
        step: StepInput,
        drafts: Sequence[Sequence[TokenId]],
        bucket: int,
    ) -> StagedVerify:
        """Write a ``VERIFY`` step padded up to ``bucket`` total rows for graph replay."""
        count = step.num_requests
        total_rows = step.num_rows
        if step.mode is not ForwardMode.VERIFY:
            raise EngineError(f"step {step.step_id} is not a VERIFY step")
        if total_rows > bucket or bucket > self._max_rows or count > self._max_requests:
            raise EngineError(
                f"step {step.step_id} has {total_rows} rows and {count} requests for verify "
                f"bucket {bucket} (max_rows={self._max_rows}, max_requests={self._max_requests})"
            )
        try:
            self._apply_table_updates(step)
        except BaseException:
            self._stale = True
            raise
        batch_rows = step.batch_rows
        if max(batch_rows) >= self._max_requests or min(batch_rows) < 0:
            raise EngineError(f"step {step.step_id} uses a batch row outside the block table")

        used_drafts: list[tuple[TokenId, ...]] = []
        offset = 0
        slots = step.draft_slots
        for index in range(count):
            known = step.tokens[index]
            wanted = slots[index]
            given = tuple(drafts[index]) if index < len(drafts) else ()
            filled = (*given[:wanted], *([known[-1]] * max(0, wanted - len(given))))
            used_drafts.append(filled)
            end = offset + step.rows_per_request[index]
            self._tokens.array[offset : offset + len(known)] = known
            self._tokens.array[offset + len(known) : end] = filled
            offset = end
        self._tokens.array[offset:bucket] = 0
        self._tokens.push(bucket)

        rows_arr = self._verify_rows.numpy()
        rows_arr[:count] = step.rows_per_request
        pad_rows = bucket - total_rows
        if count < self._max_requests:
            rows_arr[count] = pad_rows
            rows_arr[count + 1 :] = 0
        elif pad_rows > 0:
            # All max_requests slots are active; cannot add a separate padding request.
            raise EngineError(f"step {step.step_id}: no padding slot left for bucket {bucket}")

        lengths = self._graph_lengths.numpy()
        lengths[:count] = [
            start + rows
            for start, rows in zip(step.start_positions, step.rows_per_request, strict=True)
        ]
        lengths[count:] = 0
        rows = self._graph_rows[0].numpy()
        rows[:count] = self._table.rows[0, list(batch_rows)]
        rows[count:] = 0

        logit_counts = step.rows_per_request
        greedy = self._sampling_batch(step, logit_counts).all_greedy
        for buffer, neutral in (
            (self._seeds, 0),
            (self._positions, 0),
            (self._sequences, 0),
            (self._temperature, 1.0),
            (self._top_k, 0),
            (self._top_p, 1.0),
            (self._min_p, 0.0),
        ):
            buffer.array[total_rows:bucket] = neutral
            buffer.push(bucket)

        return StagedVerify(
            bucket=bucket,
            num_rows=total_rows,
            num_requests=count,
            rows_per_request=self._verify_rows,
            kv_lengths=self._graph_lengths,
            block_rows=self._graph_rows[0],
            all_greedy=greedy,
            draft_tokens=tuple(used_drafts),
            logit_counts=logit_counts,
        )

    @property
    def max_blocks(self) -> int:
        """Entries of one request's block table."""
        return int(self._graph_rows[0].shape[1])

    @property
    def block_size(self) -> int:
        """Tokens per cache block."""
        return self._block_size

    def graph_views(self, bucket: int) -> tuple[torch.Tensor, SamplingBatch, torch.Tensor]:
        """The token ids, sampling parameters and sampled-token output of a bucket's graph.

        Views of the first ``bucket`` entries of the fixed buffers: the same tensors every step,
        so a graph captured on them reads what :meth:`stage_decode` writes. ``all_greedy`` of the
        returned batch is ``False``; a caller that captures the greedy variant replaces it.
        """
        batch = SamplingBatch(
            seeds=self._seeds.device[:bucket],
            positions=self._positions.device[:bucket],
            sequences=self._sequences.device[:bucket],
            temperature=self._temperature.device[:bucket],
            top_k=self._top_k.device[:bucket],
            top_p=self._top_p.device[:bucket],
            min_p=self._min_p.device[:bucket],
            all_greedy=False,
        )
        return self._tokens.device[:bucket], batch, self._sampled.device[:bucket]

    def graph_mask(self, bucket: int, words: int) -> torch.Tensor:
        """The ``[bucket, words]`` int32 token-mask buffer a masked graph reads.

        The buffer is made on the first call, at the size of the largest step, so every bucket's
        graph reads a view of the same tensor; every bit is set (every token allowed) until
        :meth:`stage_mask` writes a step's masks. A bit is a token, as in
        :func:`~ayaka.ops.sampling.SamplingOps.apply_token_bitmask`: bit ``t % 32`` of word
        ``t // 32``.

        Raises:
            EngineError: If an earlier call used another width.
        """
        if self._masks is None:
            self._masks = _Mirrored((self._max_requests, words), torch.int32, self._target)
            self._masks.array[:] = -1
            self._masks.push(self._max_requests)
        elif self._masks.array.shape[1] != words:
            raise EngineError(
                f"the token masks are {self._masks.array.shape[1]} words wide, not {words}"
            )
        return self._masks.device[:bucket]

    def stage_mask(self, masks: Mapping[int, NDArray[np.uint32]], bucket: int) -> None:
        """Write the token masks of a decode step: one copy to the device, no wait.

        Args:
            masks: The allowed tokens, as ``uint32`` words, of the rows that have a constraint,
                by row. Every other row of the bucket, padding included, allows every token.
            bucket: Rows the graph runs.

        Raises:
            EngineError: If no masked graph made the buffer, or a row is outside the bucket.
        """
        buffer = self._masks
        if buffer is None:
            raise EngineError("no token mask buffer: capture a masked graph first")
        if any(not 0 <= row < bucket for row in masks):
            raise EngineError(f"a token mask names a row outside the bucket of {bucket}")
        array = buffer.array
        array[:bucket] = -1
        for row, words in masks.items():
            array[row] = words[: array.shape[1]].view(np.int32)
        buffer.push(bucket)

    def stage_decode(self, step: StepInput, bucket: int) -> StagedDecode:
        """Write a decode step, padded to ``bucket`` requests, for a captured graph to replay.

        The step's requests fill the first rows; the rest are padding requests (length 0, token 0,
        greedy). Token ids and sampling parameters go to the fixed device buffers; the lengths and
        block tables come back as host tensors for ``ops.attention.update_graph_metadata``. The
        block table mirror is updated as :meth:`build` does, so an eager step after this one sees
        the same table.

        Raises:
            EngineError: If the step is not a decode step of at most ``bucket`` requests, or the
                bucket is larger than the buffers.
        """
        count = step.num_requests
        if step.mode is not ForwardMode.DECODE or step.num_rows != count:
            raise EngineError(f"step {step.step_id} is not a decode step; graphs run only those")
        if not count <= bucket <= min(self._max_requests, self._max_rows):
            raise EngineError(
                f"step {step.step_id} has {count} requests for a bucket of {bucket}; the buffers "
                f"hold {min(self._max_requests, self._max_rows)}"
            )
        try:
            self._apply_table_updates(step)
        except BaseException:
            self._stale = True
            raise
        batch_rows = step.batch_rows
        if max(batch_rows) >= self._max_requests or min(batch_rows) < 0:
            raise EngineError(f"step {step.step_id} uses a batch row outside the block table")

        self._tokens.array[:count] = [known[0] for known in step.tokens]
        self._tokens.array[count:bucket] = 0
        self._tokens.push(bucket)

        lengths = self._graph_lengths.numpy()
        lengths[:count] = [start + 1 for start in step.start_positions]
        lengths[count:bucket] = 0
        group_rows: list[torch.Tensor] = []
        batch_list = list(batch_rows)
        for group in range(self._num_groups):
            rows_arr = self._graph_rows[group].numpy()
            rows_arr[:count] = self._table.rows[group, batch_list]
            rows_arr[count:bucket] = 0
            group_rows.append(self._graph_rows[group][:bucket])

        if step.state_slots or self._has_linear_attention:
            self._state_slots.array[:count] = step.state_slots or (0,) * count
            self._state_slots.array[count:bucket] = 0
            self._state_slots.push(bucket)

        logit_counts = (1,) * count
        greedy = self._sampling_batch(step, logit_counts).all_greedy
        for buffer, neutral in (
            (self._seeds, 0),
            (self._positions, 0),
            (self._temperature, 0.0),
            (self._top_k, 0),
            (self._top_p, 1.0),
            (self._min_p, 0.0),
        ):
            buffer.array[count:bucket] = neutral
            buffer.push(bucket)
        return StagedDecode(
            bucket=bucket,
            count=count,
            kv_lengths=self._graph_lengths[:bucket],
            block_rows=group_rows[0],
            all_greedy=greedy,
            group_block_rows=tuple(group_rows),
            state_slots=self._state_slots.device[:bucket]
            if (step.state_slots or self._has_linear_attention)
            else None,
        )

    def read_sampled(self, count: int) -> list[TokenId]:
        """Bring the first ``count`` sampled token ids to the host; this waits for the device."""
        self._sampled.pull(count)
        return [int(token) for token in self._sampled.array[:count].tolist()]

    def _apply_table_updates(self, step: StepInput) -> None:
        if self._stale:
            self.resync()
            return
        rows = self._table.rows
        for group, row, start, stop in step.table_updates:
            if not 0 <= group < self._num_groups:
                raise EngineError(
                    f"the block table update names group {group}; "
                    f"the table has {self._num_groups} group(s)"
                )
            mirror = self._mirrors[group]
            mirror.array[row, start:stop] = rows[group, row, start:stop]
            if mirror.device is not mirror.host:
                mirror.device[row, start:stop].copy_(
                    mirror.host[row, start:stop], non_blocking=True
                )

    def _write_logit_rows(self, step: StepInput) -> tuple[int, ...]:
        """Choose the rows to sample and write their indices; return the count per request."""
        counts: list[int] = []
        indices = self._logit_rows.array
        written = 0
        offset = 0
        for rows, wanted in zip(step.rows_per_request, step.need_logits, strict=True):
            first = offset + (rows - 1 if step.mode is ForwardMode.PREFILL else 0)
            last = offset + rows
            taken = last - first if wanted else 0
            if taken:
                indices[written : written + taken] = np.arange(first, last)
            counts.append(taken)
            written += taken
            offset = last
        self._logit_rows.push(written)
        return tuple(counts)

    def _sampling_batch(self, step: StepInput, logit_counts: tuple[int, ...]) -> SamplingBatch:
        written = 0
        greedy = True
        for index, taken in enumerate(logit_counts):
            if not taken:
                continue
            params = step.sampling_params[index]
            seed = resolve_seed(step.request_ids[index], params.seed)
            # The row's first produced token is one past the position of its input.
            first = step.start_positions[index] + 1
            if step.mode is ForwardMode.PREFILL:
                first = step.start_positions[index] + step.rows_per_request[index]
            span = slice(written, written + taken)
            self._seeds.array[span] = seed
            self._sequences.array[span] = step.sequence_index(index)
            self._positions.array[span] = np.arange(first, first + taken)
            if params.is_greedy:
                temperature, top_k, top_p, min_p = 0.0, 0, 1.0, 0.0
            else:
                temperature = params.temperature
                top_k, top_p, min_p = max(params.top_k, 0), params.top_p, params.min_p
                greedy = False
            self._temperature.array[span] = temperature
            self._top_k.array[span] = top_k
            self._top_p.array[span] = top_p
            self._min_p.array[span] = min_p
            written += taken
        for buffer in (
            self._seeds,
            self._positions,
            self._sequences,
            self._temperature,
            self._top_k,
            self._top_p,
            self._min_p,
        ):
            buffer.push(written)
        return SamplingBatch(
            seeds=self._seeds.device[:written],
            positions=self._positions.device[:written],
            sequences=self._sequences.device[:written],
            temperature=self._temperature.device[:written],
            top_k=self._top_k.device[:written],
            top_p=self._top_p.device[:written],
            min_p=self._min_p.device[:written],
            all_greedy=greedy,
        )
