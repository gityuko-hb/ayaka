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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ayaka.core.errors import EngineError
from ayaka.core.types import ForwardMode
from ayaka.sampling.batch import SamplingBatch
from ayaka.sampling.rng import resolve_seed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from ayaka.core.types import TokenId
    from ayaka.kvcache.table import BlockTable
    from ayaka.ops.attention import AttentionMetadata, AttentionOps
    from ayaka.scheduler.step import StepInput

__all__ = ["InputBuilder", "ModelInputs", "StagedDecode"]


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


@dataclass(frozen=True, slots=True, eq=False)
class StagedDecode:
    """A decode step written into the fixed buffers at a bucket's size, for a captured graph.

    Attributes:
        bucket: Rows the graph runs: the step's requests, then padding requests.
        count: The step's own requests; their rows come first.
        kv_lengths: ``[bucket]`` int32 CPU tensor, 0 for padding requests.
        block_rows: ``[bucket, max_blocks]`` int32 CPU tensor of the requests' block tables.
        all_greedy: Whether every request samples greedily, which picks the graph variant.
    """

    bucket: int
    count: int
    kv_lengths: torch.Tensor
    block_rows: torch.Tensor
    all_greedy: bool


@dataclass(frozen=True, slots=True, eq=False)
class ModelInputs:
    """The inputs of one forward, as views of the builder's fixed buffers.

    Attributes:
        token_ids: ``[N]`` int64 input token of every row, requests in step order.
        attention: The planned step, shared by every layer.
        logit_rows: ``[L]`` int64 indices of the rows whose hidden states are projected and
            sampled: the last row of a prefill chunk that ends its prompt, every row of a decode
            or verify request.
        sampling: The sampling parameters of those ``L`` rows.
        sampled: ``[L]`` int64 buffer the sampler writes the chosen token ids into.
        logit_counts: Per request, how many of the ``L`` rows are its own (0 for a prefill chunk
            that does not end the prompt).
        draft_tokens: Per request, the tokens the step used for its draft rows.
    """

    token_ids: torch.Tensor
    attention: AttentionMetadata
    logit_rows: torch.Tensor
    sampling: SamplingBatch
    sampled: torch.Tensor
    logit_counts: tuple[int, ...]
    draft_tokens: tuple[tuple[TokenId, ...], ...]


class InputBuilder:
    """Builds :class:`ModelInputs` from steps without allocating per step.

    Args:
        attention: The attention ops of the backend; ``plan`` runs once per step.
        table: The host block table the scheduler writes.
        block_size: Tokens per cache block.
        max_rows: Most rows (tokens) one step can have.
        device: The device the model runs on.

    Raises:
        EngineError: If the table has more than one KV group; a step is planned for one group.
    """

    def __init__(
        self,
        *,
        attention: AttentionOps,
        table: BlockTable,
        block_size: int,
        max_rows: int,
        device: torch.device | str,
    ) -> None:
        if table.num_groups != 1:
            raise EngineError(
                f"the input builder plans a step for one KV group, the block table has "
                f"{table.num_groups}"
            )
        target = torch.device(device)
        self._attention = attention
        self._table = table
        self._block_size = block_size
        self._max_rows = max_rows
        self._max_requests = table.max_requests
        self._stale = False

        rows, requests = max_rows, table.max_requests
        self._tokens = _Mirrored((rows,), torch.int64, target)
        self._rows_per_request = _Mirrored((requests,), torch.int32, target)
        self._kv_lengths = _Mirrored((requests,), torch.int32, target)
        self._batch_rows = _Mirrored((requests,), torch.int64, target)
        self._mirror = _Mirrored((requests, table.max_blocks_per_request), torch.int32, target)
        self._block_rows = torch.zeros(
            (requests, table.max_blocks_per_request), dtype=torch.int32, device=target
        )
        self._logit_rows = _Mirrored((rows,), torch.int64, target)
        self._seeds = _Mirrored((rows,), torch.int64, target)
        self._positions = _Mirrored((rows,), torch.int64, target)
        self._sequences = _Mirrored((rows,), torch.int32, target)  # always 0: one sequence each
        self._temperature = _Mirrored((rows,), torch.float32, target)
        self._top_k = _Mirrored((rows,), torch.int32, target)
        self._top_p = _Mirrored((rows,), torch.float32, target)
        self._min_p = _Mirrored((rows,), torch.float32, target)
        self._sampled = _Mirrored((rows,), torch.int64, target)
        # Host staging of a graph step's lengths and block tables, handed to the backend.
        pinned = target.type != "cpu"
        self._graph_lengths = torch.zeros(requests, dtype=torch.int32, pin_memory=pinned)
        self._graph_rows = torch.zeros(
            (requests, table.max_blocks_per_request), dtype=torch.int32, pin_memory=pinned
        )

    @property
    def max_rows(self) -> int:
        """Most rows one step can have."""
        return self._max_rows

    @property
    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Every device buffer by name, for tests that check their addresses stay put."""
        return {
            "tokens": self._tokens.device,
            "rows_per_request": self._rows_per_request.device,
            "kv_lengths": self._kv_lengths.device,
            "batch_rows": self._batch_rows.device,
            "block_table": self._mirror.device,
            "block_rows": self._block_rows,
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

    def resync(self) -> None:
        """Copy the whole host block table to the device mirror."""
        self._mirror.array[:] = self._table.rows[0]
        self._mirror.push(self._max_requests)
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

        block_rows = self._block_rows[:count]
        torch.index_select(self._mirror.device, 0, self._batch_rows.device[:count], out=block_rows)
        attention = self._attention.plan(
            step.mode,
            self._rows_per_request.device[:count],
            self._kv_lengths.device[:count],
            block_rows,
            self._block_size,
        )

        logit_counts = self._write_logit_rows(step)
        total = sum(logit_counts)
        return ModelInputs(
            token_ids=self._tokens.device[:offset],
            attention=attention,
            logit_rows=self._logit_rows.device[:total],
            sampling=self._sampling_batch(step, logit_counts),
            sampled=self._sampled.device[:total],
            logit_counts=logit_counts,
            draft_tokens=tuple(used_drafts),
        )

    @property
    def max_requests(self) -> int:
        """Most requests one step can have: the rows of the block table."""
        return self._max_requests

    def reset_graph_inputs(self, bucket: int) -> None:
        """Fill the first ``bucket`` rows with a valid all-padding decode step, for a capture."""
        self._tokens.array[:bucket] = 0
        self._tokens.push(bucket)
        for buffer, neutral in (
            (self._seeds, 0),
            (self._positions, 0),
            (self._temperature, 1.0),
            (self._top_k, 0),
            (self._top_p, 1.0),
            (self._min_p, 0.0),
        ):
            buffer.array[:bucket] = neutral
            buffer.push(bucket)

    @property
    def max_blocks(self) -> int:
        """Entries of one request's block table."""
        return int(self._graph_rows.shape[1])

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
        rows = self._graph_rows.numpy()
        rows[:count] = self._table.rows[0, list(batch_rows)]
        rows[count:bucket] = 0

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
            block_rows=self._graph_rows[:bucket],
            all_greedy=greedy,
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
            if group != 0:
                raise EngineError(f"the block table update names group {group}; only group 0 runs")
            self._mirror.array[row, start:stop] = rows[0, row, start:stop]
            if self._mirror.device is not self._mirror.host:
                self._mirror.device[row, start:stop].copy_(
                    self._mirror.host[row, start:stop], non_blocking=True
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
