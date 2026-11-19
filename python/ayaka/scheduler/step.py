"""The contract of one engine step: what the scheduler hands the runner and what comes back.

``StepInput`` and ``StepOutput`` are immutable snapshots. A ``StepInput`` holds no
reference to the scheduler's live :class:`~ayaka.scheduler.request.Request` objects: every field
is a tuple of plain values copied at ``schedule()`` time, so the scheduler can plan the next step,
or a request can be aborted, while the runner still reads this one. ``StepOutput`` carries
everything ``update()`` needs, so the scheduler never asks the runner anything.

Both live here, in the scheduler layer, and ``execution`` imports them from here. A request may take
**several rows** in one step: a prefill chunk of ``n`` tokens is ``n`` rows, and a decode step with
``k`` look-ahead (draft) rows is ``1 + k`` rows. The scheduler knows only the number ``k``; it fixes
how many rows the step reserves, and the runner fills the draft rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayaka.core.errors import EngineError
from ayaka.core.types import ForwardMode

if TYPE_CHECKING:
    from ayaka.core.errors import ErrorInfo
    from ayaka.core.outputs import TokenLogprob
    from ayaka.core.params import SamplingParams
    from ayaka.core.types import RequestId, TokenId

__all__ = ["StepInput", "StepOutput"]

_STEP_MODES: Final = frozenset({ForwardMode.PREFILL, ForwardMode.DECODE, ForwardMode.VERIFY})


def _require_same_length(step_id: int, lengths: dict[str, int]) -> None:
    if len(set(lengths.values())) > 1:
        raise EngineError(f"step {step_id}: per-request fields differ in length: {lengths}")


@dataclass(frozen=True, slots=True, kw_only=True)
class StepInput:
    """An immutable snapshot of one step: which requests run, on which rows and at what positions.

    Every ``tuple`` field has one entry per request, in batch order. A step is either a prefill or
    a decode step; the mixed batch is not scheduled yet.

    Attributes:
        step_id: Counts up from 1; the matching :class:`StepOutput` carries the same number.
        mode: ``PREFILL``; ``DECODE`` when every request has one row; ``VERIFY`` when at least one
            decode request reserved look-ahead rows.
        request_ids: The requests, in batch order.
        batch_rows: Each request's block table row. It is the same for the whole life of a request,
            so the runner updates only what changed.
        rows_per_request: Rows the request takes: the chunk length in a prefill step, ``1 + k`` in
            a decode step. The KV has room for all of them, committed or not.
        start_positions: Position of the request's first row.
        tokens: The input token of each known row, from the first row on. A prefill chunk lists all
            its rows; a decode request lists only its pending token, and the runner supplies the
            tokens of the remaining ``rows - len(tokens)`` draft rows.
        need_logits: Whether the request needs logits; a prefill chunk that does not end the prompt
            does not, so the head is skipped.
        sampling_params: The request's frozen sampling parameters.
        table_updates: ``(group, row, start, stop)`` ranges of the block table that changed since
            the previous step and must reach the device, freed rows included.
        sequence_indices: Per request, which output sequence of its parent it is (``n > 1``); it
            selects an independent random stream. Empty means 0 for every request.
        output_lengths: Per request, the tokens it emitted before this step; ``min_tokens``
            counts from it. Empty means 0 for every request.

    Raises:
        EngineError: If the fields disagree, a row count is below 1, or the mode does not fit the
            rows.
    """

    step_id: int
    mode: ForwardMode
    request_ids: tuple[RequestId, ...]
    batch_rows: tuple[int, ...]
    rows_per_request: tuple[int, ...]
    start_positions: tuple[int, ...]
    tokens: tuple[tuple[TokenId, ...], ...]
    need_logits: tuple[bool, ...]
    sampling_params: tuple[SamplingParams, ...]
    table_updates: tuple[tuple[int, int, int, int], ...] = ()
    sequence_indices: tuple[int, ...] = ()
    output_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        lengths = {
            "request_ids": len(self.request_ids),
            "batch_rows": len(self.batch_rows),
            "rows_per_request": len(self.rows_per_request),
            "start_positions": len(self.start_positions),
            "tokens": len(self.tokens),
            "need_logits": len(self.need_logits),
            "sampling_params": len(self.sampling_params),
        }
        for name, values in (
            ("sequence_indices", self.sequence_indices),
            ("output_lengths", self.output_lengths),
        ):
            if values:
                lengths[name] = len(values)
                if min(values) < 0:
                    raise EngineError(f"step {self.step_id}: {name} holds a negative value")
        _require_same_length(self.step_id, lengths)
        if self.mode not in _STEP_MODES:
            raise EngineError(f"step {self.step_id}: a step cannot run in mode {self.mode}")
        if not self.request_ids:
            raise EngineError(f"step {self.step_id} has no requests")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise EngineError(f"step {self.step_id} lists a request twice")
        if len(set(self.batch_rows)) != len(self.batch_rows):
            raise EngineError(f"step {self.step_id} puts two requests on one batch row")
        for rows, start, known in zip(
            self.rows_per_request, self.start_positions, self.tokens, strict=True
        ):
            if rows < 1 or start < 0:
                raise EngineError(
                    f"step {self.step_id}: a request needs at least one row and a position >= 0"
                )
            if not 1 <= len(known) <= rows:
                raise EngineError(f"step {self.step_id}: {len(known)} known tokens for {rows} rows")
            if self.mode is ForwardMode.PREFILL and len(known) != rows:
                raise EngineError(f"step {self.step_id}: a prefill chunk knows all its tokens")
            if self.mode is ForwardMode.DECODE and rows != 1:
                raise EngineError(f"step {self.step_id}: a DECODE step has one row per request")

    def sequence_index(self, index: int) -> int:
        """The output sequence of the request at batch position ``index``."""
        return self.sequence_indices[index] if self.sequence_indices else 0

    def output_length(self, index: int) -> int:
        """Tokens the request at batch position ``index`` emitted before this step."""
        return self.output_lengths[index] if self.output_lengths else 0

    @property
    def num_requests(self) -> int:
        """Requests in the step."""
        return len(self.request_ids)

    @property
    def num_rows(self) -> int:
        """Total rows of the step: its token count."""
        return sum(self.rows_per_request)

    @property
    def draft_slots(self) -> tuple[int, ...]:
        """Per request, the rows whose input tokens the runner must supply (``rows - known``)."""
        return tuple(
            rows - len(known)
            for rows, known in zip(self.rows_per_request, self.tokens, strict=True)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StepOutput:
    """What the runner reports after executing a step.

    Attributes:
        step_id: The step this answers; it must be the step ``schedule()`` handed out.
        request_ids: The requests of that step, in the same order.
        tokens: Per request, the tokens the step produced: none for a prefill chunk that does not
            end the prompt, one for the chunk that does, and ``a + 1`` after a decode step in which
            ``a`` drafts were accepted. May be empty only when ``error`` is set.
        accepted: Per request, accepted drafts; empty means none anywhere.
        logprobs: Per request, one entry per produced token or ``None``; empty means no logprobs.
        releasable: Requests whose resources the runner no longer reads, so the scheduler may free
            their KV. ``None`` means every request of this step (the synchronous case). With a
            lagging runner it also lists requests that finished in an earlier step; the scheduler
            frees a finished request only once it appears here.
        error: A step-level failure; every request of the step then finishes with ``ERROR``.

    Raises:
        EngineError: If the per-request fields disagree in length.
    """

    step_id: int
    request_ids: tuple[RequestId, ...]
    tokens: tuple[tuple[TokenId, ...], ...] = ()
    accepted: tuple[int, ...] = ()
    logprobs: tuple[tuple[TokenLogprob, ...] | None, ...] = ()
    releasable: tuple[RequestId, ...] | None = None
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        count = len(self.request_ids)
        if (self.error is None and len(self.tokens) != count) or (
            self.tokens and len(self.tokens) != count
        ):
            raise EngineError(
                f"step {self.step_id}: {len(self.tokens)} token lists for {count} requests"
            )
        for name, values in (("accepted", self.accepted), ("logprobs", self.logprobs)):
            if values and len(values) != count:
                raise EngineError(
                    f"step {self.step_id}: {len(values)} {name} entries for {count} requests"
                )
