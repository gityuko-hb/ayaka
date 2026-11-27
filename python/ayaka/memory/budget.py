"""The memory ledger and the pure division of a device's memory.

Two things live here, deliberately apart:

* :func:`plan_budget` decides how one device's memory is split. It is a pure
  function of integers: no torch, no I/O, no device. ``ayaka plan``, engine
  start-up and a runtime resize all call this one function, so the table a user
  reads before starting is the table the engine uses.
* :class:`MemoryBudget` is the ledger that holds the result while the engine runs:
  who reserved how many bytes. It never divides anything itself.

Every pool prices itself with a plain cost record (:class:`KVCost`,
:class:`DraftCost`, :class:`ExpertCost`, :class:`StateCost`, :class:`VisionCost`),
so adding a kind of pool never changes ``plan_budget``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Final, Literal, cast

from ayaka.core.errors import CapacityError
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.validation import require_int, require_text

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "BUDGET_OWNERS",
    "BudgetInputs",
    "BudgetOwner",
    "BudgetPlan",
    "DraftCost",
    "ExpertCost",
    "KVCost",
    "MemoryBudget",
    "StateCost",
    "VisionCost",
    "plan_budget",
]

type BudgetOwner = Literal[
    "weights",
    "kv",
    "state",
    "vision",
    "expert_cache",
    "workspace",
    "graph",
    "draft",
    "runtime",
    "reserve",
]

#: The fixed set of owners a ledger knows. ``runtime`` is memory a library takes
#: on its own after warm-up; ``reserve`` is the safety margin.
BUDGET_OWNERS: Final[tuple[BudgetOwner, ...]] = (
    "weights",
    "kv",
    "state",
    "vision",
    "expert_cache",
    "workspace",
    "graph",
    "draft",
    "runtime",
    "reserve",
)


def _short(what: str, need: int, have: int) -> CapacityError:
    """The capacity error: three numbers, never just a refusal."""
    return CapacityError(
        f"{what}: need {need} bytes, have {have}, short {need - have}",
        context={"what": what, "need": need, "have": have, "short": need - have},
    )


@dataclass(frozen=True, slots=True)
class DraftCost:
    """What a speculative draft source costs, known before it is loaded."""

    name: str
    fixed_bytes: int
    bytes_per_kv_token: int = 0

    def __post_init__(self) -> None:
        require_text(self.name, "name")
        require_int(self.fixed_bytes, "fixed_bytes")
        require_int(self.bytes_per_kv_token, "bytes_per_kv_token")


@dataclass(frozen=True, slots=True)
class KVCost:
    """What one KV group costs (``KVSpec.bytes_per_block`` and ``fixed_bytes``)."""

    bytes_per_block: int
    fixed_bytes: int = 0
    block_size: int = 16

    def __post_init__(self) -> None:
        require_int(self.bytes_per_block, "bytes_per_block", minimum=1)
        require_int(self.fixed_bytes, "fixed_bytes")
        require_int(self.block_size, "block_size", minimum=1)


@dataclass(frozen=True, slots=True)
class ExpertCost:
    """What an expert cache slot costs; only a MoE model with offload has one."""

    bytes_per_slot: int
    min_slots: int
    max_slots: int

    def __post_init__(self) -> None:
        require_int(self.bytes_per_slot, "bytes_per_slot", minimum=1)
        require_int(self.min_slots, "min_slots")
        require_int(self.max_slots, "max_slots")
        if self.max_slots < self.min_slots:
            raise ValueError("max_slots must be >= min_slots")


@dataclass(frozen=True, slots=True)
class StateCost:
    """What per-request recurrent state slots cost on hybrid linear-attention models.

    Attributes:
        bytes_per_slot: Recurrent SSM + conv1d bytes for one request slot across
            all linear-attention layers.
        max_slots: Total slots allocated (including padding slot 0).
    """

    bytes_per_slot: int
    max_slots: int

    def __post_init__(self) -> None:
        require_int(self.bytes_per_slot, "bytes_per_slot", minimum=0)
        require_int(self.max_slots, "max_slots", minimum=2)

    @property
    def total_bytes(self) -> int:
        """Total recurrent state pool bytes."""
        return self.bytes_per_slot * self.max_slots


@dataclass(frozen=True, slots=True)
class VisionCost:
    """What a multimodal vision tower costs in weights and image activations.

    Attributes:
        weights_bytes: Vision encoder + projector parameter bytes when loaded
            eagerly (or 0 when deferred/lazy).
        activation_bytes: Peak activation scratch bytes for encoding a target
            high-resolution image (e.g. ``1024x768``).
    """

    weights_bytes: int = 0
    activation_bytes: int = 0

    def __post_init__(self) -> None:
        require_int(self.weights_bytes, "weights_bytes", minimum=0)
        require_int(self.activation_bytes, "activation_bytes", minimum=0)


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetInputs:
    """Everything :func:`plan_budget` needs. All sizes are integers in bytes."""

    baseline_bytes: int
    memory_ratio: float
    weights_bytes: int
    kv: tuple[KVCost, ...]
    kv_reserve_tokens: int
    late_alloc_bytes: int = 0
    drafts: tuple[DraftCost, ...] = ()
    kv_blocks_override: int | None = None
    expert: ExpertCost | None = None
    state: StateCost | None = None
    vision: VisionCost | None = None
    graph_bytes_estimate: int = 0
    activation_bytes_per_token: int = 0
    activation_fixed_bytes: int = 0
    safety_bytes: int = 0

    def __post_init__(self) -> None:
        require_int(self.baseline_bytes, "baseline_bytes")
        require_int(self.late_alloc_bytes, "late_alloc_bytes")
        require_int(self.weights_bytes, "weights_bytes")
        require_int(self.kv_reserve_tokens, "kv_reserve_tokens", minimum=1)
        require_int(self.graph_bytes_estimate, "graph_bytes_estimate")
        require_int(self.activation_bytes_per_token, "activation_bytes_per_token")
        require_int(self.activation_fixed_bytes, "activation_fixed_bytes")
        require_int(self.safety_bytes, "safety_bytes")
        ratio = cast("object", self.memory_ratio)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("memory_ratio must be a number")
        if not (math.isfinite(ratio) and 0 < ratio <= 1):
            raise ValueError("memory_ratio must be in (0, 1]")
        if not self.kv:
            raise ValueError("kv must hold at least one group")
        if len({group.block_size for group in self.kv}) != 1:
            raise ValueError("every KV group must use the same block_size")
        if self.kv_blocks_override is not None:
            require_int(self.kv_blocks_override, "kv_blocks_override", minimum=2)


@dataclass(frozen=True, slots=True)
class BudgetPlan:
    """How one device's memory is divided."""

    usable_bytes: int
    headroom_bytes: int
    kv_blocks: tuple[int, ...]
    kv_tokens: int
    expert_slots: int
    max_chunk_tokens: int
    max_running: int
    loaded_drafts: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    kv_bytes: int
    draft_bytes: int
    expert_bytes: int
    graph_bytes: int
    state_bytes: int = 0
    vision_bytes: int = 0


def _kv_cost(unit: int, draft_per_block: int, blocks: int) -> int:
    """Bytes of ``blocks`` blocks in every group plus the drafts' KV for the usable ones."""
    return blocks * unit + (blocks - 1) * draft_per_block


def _usable(baseline: int, ratio: float) -> int:
    """``floor(ratio * baseline)`` exactly, reading the ratio as the decimal it was written as."""
    return int(Fraction(repr(float(ratio))) * baseline)


def plan_budget(inputs: BudgetInputs) -> BudgetPlan:
    """Divide a device's memory between weights, state, vision, KV, experts and headroom."""
    usable = _usable(inputs.baseline_bytes, inputs.memory_ratio)
    headroom = inputs.baseline_bytes - usable
    block_size = inputs.kv[0].block_size
    unit = sum(group.bytes_per_block for group in inputs.kv)
    kv_fixed = sum(group.fixed_bytes for group in inputs.kv)
    state_bytes = inputs.state.total_bytes if inputs.state is not None else 0
    vision_weights = inputs.vision.weights_bytes if inputs.vision is not None else 0
    vision_act = inputs.vision.activation_bytes if inputs.vision is not None else 0
    expert = inputs.expert
    expert_min_bytes = expert.min_slots * expert.bytes_per_slot if expert else 0
    floor_blocks = div_ceil(inputs.kv_reserve_tokens, block_size) + 1

    def kv_need(draft_per_token: int, blocks: int) -> int:
        return _kv_cost(unit, draft_per_token * block_size, blocks)

    base_fixed = (
        inputs.weights_bytes + inputs.late_alloc_bytes + kv_fixed + state_bytes + vision_weights
    )
    if base_fixed > usable:
        raise _short("weights and fixed memory", base_fixed, usable)

    loaded: list[DraftCost] = []
    skipped: list[tuple[str, str]] = []
    draft_fixed = 0
    draft_per_token = 0
    for draft in inputs.drafts:
        fixed = base_fixed + draft_fixed + draft.fixed_bytes
        per_token = draft_per_token + draft.bytes_per_kv_token
        need = fixed + expert_min_bytes + kv_need(per_token, floor_blocks)
        if need > usable:
            skipped.append((draft.name, f"needs {need} bytes with the KV floor, have {usable}"))
            continue
        loaded.append(draft)
        draft_fixed += draft.fixed_bytes
        draft_per_token += draft.bytes_per_kv_token

    override = inputs.kv_blocks_override
    if override is not None and override < floor_blocks:
        raise _short(
            "KV blocks fixed by the user",
            kv_need(draft_per_token, floor_blocks),
            kv_need(draft_per_token, override),
        )
    pool = usable - base_fixed - draft_fixed
    kv_min = kv_need(draft_per_token, override or floor_blocks)
    if pool < kv_min + expert_min_bytes:
        what = "KV blocks fixed by the user" if override else "KV floor"
        raise _short(what, base_fixed + draft_fixed + kv_min + expert_min_bytes, usable)

    expert_slots = 0
    if expert is not None:
        slots = (pool - kv_min) // expert.bytes_per_slot
        expert_slots = min(max(slots, expert.min_slots), expert.max_slots)
    expert_bytes = expert_slots * (expert.bytes_per_slot if expert else 0)

    kv_pool = pool - expert_bytes
    if override:
        blocks = override
    else:
        per_block_draft = draft_per_token * block_size
        blocks = (kv_pool + per_block_draft) // (unit + per_block_draft)
    kv_tokens = (blocks - 1) * block_size
    kv_bytes = blocks * unit + kv_fixed
    draft_bytes = draft_fixed + draft_per_token * kv_tokens

    chunk_room = headroom - inputs.safety_bytes - inputs.activation_fixed_bytes - vision_act
    graph_bytes = inputs.graph_bytes_estimate
    max_chunk = _chunk_limit(chunk_room - graph_bytes, inputs.activation_bytes_per_token, kv_tokens)
    if max_chunk < block_size and graph_bytes > 0:
        retry = _chunk_limit(chunk_room, inputs.activation_bytes_per_token, kv_tokens)
        if retry >= block_size:
            skipped.append(
                (
                    "graph",
                    f"headroom {headroom} leaves no room for a {block_size}-token chunk "
                    f"beside {graph_bytes} bytes of graphs",
                )
            )
            graph_bytes = 0
            max_chunk = retry
    if max_chunk < block_size:
        need = (
            inputs.safety_bytes
            + inputs.activation_fixed_bytes
            + vision_act
            + graph_bytes
            + block_size * inputs.activation_bytes_per_token
        )
        raise _short("headroom for one prefill block", need, headroom)

    per_request_blocks = div_ceil(inputs.kv_reserve_tokens, block_size)
    max_running = (blocks - 1) // per_request_blocks
    if inputs.state is not None:
        max_running = min(max_running, inputs.state.max_slots - 1)
    return BudgetPlan(
        usable_bytes=usable,
        headroom_bytes=headroom,
        kv_blocks=(blocks,) * len(inputs.kv),
        kv_tokens=kv_tokens,
        expert_slots=expert_slots,
        max_chunk_tokens=max_chunk,
        max_running=max_running,
        loaded_drafts=tuple(draft.name for draft in loaded),
        skipped=tuple(skipped),
        kv_bytes=kv_bytes,
        draft_bytes=draft_bytes,
        expert_bytes=expert_bytes,
        graph_bytes=graph_bytes,
        state_bytes=state_bytes,
        vision_bytes=vision_weights + vision_act,
    )


def _chunk_limit(room: int, bytes_per_token: int, kv_tokens: int) -> int:
    """Largest chunk that fits ``room``; a chunk can never exceed one request's KV."""
    if room < 0:
        return 0
    if bytes_per_token == 0:
        return kv_tokens
    return min(room // bytes_per_token, kv_tokens)


class MemoryBudget:
    """Who holds how many bytes of one device, while the engine runs."""

    def __init__(self, capacity_bytes: int) -> None:
        self._capacity = require_int(capacity_bytes, "capacity_bytes")
        self._held: dict[BudgetOwner, int] = dict.fromkeys(BUDGET_OWNERS, 0)

    @property
    def capacity_bytes(self) -> int:
        """Total bytes that may be reserved."""
        return self._capacity

    @property
    def reserved_bytes(self) -> int:
        """Bytes reserved by every owner."""
        return sum(self._held.values())

    @property
    def free_bytes(self) -> int:
        """Bytes still available to reserve."""
        return self._capacity - self.reserved_bytes

    def held(self, owner: BudgetOwner) -> int:
        """Bytes ``owner`` holds."""
        return self._held[self._check_owner(owner)]

    def reserve(self, owner: BudgetOwner, nbytes: int) -> None:
        """Charge ``nbytes`` to ``owner``."""
        key = self._check_owner(owner)
        require_int(nbytes, "nbytes")
        if nbytes > self.free_bytes:
            raise _short(f"reserve {key}", nbytes, self.free_bytes)
        self._held[key] += nbytes

    def release(self, owner: BudgetOwner, nbytes: int) -> None:
        """Return ``nbytes`` from ``owner``."""
        key = self._check_owner(owner)
        require_int(nbytes, "nbytes")
        if nbytes > self._held[key]:
            raise ValueError(
                f"cannot release {nbytes} bytes from {key}: it holds {self._held[key]}"
            )
        self._held[key] -= nbytes

    def snapshot(self) -> Mapping[BudgetOwner, int]:
        """A copy of every owner's bytes, including owners that hold zero."""
        return dict(self._held)

    @staticmethod
    def _check_owner(owner: BudgetOwner) -> BudgetOwner:
        if owner not in BUDGET_OWNERS:
            raise ValueError(f"unknown owner {owner!r}; expected one of {BUDGET_OWNERS}")
        return owner
