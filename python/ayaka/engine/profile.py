"""The start-up run: measure what the memory plan could only estimate, and keep the numbers.

Before the engine takes requests on a GPU, :func:`measure_startup` runs the model
once at the largest prefill chunk and once per decode bucket, on dummy tokens
written into cache blocks that no request holds yet, and reads the device:

==========================  =========================================================
``weights_bytes``           the parameters, as allocated
``kv_bytes``                the KV pools, as allocated
``runtime_bytes``           memory the device lost that torch's allocator does not
                            hold, read after the warm-up: CUDA context, kernel modules
                            loaded on first use, library handles
``activation_peak_bytes``   the peak the allocator reached during the prefill, above
                            what it held before: scratch of the largest step
``workspace_bytes``         what the allocator holds after the warm-up, its cache of
                            freed scratch released, beyond weights and KV: library
                            workspaces, rounding; at least the activation peak
``prefill_seconds``         that prefill
``decode_seconds``          one eager decode step per bucket
==========================  =========================================================

The result is saved under the cache directory, keyed by the device, the model,
the dtype, the backends and the library versions, and read back by the next
start-up, which plans with the measured numbers instead of zeros
(:func:`load_profile`). :func:`reconcile` then compares the ledger the engine
reserved with what the allocator holds: the bytes it reserved since the baseline
plus ``reused_bytes``, and the ``runtime_bytes`` the ledger records outside the
allocator. The driver's free memory is kept as a diagnostic only, because
another process on the device can move it without the ledger changing.

The dummy steps write keys and values into blocks that are free: a request that
gets one of them later writes its own keys and values before it reads any, and
the positions it has not written are never read.

This module imports torch inside the functions that run the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from ayaka.__version__ import __version__
from ayaka.core.errors import EngineError
from ayaka.core.types import ForwardMode
from ayaka.layers.context import ForwardContext
from ayaka.ops.linear_attention import LinearAttentionMetadata
from ayaka.utils.math_utils import div_ceil

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import torch

    from ayaka.device.context import Device, MemoryReading
    from ayaka.execution.graph import CapturedBucket
    from ayaka.memory.budget import MemoryBudget
    from ayaka.models.base import CausalLM
    from ayaka.ops.kvcache import KVBuffers, LatentKVBuffers
    from ayaka.ops.linear_attention import RecurrentStateBuffers

__all__ = [
    "PROFILE_VERSION",
    "ProfileKey",
    "Reconciliation",
    "StartupProfile",
    "StartupReport",
    "load_profile",
    "measure_startup",
    "profile_directory",
    "profile_save_issues",
    "reconcile",
    "save_profile",
    "should_save_profile",
]

logger = logging.getLogger(__name__)

#: Bumped when the meaning of a field changes, so an old file is measured again, not misread.
PROFILE_VERSION: Final = 2
_CACHE_VARIABLE: Final = "AYAKA_CACHE_DIR"


@dataclass(frozen=True, slots=True)
class ProfileKey:
    """Everything a measurement depends on; change any of it and the profile is measured again.

    Attributes:
        device: The device's name and architecture.
        model: The fingerprint of the model's ``ModelConfig``.
        dtype: The compute dtype.
        providers: ``family=backend`` pairs of the ops, sorted.
        block_size: Tokens per KV block.
        versions: Ayaka's and torch's versions, and any kernel library's.
    """

    device: str
    model: str
    dtype: str
    providers: str
    block_size: int
    versions: str

    def digest(self) -> str:
        """A short stable name for the file of this key."""
        text = json.dumps([PROFILE_VERSION, *asdict(self).values()], sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def build(
        cls,
        *,
        device: str,
        model: str,
        dtype: str,
        providers: Iterable[tuple[str, str]],
        block_size: int,
        libraries: Mapping[str, str] | None = None,
    ) -> ProfileKey:
        """A key from the parts, with Ayaka's version and the given libraries' versions.

        ``providers`` is ``(family, backend)`` pairs, such as ``OpsBackend.providers.items()``.
        """
        versions = {"ayaka": __version__, **(libraries or {})}
        return cls(
            device=device,
            model=model,
            dtype=dtype,
            providers=",".join(f"{family}={name}" for family, name in sorted(providers)),
            block_size=block_size,
            versions=",".join(f"{name}={version}" for name, version in sorted(versions.items())),
        )


@dataclass(frozen=True, slots=True)
class StartupProfile:
    """What one start-up measured. Sizes in bytes, times in seconds.

    Attributes:
        key: What the numbers depend on.
        weights_bytes: Parameters as allocated.
        kv_bytes: KV pools as allocated.
        runtime_bytes: Device memory taken outside the allocator.
        activation_tokens: Rows of the measured prefill step.
        activation_peak_bytes: Allocator peak during that step, above what it held before.
        workspace_bytes: What the allocator holds after the warm-up, once its cache of freed
            scratch is released, beyond weights and KV: the matrix libraries' workspaces and the
            allocator's rounding; never less than ``activation_peak_bytes``.
        reused_bytes: Weights and KV that landed in memory the process's allocator held before
            the baseline was read; 0 in a fresh process. The device never lost those bytes.
        graph_bytes: Memory the captured graphs took; 0 without graphs.
        sampling_peak_bytes: Allocator peak while sampling the logits of the widest sampling
            step, above what it held before: every row of the widest verify window with a
            drafter, one row per request without. The allocator gives it back after the step,
            so the ledger keeps it as ``reserve``, memory left free for that step.
        prefill_seconds: Wall time of the measured prefill step, device work included.
        decode_seconds: Bucket size to the wall time of one eager decode step.
    """

    key: ProfileKey
    weights_bytes: int
    kv_bytes: int
    runtime_bytes: int
    activation_tokens: int
    activation_peak_bytes: int
    workspace_bytes: int = 0
    reused_bytes: int = 0
    graph_bytes: int = 0
    sampling_peak_bytes: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: Mapping[int, float] = field(default_factory=lambda: {})

    @property
    def activation_bytes_per_token(self) -> int:
        """Activation peak per prefill row, rounded up: what ``plan_budget`` takes."""
        return div_ceil(self.activation_peak_bytes, max(self.activation_tokens, 1))

    def to_json(self) -> str:
        """The profile as a JSON document."""
        payload = asdict(self)
        payload["decode_seconds"] = {str(k): v for k, v in self.decode_seconds.items()}
        payload["version"] = PROFILE_VERSION
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> StartupProfile:
        """Read :meth:`to_json`'s output back.

        Raises:
            ValueError: If the document is not a profile of this :data:`PROFILE_VERSION`.
        """
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("a startup profile is a JSON object")
        data = cast("dict[str, Any]", raw)
        if data.pop("version", None) != PROFILE_VERSION:
            raise ValueError(f"not a startup profile of version {PROFILE_VERSION}")
        key = ProfileKey(**cast("dict[str, Any]", data.pop("key")))
        decode = {
            int(k): float(v) for k, v in cast("dict[str, Any]", data.pop("decode_seconds")).items()
        }
        return cls(key=key, decode_seconds=decode, **data)


def profile_directory() -> Path:
    """Where profiles are kept: ``$AYAKA_CACHE_DIR/profiles``, else ``~/.cache/ayaka/profiles``.

    ``XDG_CACHE_HOME`` replaces ``~/.cache`` when it is set.
    """
    root = os.environ.get(_CACHE_VARIABLE)
    if root:
        return Path(root).expanduser() / "profiles"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ayaka" / "profiles"


def load_profile(key: ProfileKey, directory: Path | None = None) -> StartupProfile | None:
    """The profile saved for ``key``, or ``None`` if there is none or it cannot be read.

    A file that does not parse, or whose key differs, is ignored with a log line: it only
    makes this start-up plan without measurements.
    """
    path = (directory or profile_directory()) / f"{key.digest()}.json"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        profile = StartupProfile.from_json(text)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("ignoring the startup profile %s: %s", path, exc)
        return None
    if profile.key != key:
        logger.warning("ignoring the startup profile %s: it was measured for another key", path)
        return None
    return profile


def save_profile(profile: StartupProfile, directory: Path | None = None) -> Path | None:
    """Write ``profile`` for the next start-up; return the path, or ``None`` if it cannot.

    The file is written to a temporary name and renamed, so a reader never sees half of it.
    A failure to write is logged, not raised: the engine runs without a cache directory.
    """
    folder = directory or profile_directory()
    path = folder / f"{profile.key.digest()}.json"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(profile.to_json(), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        logger.warning("cannot save the startup profile to %s: %s", path, exc)
        return None
    return path


@dataclass(frozen=True, slots=True)
class StartupReport:
    """What the start-up run of a GPU engine found and reserved.

    Attributes:
        profile: The measurements, as saved for the next start-up.
        budget: The ledger: weights, KV, runtime, workspace and graph bytes.
        reconciliation: The ledger against the device.
        graphs: What each captured decode bucket cost; empty without graphs.
    """

    profile: StartupProfile
    budget: MemoryBudget
    reconciliation: Reconciliation
    graphs: tuple[CapturedBucket, ...] = ()


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """The ledger against the allocator, after start-up.

    Attributes:
        ledger_bytes: What the engine reserved in its :class:`~ayaka.memory.budget.MemoryBudget`.
        measured_bytes: What the allocator reserved since the baseline plus ``reused_bytes``, and
            the ledger's ``runtime`` bytes, which the allocator does not hold.
        tolerance_bytes: The largest difference that still counts as a match.
        device_bytes: What the driver reports the device lost since the baseline. Diagnostic only:
            another process on the device can return or take memory between the readings.
    """

    ledger_bytes: int
    measured_bytes: int
    tolerance_bytes: int
    device_bytes: int = 0

    @property
    def difference_bytes(self) -> int:
        """``measured - ledger``: positive when the device holds more than the ledger says."""
        return self.measured_bytes - self.ledger_bytes

    @property
    def device_difference_bytes(self) -> int:
        """``device - ledger``: the driver's view, kept out of :attr:`matches`."""
        return self.device_bytes - self.ledger_bytes

    @property
    def matches(self) -> bool:
        """Whether the two agree within the tolerance."""
        return abs(self.difference_bytes) <= self.tolerance_bytes


def reconcile(
    budget: MemoryBudget,
    *,
    baseline_bytes: int,
    reading: MemoryReading,
    reused_bytes: int = 0,
    baseline_reserved_bytes: int = 0,
    tolerance_ratio: float = 0.05,
    tolerance_floor: int = 64 * 2**20,
) -> Reconciliation:
    """Compare the ledger with what the allocator holds, not with free device memory.

    ``baseline_reserved_bytes`` is what the allocator already held at the baseline; only what it
    reserved since is this engine's. ``reused_bytes`` (:attr:`StartupProfile.reused_bytes`) adds
    tensors that landed in that pre-baseline reservation. The ledger's ``runtime`` owner is added
    back because it is memory outside the allocator, and its ``reserve`` owner because it is
    memory kept free for the widest sampling step, which the allocator holds only during it.

    ``baseline_bytes`` is read before anything was allocated and becomes
    :attr:`Reconciliation.device_bytes`, the driver's diagnostic view. It does not decide the match:
    another process can return or take memory between the two readings, and under WSL2 the driver
    and CUDA can report different free values.

    The tolerance is the larger of ``tolerance_ratio`` of the measured bytes and
    ``tolerance_floor``: the allocator rounds every block up and keeps a cache.
    """
    allocator = max(reading.reserved - baseline_reserved_bytes, 0) + reused_bytes
    measured = allocator + budget.held("runtime") + budget.held("reserve")
    tolerance = max(int(measured * tolerance_ratio), tolerance_floor)
    return Reconciliation(
        ledger_bytes=budget.reserved_bytes,
        measured_bytes=measured,
        tolerance_bytes=tolerance,
        device_bytes=max(baseline_bytes - reading.free, 0),
    )


def profile_save_issues(
    reconciliation: Reconciliation,
    reading: MemoryReading,
) -> tuple[str, ...]:
    """Reasons a start-up measurement is not trustworthy enough to save for future plans.

    Returns an empty tuple when the measurement reflects a clean fit on the device:

    * ``reconciliation.matches``: the allocator and runtime bytes agree with the ledger;
    * ``reading.free > 0``: the device was not exhausted or paged out (for example under WDDM,
      where ``cudaMemGetInfo`` reads ``0`` free and clamps ``runtime_bytes`` below reality).

    Pre-baseline allocator reuse (``StartupProfile.reused_bytes > 0``) is intentionally allowed:
    :func:`measure_startup` and :func:`reconcile` already account for it, and shared processes
    (such as sequential engines or tests) routinely retain pre-baseline segments.
    """
    issues: list[str] = []
    if not reconciliation.matches:
        issues.append(
            f"memory reconciliation mismatch of {reconciliation.difference_bytes} bytes "
            f"(tolerance {reconciliation.tolerance_bytes})"
        )
    if reading.free <= 0:
        issues.append(f"device reports {reading.free} free bytes after startup")
    return tuple(issues)


def should_save_profile(
    reconciliation: Reconciliation,
    reading: MemoryReading,
) -> bool:
    """Whether the start-up measurement is clean enough to save for future plans."""
    return not profile_save_issues(reconciliation, reading)


def measure_startup(
    model: CausalLM,
    kv_buffers: Mapping[int, KVBuffers],
    *,
    key: ProfileKey,
    device: Device,
    baseline_bytes: int,
    baseline_reserved_bytes: int = 0,
    kv_bytes: int,
    extra_weights_bytes: int = 0,
    chunk_tokens: int,
    buckets: Sequence[int],
    block_size: int,
    usable_blocks: int,
    latent_kv_buffers: Mapping[int, LatentKVBuffers] | None = None,
    state_buffers: Mapping[int, RecurrentStateBuffers] | None = None,
    group_layers: Sequence[tuple[int, ...]] = (),
    state_bytes: int = 0,
    sampled_rows: int = 0,
    draw: Callable[[torch.Tensor], None] | None = None,
) -> StartupProfile:
    """Run the dummy prefill and decode steps and read the device.

    Args:
        model: The loaded model.
        kv_buffers: Its cache tensors, by layer id.
        key: What the profile is for.
        device: The model's device.
        baseline_bytes: Free device memory before anything was allocated.
        baseline_reserved_bytes: What torch's allocator already held when the baseline was read
            (0 in a fresh process). Only what it reserved since is this engine's.
        kv_bytes: Bytes of the KV pools.
        chunk_tokens: Rows of the prefill step to measure: the scheduler's largest chunk.
        buckets: Requests per decode step to time.
        block_size: Tokens per block.
        usable_blocks: Blocks the pool can hand out (block 0 excluded); the dummy steps use
            the first of them.
        sampled_rows: Most logit rows one step samples: every row of the widest verify window
            when a drafter runs, else one per request.
        draw: Samples ``[rows, vocab]`` logits as the engine's sampler does at its costliest
            (``ModelRunner.draw_for_measurement``). With ``sampled_rows`` above one, the peak of
            that draw becomes ``sampling_peak_bytes``: the prefill samples one row, a verify step
            every row of its windows.

    Raises:
        EngineError: If the pool has too few blocks for the dummy prefill.
    """
    import torch

    chunk = max(1, min(chunk_tokens, usable_blocks * block_size, model.config.context_length))
    weights = sum(p.numel() * p.element_size() for p in model.parameters()) + (
        model.vision_weight_bytes() if model.vision_loaded() else 0
    )
    target = device.torch_device
    attention = model.ops.attention

    latent_map = dict(latent_kv_buffers or {})
    state_map = dict(state_buffers or {})
    num_groups = max(len(group_layers), 1)

    def step(rows: list[int], lengths: list[int], tables: list[list[int]]) -> torch.Tensor:
        width = max(len(table) for table in tables)
        block_rows = torch.tensor(
            [table + [0] * (width - len(table)) for table in tables],
            dtype=torch.int32,
            device=target,
        )
        mode = ForwardMode.PREFILL if max(rows) > 1 else ForwardMode.DECODE
        rows_t = torch.tensor(rows, dtype=torch.int32, device=target)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device=target)
        metas = tuple(
            attention.plan(mode, rows_t, lengths_t, block_rows, block_size)
            for _ in range(num_groups)
        )
        layer_attention = {
            layer_id: meta
            for meta, layer_ids in zip(metas, group_layers, strict=False)
            for layer_id in layer_ids
        }
        lin_meta = None
        if state_map:
            cu_list = [0]
            for r in rows:
                cu_list.append(cu_list[-1] + r)
            lin_meta = LinearAttentionMetadata(
                is_prefill=mode is ForwardMode.PREFILL,
                cu_seqlens=torch.tensor(cu_list, dtype=torch.int32, device=target),
                state_slots=torch.zeros(len(rows), dtype=torch.int32, device=target),
                rows_per_request=rows_t,
            )
        tokens = torch.zeros(sum(rows), dtype=torch.int64, device=target)
        hidden = model(
            tokens,
            ForwardContext(
                attention=metas[0],
                kv_buffers=kv_buffers,
                layer_attention=layer_attention,
                latent_kv_buffers=latent_map,
                state_buffers=state_map,
                linear_attention=lin_meta,
            ),
        )
        last = torch.tensor(
            [sum(rows[: i + 1]) - 1 for i in range(len(rows))], dtype=torch.int64, device=target
        )
        model.compute_logits(hidden.index_select(0, last))
        return hidden

    needed = div_ceil(chunk, block_size)
    if needed > usable_blocks:
        raise EngineError(f"the startup run needs {needed} blocks, the pool has {usable_blocks}")
    with torch.inference_mode():
        step([chunk], [chunk], [list(range(1, needed + 1))])  # warm-up: lazy set-up happens here
        device.synchronize()
        before = device.memory().allocated
        device.reset_peak()
        started = time.monotonic()
        step([chunk], [chunk], [list(range(1, needed + 1))])
        device.synchronize()
        prefill_seconds = time.monotonic() - started
        peak = device.memory().peak_allocated - before

        draw_peak = 0
        if draw is not None and sampled_rows > 1:
            hidden = step([chunk], [chunk], [list(range(1, needed + 1))])
            rows = torch.arange(sampled_rows, device=target) % chunk  # rows repeat if chunk < rows
            device.synchronize()
            before_draw = device.memory().allocated
            device.reset_peak()
            draw(model.compute_logits(hidden.index_select(0, rows)))
            device.synchronize()
            draw_peak = device.memory().peak_allocated - before_draw
            del hidden

        decode: dict[int, float] = {}
        for bucket in buckets:
            count = min(bucket, usable_blocks)
            tables = [[1 + i] for i in range(count)]
            step([1] * count, [1] * count, tables)
            device.synchronize()
            started = time.monotonic()
            step([1] * count, [1] * count, tables)
            device.synchronize()
            decode[bucket] = time.monotonic() - started
    # The steps' scratch is free now but still cached. Capturing CUDA graphs, which comes next,
    # gives that cache back to the driver (torch.cuda.graph empties the cache when it starts), so
    # it is given back here too: the reading below and the reconciliation after the capture then
    # see the allocator in the same state. The peak above already counts the scratch.
    device.release_cache()
    # After the warm-up: everything the device lost since the baseline that this engine's
    # allocator does not hold. The CUDA context, kernel modules loaded on first use, library
    # handles and, under WSL2, the gap between what NVML (the baseline) and CUDA report free.
    reading = device.memory()
    reserved = reading.reserved - baseline_reserved_bytes
    runtime = max(baseline_bytes - reading.free - reserved, 0)
    # In a process that held GPU memory before this engine (an earlier engine, a test), part of the
    # weights and the cache can land in memory the allocator already held: the device did not lose
    # it, so it is counted here and added back when the ledger is compared with the device.
    cache_total = kv_bytes + state_bytes
    reused = max(weights + extra_weights_bytes + cache_total - reserved, 0)
    # What the allocator holds beyond the weights and the cache: the step's activations, the
    # matrix libraries' workspaces and the allocator's own rounding.
    workspace = max(reserved + reused - weights - extra_weights_bytes - cache_total, max(peak, 0))
    profile = StartupProfile(
        key=key,
        weights_bytes=weights,
        kv_bytes=kv_bytes,
        runtime_bytes=runtime,
        activation_tokens=chunk,
        activation_peak_bytes=max(peak, 0),
        workspace_bytes=workspace,
        reused_bytes=reused,
        sampling_peak_bytes=max(draw_peak, 0),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode,
    )
    logger.info(
        "startup run: %d-token prefill peaked %d bytes above its inputs in %.3fs; sampling %d "
        "rows peaked %d bytes; the allocator holds %d bytes beyond weights and KV; runtime "
        "outside the allocator %d bytes; eager decode %s",
        chunk,
        profile.activation_peak_bytes,
        prefill_seconds,
        sampled_rows,
        draw_peak,
        workspace,
        runtime,
        {bucket: f"{seconds * 1e3:.2f}ms" for bucket, seconds in decode.items()},
    )
    return profile
