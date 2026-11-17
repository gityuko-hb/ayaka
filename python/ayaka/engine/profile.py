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
``workspace_bytes``         what the allocator holds after the warm-up beyond weights
                            and KV: activations, library workspaces, rounding
``prefill_seconds``         that prefill
``decode_seconds``          one eager decode step per bucket
==========================  =========================================================

The result is saved under the cache directory, keyed by the device, the model,
the dtype, the backends and the library versions, and read back by the next
start-up, which plans with the measured numbers instead of zeros
(:func:`load_profile`). :func:`reconcile` then compares the ledger the engine
reserved with what the device reports. Since
``runtime_bytes`` is what the allocator does not hold, the check is in effect
that the allocator holds what the ledger's owners say (weights, KV, the
activation peak, the graphs), plus its own rounding.

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
from ayaka.utils.math_utils import div_ceil

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ayaka.device.context import Device, MemoryReading
    from ayaka.execution.graph import CapturedBucket
    from ayaka.memory.budget import MemoryBudget
    from ayaka.models.base import CausalLM
    from ayaka.ops.kvcache import KVBuffers

__all__ = [
    "PROFILE_VERSION",
    "ProfileKey",
    "Reconciliation",
    "StartupProfile",
    "StartupReport",
    "load_profile",
    "measure_startup",
    "profile_directory",
    "reconcile",
    "save_profile",
]

logger = logging.getLogger(__name__)

#: Bumped when the meaning of a field changes, so an old file is measured again, not misread.
PROFILE_VERSION: Final = 1
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
        workspace_bytes: What the allocator holds after the warm-up beyond weights and KV: the
            activations, the matrix libraries' workspaces and the allocator's rounding.
        reused_bytes: Weights and KV that landed in memory the process's allocator held before
            the baseline was read; 0 in a fresh process. The device never lost those bytes.
        graph_bytes: Memory the captured graphs took; 0 without graphs.
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
    """The ledger against the device, after start-up.

    Attributes:
        ledger_bytes: What the engine reserved in its :class:`~ayaka.memory.budget.MemoryBudget`.
        measured_bytes: What the device lost since the baseline was read.
        tolerance_bytes: The largest difference that still counts as a match.
    """

    ledger_bytes: int
    measured_bytes: int
    tolerance_bytes: int

    @property
    def difference_bytes(self) -> int:
        """``measured - ledger``: positive when the device lost more than the ledger says."""
        return self.measured_bytes - self.ledger_bytes

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
    tolerance_ratio: float = 0.05,
    tolerance_floor: int = 64 * 2**20,
) -> Reconciliation:
    """Compare the ledger with the device: ``baseline - free now`` against the reserved bytes.

    ``reused_bytes`` (:attr:`StartupProfile.reused_bytes`) is added to what the device lost:
    tensors of this engine that sit in memory the process held before the baseline.

    The tolerance is the larger of ``tolerance_ratio`` of the measured bytes and
    ``tolerance_floor``: the allocator rounds every block up and keeps a cache, and other
    processes on the device move ``free`` a little.
    """
    measured = max(baseline_bytes - reading.free, 0) + reused_bytes
    tolerance = max(int(measured * tolerance_ratio), tolerance_floor)
    return Reconciliation(
        ledger_bytes=budget.reserved_bytes, measured_bytes=measured, tolerance_bytes=tolerance
    )


def measure_startup(
    model: CausalLM,
    kv_buffers: Mapping[int, KVBuffers],
    *,
    key: ProfileKey,
    device: Device,
    baseline_bytes: int,
    baseline_reserved_bytes: int = 0,
    kv_bytes: int,
    chunk_tokens: int,
    buckets: Sequence[int],
    block_size: int,
    usable_blocks: int,
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

    Raises:
        EngineError: If the pool has too few blocks for the dummy prefill.
    """
    import torch

    chunk = max(
        1, min(chunk_tokens, usable_blocks * block_size, model.config.max_position_embeddings)
    )
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    target = device.torch_device
    attention = model.ops.attention

    def step(rows: list[int], lengths: list[int], tables: list[list[int]]) -> None:
        width = max(len(table) for table in tables)
        block_rows = torch.tensor(
            [table + [0] * (width - len(table)) for table in tables],
            dtype=torch.int32,
            device=target,
        )
        meta = attention.plan(
            ForwardMode.PREFILL if max(rows) > 1 else ForwardMode.DECODE,
            torch.tensor(rows, dtype=torch.int32, device=target),
            torch.tensor(lengths, dtype=torch.int32, device=target),
            block_rows,
            block_size,
        )
        tokens = torch.zeros(sum(rows), dtype=torch.int64, device=target)
        hidden = model(tokens, ForwardContext(attention=meta, kv_buffers=kv_buffers))
        last = torch.tensor(
            [sum(rows[: i + 1]) - 1 for i in range(len(rows))], dtype=torch.int64, device=target
        )
        model.compute_logits(hidden.index_select(0, last))

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
    # After the warm-up: everything the device lost since the baseline that this engine's
    # allocator does not hold. The CUDA context, kernel modules loaded on first use, library
    # handles and, under WSL2, the gap between what NVML (the baseline) and CUDA report free.
    reading = device.memory()
    reserved = reading.reserved - baseline_reserved_bytes
    runtime = max(baseline_bytes - reading.free - reserved, 0)
    # In a process that held GPU memory before this engine (an earlier engine, a test), part of the
    # weights and the cache can land in memory the allocator already held: the device did not lose
    # it, so it is counted here and added back when the ledger is compared with the device.
    reused = max(weights + kv_bytes - reserved, 0)
    # What the allocator holds beyond the weights and the cache: the step's activations, the
    # matrix libraries' workspaces and the allocator's own rounding.
    workspace = max(reserved + reused - weights - kv_bytes, max(peak, 0))
    profile = StartupProfile(
        key=key,
        weights_bytes=weights,
        kv_bytes=kv_bytes,
        runtime_bytes=runtime,
        activation_tokens=chunk,
        activation_peak_bytes=max(peak, 0),
        workspace_bytes=workspace,
        reused_bytes=reused,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode,
    )
    logger.info(
        "startup run: %d-token prefill peaked %d bytes above its inputs in %.3fs; the allocator "
        "holds %d bytes beyond weights and KV; runtime outside the allocator %d bytes; "
        "eager decode %s",
        chunk,
        profile.activation_peak_bytes,
        prefill_seconds,
        workspace,
        runtime,
        {bucket: f"{seconds * 1e3:.2f}ms" for bucket, seconds in decode.items()},
    )
    return profile
