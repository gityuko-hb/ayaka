"""Backend registry: which library runs each op family.

A backend is a named set of op-family implementations with a
:class:`BackendCaps` record of where they run and what they can do.
:func:`load_backend` takes backend names in priority order and returns one
:class:`OpsBackend`: for every family, the first listed backend that implements
it, supports the requested device and dtype, and has every capability the
engine asked for (running inside a CUDA graph, several query rows per request).
The choice is made once, when the engine is built, and it is logged; no op
switches to another backend during a call, and nothing is checked again on the
hot path.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

from ayaka.core.errors import BackendError, CapabilityError, ConfigError
from ayaka.ops.backends.flashinfer import HEAD_DIMS as FLASHINFER_HEAD_DIMS
from ayaka.utils.dtypes import COMPUTE_DTYPES, DType
from ayaka.utils.torch_utils import require_torch
from ayaka.utils.validation import require_text

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ayaka.ops.activation import ActivationOps
    from ayaka.ops.attention import AttentionOps
    from ayaka.ops.embedding import EmbeddingOps
    from ayaka.ops.kvcache import KVCacheOps
    from ayaka.ops.linear import LinearOps
    from ayaka.ops.moe import MoeOps
    from ayaka.ops.norm import NormOps
    from ayaka.ops.rope import RopeOps
    from ayaka.ops.sampling import SamplingOps

__all__ = [
    "GRAPH_FAMILIES",
    "BackendCaps",
    "BackendFamilies",
    "FamilyName",
    "OpsBackend",
    "available_backends",
    "load_backend",
    "register_backend",
]

logger = logging.getLogger(__name__)

type FamilyName = Literal[
    "norm",
    "activation",
    "embedding",
    "linear",
    "rope",
    "kvcache",
    "attention",
    "sampling",
    "moe",
]
"""One op family: a field of :class:`OpsBackend` and a key of :class:`BackendFamilies`."""

#: Op families, in the order logs and error messages list them. One field of
#: :class:`OpsBackend` and one key of :class:`BackendFamilies` per entry.
_FAMILIES: Final[tuple[FamilyName, ...]] = (
    "norm",
    "activation",
    "embedding",
    "linear",
    "rope",
    "kvcache",
    "attention",
    "sampling",
    "moe",
)

#: The families a captured decode step calls: the model's forward, the cache write,
#: attention over graph metadata and the sampler. With ``cuda_graph=True``,
#: :func:`load_backend` takes each of them only from a backend that lists it in
#: :attr:`BackendCaps.graph_safe`.
GRAPH_FAMILIES: Final[frozenset[FamilyName]] = frozenset(_FAMILIES)

_CONFIG_PATH: Final = "ops.backends"
_NOT_IMPLEMENTED: Final = "does not implement this family"


class BackendFamilies(TypedDict, total=False):
    """The implementations one backend supplies, keyed by family name."""

    norm: NormOps
    activation: ActivationOps
    embedding: EmbeddingOps
    linear: LinearOps
    rope: RopeOps
    kvcache: KVCacheOps
    attention: AttentionOps
    sampling: SamplingOps
    moe: MoeOps


@dataclass(frozen=True, slots=True)
class BackendCaps:
    """What one backend implements, where it runs and what it can do.

    Attributes:
        name: Registry name, as written in the list given to :func:`load_backend`.
        families: Op families the backend implements.
        platforms: ``sys.platform`` values it runs on.
        devices: Torch device types it runs on (``"cpu"``, ``"cuda"``).
        dtypes: Compute dtypes its ops accept.
        graph_safe: Families whose ops can be captured in a CUDA graph and replayed:
            they read nothing back to the host, allocate nothing that outlives the
            call and, for attention, run on the fixed metadata of
            :meth:`~ayaka.ops.attention.AttentionOps.graph_metadata`. A subset of
            ``families``.
        external_uniform: The sampling family draws its token from the random
            number it is given (:meth:`~ayaka.ops.sampling.SamplingOps.sample`'s
            ``u``) rather than from a generator of its own. Every sampling family
            must; a backend whose library cannot does not implement ``sampling``.
        multi_row_attention: The attention family serves several query rows per
            request (prefill chunks, verify windows), not only decode rows. Every
            engine prefills, so :func:`load_backend` takes attention only from a
            backend that has it.
        attention_head_dims: The head widths the attention family has kernels for, or
            ``None`` for any. Given a model's ``head_dim``, :func:`load_backend` takes
            attention from a backend that has it.

    Raises:
        ValueError: If ``name`` is blank, ``families`` names an unknown family,
            ``graph_safe`` names a family the backend does not implement, or a
            backend implements ``sampling`` without ``external_uniform``.
    """

    name: str
    families: frozenset[FamilyName]
    platforms: frozenset[str]
    devices: frozenset[str]
    dtypes: frozenset[DType]
    graph_safe: frozenset[FamilyName] = frozenset()
    external_uniform: bool = False
    multi_row_attention: bool = False
    attention_head_dims: frozenset[int] | None = None

    def __post_init__(self) -> None:
        require_text(self.name, "name")
        unknown = sorted(self.families.difference(_FAMILIES))
        if unknown:
            raise ValueError(f"unknown op families {unknown}; known families: {list(_FAMILIES)}")
        outside = sorted(self.graph_safe.difference(self.families))
        if outside:
            raise ValueError(f"graph_safe names families the backend does not implement: {outside}")
        if "sampling" in self.families and not self.external_uniform:
            raise ValueError(
                "a sampling family must take its random numbers from the caller "
                "(external_uniform=True); see ayaka.ops.sampling"
            )


@dataclass(frozen=True, slots=True)
class OpsBackend:
    """One implementation of every op family, as chosen by :func:`load_backend`.

    Layers receive this record, or the one family they use, when they are built.
    ``providers`` maps each family name to the backend that supplies it.
    ``graph_safe`` is true when it was built with ``cuda_graph=True``, so every
    family can be captured in a CUDA graph.
    """

    norm: NormOps
    activation: ActivationOps
    embedding: EmbeddingOps
    linear: LinearOps
    rope: RopeOps
    kvcache: KVCacheOps
    attention: AttentionOps
    sampling: SamplingOps
    moe: MoeOps
    providers: Mapping[FamilyName, str]
    graph_safe: bool = False


_registry: dict[str, tuple[BackendCaps, Callable[[], BackendFamilies]]] = {}


def register_backend(caps: BackendCaps, loader: Callable[[], BackendFamilies]) -> None:
    """Add a backend to the registry.

    Args:
        caps: What the backend implements and where it runs.
        loader: Returns an implementation for every family in ``caps.families``.
            :func:`load_backend` calls it only when the backend is about to be
            used, so this is where the backend imports its library. Raising
            :class:`~ayaka.core.errors.CapabilityError` from it marks the
            backend as unavailable on this machine.

    Raises:
        ValueError: If a backend named ``caps.name`` is already registered.
    """
    if caps.name in _registry:
        raise ValueError(f"ops backend {caps.name!r} is already registered")
    _registry[caps.name] = (caps, loader)


def available_backends() -> tuple[BackendCaps, ...]:
    """Return the capability record of every registered backend.

    Registered is not the same as installed: whether a backend's library can be
    imported is only known once :func:`load_backend` needs it.
    """
    return tuple(caps for caps, _ in _registry.values())


def load_backend(
    names: Sequence[str],
    *,
    device: str,
    dtype: DType,
    cuda_graph: bool = False,
    head_dim: int | None = None,
) -> OpsBackend:
    """Build the ops for one device from backends listed in priority order.

    Each family comes from the first listed backend that implements it, runs on
    this platform and ``device``, accepts ``dtype``, has the capabilities the
    family needs and can be loaded. A backend is loaded only when it is the best
    candidate for some family. Attention always needs
    :attr:`BackendCaps.multi_row_attention`; with ``cuda_graph`` every family of
    :data:`GRAPH_FAMILIES` must also be in the backend's
    :attr:`BackendCaps.graph_safe`.

    Args:
        names: Registered backend names, most preferred first.
        device: Torch device type the ops will run on: ``"cpu"`` or ``"cuda"``.
        dtype: Compute dtype of the model.
        cuda_graph: Whether decode steps will be captured in CUDA graphs.
        head_dim: The model's attention head width, so that attention comes from a
            backend with kernels for it; ``None`` when no model is known yet.

    Raises:
        ConfigError: If ``names`` is empty, repeats a name or contains an
            unregistered one, or if some family is left without a backend. In
            the last case the message names every such family and why each
            listed backend was rejected for it.
        BackendError: If a loader does not return a family its record declares.
    """
    _check_names(names)
    loaded: dict[str, Mapping[str, object]] = {}
    unavailable: dict[str, str] = {}
    chosen: dict[FamilyName, object] = {}
    providers: dict[FamilyName, str] = {}
    missing: dict[FamilyName, tuple[str, ...]] = {}

    for family in _FAMILIES:
        rejections: list[str] = []
        passed_over: list[str] = []
        for name in names:
            reason = _rejection(
                _registry[name][0], family, device, dtype, cuda_graph=cuda_graph, head_dim=head_dim
            )
            if reason is None:
                reason = _ensure_loaded(name, loaded, unavailable)
            if reason is None:
                chosen[family] = loaded[name][family]
                providers[family] = name
                break
            rejections.append(f"{name} {reason}")
            if reason != _NOT_IMPLEMENTED:
                passed_over.append(rejections[-1])
        else:
            missing[family] = tuple(rejections)
            continue
        for rejection in passed_over:
            logger.info("ops family %r: %s; using %s", family, rejection, providers[family])

    if missing:
        details = "".join(
            f"\n  {family}: {'; '.join(rejections)}" for family, rejections in missing.items()
        )
        graph = " inside a CUDA graph" if cuda_graph else ""
        raise ConfigError(
            _CONFIG_PATH,
            "BACKEND_CAPABILITY_MISSING",
            f"no backend in {list(names)} provides {', '.join(missing)} "
            f"for device {device!r} and dtype {dtype.label}{graph}:{details}",
            context={"missing": missing},
        )

    logger.info(
        "ops backends for %s %s%s: %s",
        device,
        dtype.label,
        " (cuda graph)" if cuda_graph else "",
        ", ".join(f"{family}={name}" for family, name in providers.items()),
    )
    return OpsBackend(
        norm=cast("NormOps", chosen["norm"]),
        activation=cast("ActivationOps", chosen["activation"]),
        embedding=cast("EmbeddingOps", chosen["embedding"]),
        linear=cast("LinearOps", chosen["linear"]),
        rope=cast("RopeOps", chosen["rope"]),
        kvcache=cast("KVCacheOps", chosen["kvcache"]),
        attention=cast("AttentionOps", chosen["attention"]),
        sampling=cast("SamplingOps", chosen["sampling"]),
        moe=cast("MoeOps", chosen["moe"]),
        providers=MappingProxyType(providers),
        graph_safe=cuda_graph,
    )


def _check_names(names: Sequence[str]) -> None:
    if isinstance(names, str):
        raise TypeError("names must be a sequence of backend names, not one string")
    if not names:
        raise ConfigError(_CONFIG_PATH, "BACKEND_LIST_EMPTY", "at least one backend must be listed")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ConfigError(_CONFIG_PATH, "BACKEND_DUPLICATE", f"listed more than once: {repeated}")
    unknown = [name for name in names if name not in _registry]
    if unknown:
        raise ConfigError(
            _CONFIG_PATH,
            "BACKEND_UNKNOWN",
            f"not registered: {unknown}; registered backends: {sorted(_registry)}",
        )


def _rejection(
    caps: BackendCaps,
    family: FamilyName,
    device: str,
    dtype: DType,
    *,
    cuda_graph: bool,
    head_dim: int | None,
) -> str | None:
    """Return why ``caps`` cannot serve ``family`` here, or ``None`` if it can."""
    if family not in caps.families:
        return _NOT_IMPLEMENTED
    problems: list[str] = []
    if sys.platform not in caps.platforms:
        problems.append(f"does not run on {sys.platform}")
    if device not in caps.devices:
        problems.append(f"has no {device} support")
    if dtype not in caps.dtypes:
        problems.append(f"has no {dtype.label} support")
    if family == "attention" and not caps.multi_row_attention:
        problems.append("serves only one query row per request")
    heads = caps.attention_head_dims
    if (
        family == "attention"
        and head_dim is not None
        and heads is not None
        and head_dim not in heads
    ):
        problems.append(f"has no kernels for heads of {head_dim} (only {sorted(heads)})")
    if cuda_graph and family in GRAPH_FAMILIES and family not in caps.graph_safe:
        problems.append("cannot run inside a CUDA graph")
    return ", ".join(problems) or None


def _ensure_loaded(
    name: str, loaded: dict[str, Mapping[str, object]], unavailable: dict[str, str]
) -> str | None:
    """Load ``name`` on first use; return why it is unavailable, or ``None``."""
    if name not in loaded and name not in unavailable:
        caps, loader = _registry[name]
        try:
            families = cast("Mapping[str, object]", loader())
        except CapabilityError as exc:
            unavailable[name] = f"is unavailable ({exc.detail}; {exc.remedy})"
            logger.warning("ops backend %r %s", name, unavailable[name])
        else:
            absent = sorted(caps.families - families.keys())
            if absent:
                raise BackendError(
                    f"ops backend {name!r} declares {absent} but its loader did not return them",
                    context={"backend": name, "families": absent},
                )
            loaded[name] = families
    return unavailable.get(name)


def _load_torch() -> BackendFamilies:
    require_torch(
        capability="ops.torch",
        remedy="install PyTorch with `uv sync --extra cpu` (or `--extra cu130` for CUDA)",
    )
    from ayaka.ops.backends.torch import (
        activation,
        attention,
        embedding,
        kvcache,
        linear,
        moe,
        norm,
        rope,
        sampling,
    )

    return {
        "norm": norm,
        "activation": activation,
        "embedding": embedding,
        "linear": linear,
        "rope": rope,
        "kvcache": kvcache,
        "attention": attention,
        "sampling": sampling,
        "moe": moe,
    }


def _load_flashinfer() -> BackendFamilies:
    from ayaka.ops.backends.flashinfer import load

    return load()


register_backend(
    BackendCaps(
        name="torch",
        families=frozenset(_FAMILIES),
        platforms=frozenset({"linux", "win32", "darwin"}),
        devices=frozenset({"cpu", "cuda"}),
        dtypes=frozenset(COMPUTE_DTYPES),
        graph_safe=frozenset(_FAMILIES),
        external_uniform=True,
        multi_row_attention=True,
    ),
    _load_torch,
)

register_backend(
    BackendCaps(
        name="flashinfer",
        families=frozenset({"attention"}),
        platforms=frozenset({"linux"}),
        devices=frozenset({"cuda"}),
        dtypes=frozenset({DType.FP16, DType.BF16}),
        graph_safe=frozenset({"attention"}),
        multi_row_attention=True,
        attention_head_dims=FLASHINFER_HEAD_DIMS,
    ),
    _load_flashinfer,
)
