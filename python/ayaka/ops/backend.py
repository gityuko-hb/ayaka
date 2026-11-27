"""Backend registry: which library runs each op family.

A backend is a named set of op-family implementations with a
:class:`BackendCaps` record of where they run and what they can do.
:func:`load_backend` takes backend names in priority order and returns one
:class:`OpsBackend`.
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
    from ayaka.ops.linear_attention import LinearAttentionOps
    from ayaka.ops.mla import MLAOps
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
    "linear_attention",
    "mla",
]
"""One op family: a field of :class:`OpsBackend` and a key of :class:`BackendFamilies`."""

#: Op families, in the order logs and error messages list them.
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
    "linear_attention",
    "mla",
)

#: The families a captured decode step may call.
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
    linear_attention: LinearAttentionOps
    mla: MLAOps


@dataclass(frozen=True, slots=True)
class BackendCaps:
    """What one backend implements, where it runs and what it can do."""

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
    """One implementation of every op family, as chosen by :func:`load_backend`."""

    norm: NormOps
    activation: ActivationOps
    embedding: EmbeddingOps
    linear: LinearOps
    rope: RopeOps
    kvcache: KVCacheOps
    attention: AttentionOps
    sampling: SamplingOps
    moe: MoeOps
    linear_attention: LinearAttentionOps
    mla: MLAOps
    providers: Mapping[FamilyName, str]
    graph_safe: bool = False


_registry: dict[str, tuple[BackendCaps, Callable[[], BackendFamilies]]] = {}


def register_backend(caps: BackendCaps, loader: Callable[[], BackendFamilies]) -> None:
    """Add a backend to the registry."""
    if caps.name in _registry:
        raise ValueError(f"ops backend {caps.name!r} is already registered")
    _registry[caps.name] = (caps, loader)


def available_backends() -> tuple[BackendCaps, ...]:
    """Return the capability record of every registered backend."""
    return tuple(caps for caps, _ in _registry.values())


def load_backend(
    names: Sequence[str],
    *,
    device: str,
    dtype: DType,
    cuda_graph: bool = False,
    head_dim: int | None = None,
) -> OpsBackend:
    """Build the ops for one device from backends listed in priority order."""
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
        linear_attention=cast("LinearAttentionOps", chosen["linear_attention"]),
        mla=cast("MLAOps", chosen["mla"]),
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
        linear_attention,
        mla,
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
        "linear_attention": linear_attention,
        "mla": mla,
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
        families=frozenset({"attention", "linear_attention", "mla"}),
        platforms=frozenset({"linux"}),
        devices=frozenset({"cuda"}),
        dtypes=frozenset({DType.FP16, DType.BF16}),
        graph_safe=frozenset({"attention", "linear_attention", "mla"}),
        multi_row_attention=True,
        attention_head_dims=FLASHINFER_HEAD_DIMS,
    ),
    _load_flashinfer,
)
