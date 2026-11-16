"""Backend registry: which library runs each op family.

A backend is a named set of op-family implementations with a
:class:`BackendCaps` record of where they run. :func:`load_backend` takes
backend names in priority order and returns one :class:`OpsBackend`: for every
family, the first listed backend that implements it and supports the requested
device and dtype. The choice is made once, when the engine is built, and it is
logged; no op switches to another backend during a call.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypedDict, cast

from ayaka.core.errors import BackendError, CapabilityError, ConfigError
from ayaka.utils.dtypes import COMPUTE_DTYPES
from ayaka.utils.torch_utils import require_torch
from ayaka.utils.validation import require_text

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ayaka.ops.activation import ActivationOps
    from ayaka.ops.embedding import EmbeddingOps
    from ayaka.ops.linear import LinearOps
    from ayaka.ops.norm import NormOps
    from ayaka.utils.dtypes import DType

__all__ = [
    "BackendCaps",
    "BackendFamilies",
    "OpsBackend",
    "available_backends",
    "load_backend",
    "register_backend",
]

logger = logging.getLogger(__name__)

#: Op families, in the order logs and error messages list them. One field of
#: :class:`OpsBackend` and one key of :class:`BackendFamilies` per entry.
_FAMILIES: Final = ("norm", "activation", "embedding", "linear")

_CONFIG_PATH: Final = "ops.backends"
_NOT_IMPLEMENTED: Final = "does not implement this family"


class BackendFamilies(TypedDict, total=False):
    """The implementations one backend supplies, keyed by family name."""

    norm: NormOps
    activation: ActivationOps
    embedding: EmbeddingOps
    linear: LinearOps


@dataclass(frozen=True, slots=True)
class BackendCaps:
    """What one backend implements and where it runs.

    Attributes:
        name: Registry name, as written in the list given to :func:`load_backend`.
        families: Op families the backend implements.
        platforms: ``sys.platform`` values it runs on.
        devices: Torch device types it runs on (``"cpu"``, ``"cuda"``).
        dtypes: Compute dtypes its ops accept.

    Raises:
        ValueError: If ``name`` is blank or ``families`` names an unknown family.
    """

    name: str
    families: frozenset[str]
    platforms: frozenset[str]
    devices: frozenset[str]
    dtypes: frozenset[DType]

    def __post_init__(self) -> None:
        require_text(self.name, "name")
        unknown = sorted(self.families.difference(_FAMILIES))
        if unknown:
            raise ValueError(f"unknown op families {unknown}; known families: {list(_FAMILIES)}")


@dataclass(frozen=True, slots=True)
class OpsBackend:
    """One implementation of every op family, as chosen by :func:`load_backend`.

    Layers receive this record, or the one family they use, when they are built.
    ``providers`` maps each family name to the backend that supplies it.
    """

    norm: NormOps
    activation: ActivationOps
    embedding: EmbeddingOps
    linear: LinearOps
    providers: Mapping[str, str]


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


def load_backend(names: Sequence[str], *, device: str, dtype: DType) -> OpsBackend:
    """Build the ops for one device from backends listed in priority order.

    Each family comes from the first listed backend that implements it, runs on
    this platform and ``device``, accepts ``dtype`` and can be loaded. A backend
    is loaded only when it is the best candidate for some family.

    Args:
        names: Registered backend names, most preferred first.
        device: Torch device type the ops will run on: ``"cpu"`` or ``"cuda"``.
        dtype: Compute dtype of the model.

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
    chosen: dict[str, object] = {}
    providers: dict[str, str] = {}
    missing: dict[str, tuple[str, ...]] = {}

    for family in _FAMILIES:
        rejections: list[str] = []
        passed_over: list[str] = []
        for name in names:
            reason = _rejection(_registry[name][0], family, device, dtype)
            if reason is None:
                reason = _ensure_loaded(name, loaded, unavailable)
            if reason is None:
                chosen[family] = loaded[name][family]
                providers[family] = name
                break
            rejections.append(f"{name} {reason}")
            if reason is not _NOT_IMPLEMENTED:
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
        raise ConfigError(
            _CONFIG_PATH,
            "BACKEND_CAPABILITY_MISSING",
            f"no backend in {list(names)} provides {', '.join(missing)} "
            f"for device {device!r} and dtype {dtype.label}:{details}",
            context={"missing": missing},
        )

    logger.info(
        "ops backends for %s %s: %s",
        device,
        dtype.label,
        ", ".join(f"{family}={name}" for family, name in providers.items()),
    )
    return OpsBackend(
        norm=cast("NormOps", chosen["norm"]),
        activation=cast("ActivationOps", chosen["activation"]),
        embedding=cast("EmbeddingOps", chosen["embedding"]),
        linear=cast("LinearOps", chosen["linear"]),
        providers=MappingProxyType(providers),
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


def _rejection(caps: BackendCaps, family: str, device: str, dtype: DType) -> str | None:
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
    from ayaka.ops.backends.torch import activation, embedding, linear, norm

    return {"norm": norm, "activation": activation, "embedding": embedding, "linear": linear}


register_backend(
    BackendCaps(
        name="torch",
        families=frozenset(_FAMILIES),
        platforms=frozenset({"linux", "win32", "darwin"}),
        devices=frozenset({"cpu", "cuda"}),
        dtypes=frozenset(COMPUTE_DTYPES),
    ),
    _load_torch,
)
