"""Where the model runs and with which compute settings.

``DeviceConfig`` is a frozen host value. It names a device and a dtype; it never
probes hardware and imports no torch, so it can be built, compared and serialized
on a machine without CUDA. Whether the named device exists and whether a backend
supports the dtype is decided later: ``ayaka.ops.backend.load_backend`` reports
every family a backend cannot serve for this device and dtype.

The defaults are constants, not a reading of the environment: the same config
means the same run on every machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

from ayaka.configs.base import ConfigMixin, validate_buckets
from ayaka.core.errors import ConfigError
from ayaka.utils.dtypes import COMPUTE_DTYPES, DType
from ayaka.utils.validation import require_bool, require_text

__all__ = ["DeviceConfig"]

_DEVICE = re.compile(r"(cpu|cuda(?::(\d+))?)")


@dataclass(frozen=True, slots=True)
class DeviceConfig(ConfigMixin):
    """Device, compute dtype, ops backends and decode-graph settings.

    Attributes:
        device: ``"cpu"``, ``"cuda"`` or ``"cuda:N"`` with ``N`` a non-negative
            index.
        dtype: Compute dtype of the model. One of
            :data:`ayaka.utils.dtypes.COMPUTE_DTYPES` (fp16, bf16, fp32). Pass a
            :class:`~ayaka.utils.dtypes.DType`; names such as ``"bf16"`` are
            parsed by the caller with ``DType.from_str``.
        backends: Ops backend names in priority order, as accepted by
            ``ayaka.ops.backend.load_backend``: each op family comes from the
            first backend that serves it. Whether a name is registered is checked
            there, not here.
        cuda_graph: Capture plain decode steps in CUDA graphs at start-up and replay them
            (``ayaka.execution.graph``); every op family then comes from a backend that declares
            it graph-safe. ``None`` (the default) means on for a CUDA device and off on the CPU,
            which :attr:`uses_cuda_graph` resolves; ``True`` requires a CUDA device. On by
            default since replaying graphs made batch-1 fp16 decode steps 2.13x (Llama-3.2-1B) to
            9.09x (SmolLM2-135M) faster than eager ones on an NVIDIA L4 (#99, a gate of 1.5x).
        graph_buckets: Decode batch sizes a graph is captured for, ascending and without
            duplicates; sizes above the engine's running requests are dropped. A decode step
            replays the smallest bucket that holds it. Validated even when ``cuda_graph`` is off.
        vision_load: When to load a multimodal vision encoder (``"eager"`` at startup or
            ``"lazy"`` on the first image request).

    Raises:
        TypeError: If a field has the wrong type (a raw string for ``dtype``, a
            non-``bool`` flag, a list for a tuple field).
        ValueError: If ``backends`` or ``graph_buckets`` is malformed.
        ConfigError: ``INVALID_DEVICE`` for an unknown device string,
            ``INVALID_CHOICE`` for a dtype outside the compute dtypes or an unknown
            ``vision_load`` mode, ``GRAPH_NEEDS_CUDA`` when ``cuda_graph`` is set on the CPU.
    """

    device: str = "cpu"
    dtype: DType = DType.FP32
    backends: tuple[str, ...] = ("torch",)
    cuda_graph: bool | None = None
    graph_buckets: tuple[int, ...] = (1, 2, 4, 8)
    vision_load: Literal["eager", "lazy"] = "eager"

    def __post_init__(self) -> None:
        require_text(self.device, "device.device")
        if _DEVICE.fullmatch(self.device) is None:
            raise ConfigError(
                "device.device",
                "INVALID_DEVICE",
                f"{self.device!r} is not 'cpu', 'cuda' or 'cuda:<index>'",
            )
        if not isinstance(cast("object", self.dtype), DType):
            raise TypeError("device.dtype must be a DType member")
        if self.dtype not in COMPUTE_DTYPES:
            allowed = [dtype.label for dtype in COMPUTE_DTYPES]
            raise ConfigError(
                "device.dtype",
                "INVALID_CHOICE",
                f"{self.dtype.label} is not a compute dtype; must be one of {allowed}",
            )
        backends = cast("object", self.backends)
        if not isinstance(backends, tuple) or not backends:
            raise ValueError("device.backends must be a non-empty tuple of names")
        for name in self.backends:
            require_text(name, "device.backends entry")
        if len(set(self.backends)) != len(self.backends):
            raise ValueError("device.backends must not repeat a name")
        if self.cuda_graph is not None:
            require_bool(self.cuda_graph, "device.cuda_graph")
        validate_buckets(self.graph_buckets)
        if self.vision_load not in ("eager", "lazy"):
            raise ConfigError(
                "device.vision_load",
                "INVALID_CHOICE",
                f"vision_load must be 'eager' or 'lazy', got {self.vision_load!r}",
            )
        if self.cuda_graph and self.device_type != "cuda":
            raise ConfigError(
                "device.cuda_graph",
                "GRAPH_NEEDS_CUDA",
                f"CUDA graphs need a CUDA device, got {self.device!r}",
            )

    @property
    def uses_cuda_graph(self) -> bool:
        """Whether decode steps run in CUDA graphs: ``cuda_graph``, or a CUDA device when unset."""
        return self.device_type == "cuda" if self.cuda_graph is None else self.cuda_graph

    @property
    def device_type(self) -> str:
        """``"cpu"`` or ``"cuda"``: the device without its index."""
        return self.device.partition(":")[0]

    @property
    def device_index(self) -> int | None:
        """The ``N`` of ``"cuda:N"``, or ``None`` when the string has no index."""
        _, _, index = self.device.partition(":")
        return int(index) if index else None
