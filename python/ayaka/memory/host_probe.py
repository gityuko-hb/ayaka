"""Runtime host pinned-memory allocation and H2D/D2H bandwidth probe."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ayaka.utils.torch_memory import empty_host_tensor
from ayaka.utils.torch_utils import require_torch

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["PinningProbeResult", "probe_host_pinning"]

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PinningProbeResult:
    """Outcome of probing pinned host allocation and device transfer for one buffer size."""

    size_mb: int
    pinned_ok: bool
    pageable_fallback: bool
    alloc_ms: float
    h2d_gbps: float | None
    d2h_gbps: float | None
    error: str | None = None


def probe_host_pinning(
    sizes_mb: Sequence[int] = (64, 256),
    *,
    allow_pageable: bool = True,
    device: Any = None,
) -> tuple[PinningProbeResult, ...]:
    """Probe host pinned allocation ceiling and H2D/D2H bandwidth when CUDA is present."""
    torch = require_torch(capability="host_probe")
    results: list[PinningProbeResult] = []
    has_cuda = bool(torch.cuda.is_available())
    target_device = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda", 0) if has_cuda else None)
    )
    use_cuda = target_device is not None and target_device.type == "cuda" and has_cuda

    for size_mb in sizes_mb:
        num_elements = (int(size_mb) * _MIB) // 2  # float16 = 2 bytes
        started = time.perf_counter()
        try:
            host_tensor, actually_pinned = empty_host_tensor(
                (num_elements,),
                torch.float16,
                pinned=True,
                allow_pageable=allow_pageable,
            )
            host_tensor.zero_()
            alloc_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            alloc_ms = (time.perf_counter() - started) * 1000.0
            results.append(
                PinningProbeResult(
                    size_mb=int(size_mb),
                    pinned_ok=False,
                    pageable_fallback=False,
                    alloc_ms=alloc_ms,
                    h2d_gbps=None,
                    d2h_gbps=None,
                    error=str(exc),
                )
            )
            continue

        h2d_gbps: float | None = None
        d2h_gbps: float | None = None
        error: str | None = None
        if use_cuda and target_device is not None:
            try:
                dev_tensor = torch.empty((num_elements,), dtype=torch.float16, device=target_device)
                nbytes = float(int(size_mb) * _MIB)
                torch.cuda.synchronize(target_device)
                t0 = time.perf_counter()
                dev_tensor.copy_(host_tensor, non_blocking=True)
                torch.cuda.synchronize(target_device)
                t1 = time.perf_counter()
                host_tensor.copy_(dev_tensor, non_blocking=True)
                torch.cuda.synchronize(target_device)
                t2 = time.perf_counter()
                h2d_gbps = (nbytes / max(t1 - t0, 1e-9)) / 1e9
                d2h_gbps = (nbytes / max(t2 - t1, 1e-9)) / 1e9
                del dev_tensor
            except Exception as exc:
                error = str(exc)
        del host_tensor
        results.append(
            PinningProbeResult(
                size_mb=int(size_mb),
                pinned_ok=actually_pinned,
                pageable_fallback=not actually_pinned,
                alloc_ms=alloc_ms,
                h2d_gbps=h2d_gbps,
                d2h_gbps=d2h_gbps,
                error=error,
            )
        )
    return tuple(results)
