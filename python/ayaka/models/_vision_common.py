"""Shared checkpoint helpers and buffer initialization for multimodal vision towers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, cast

import torch
from safetensors import safe_open
from torch import nn

from ayaka.core.errors import ModelLoadError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from contextlib import AbstractContextManager
    from pathlib import Path

    from ayaka.core.multimodal import ImageInput, VisionSpec

__all__ = [
    "VisionLifecycle",
    "VisionTowerProtocol",
    "count_module_tensor_bytes",
    "load_prefixed_weights",
]


class VisionTowerProtocol(Protocol):
    @property
    def is_loaded(self) -> bool: ...
    def weight_bytes(self) -> int: ...
    def activation_bytes(self, height: int = 768, width: int = 1024) -> int: ...
    def load_from_safetensors(
        self, directory: Path, *, device: torch.device | str | None, dtype: torch.dtype
    ) -> None: ...
    def encode_images(
        self, images: Sequence[ImageInput], spec: VisionSpec
    ) -> tuple[torch.Tensor, ...]: ...


class VisionLifecycle:
    """Lifecycle container keeping vision weights detached from ``nn.Module`` children."""

    def __init__(
        self,
        tower: VisionTowerProtocol | None,
        spec: VisionSpec | None,
        *,
        dtype: torch.dtype,
        device_fn: Callable[[], torch.device],
    ) -> None:
        self._tower = tower
        self._spec = spec
        self._dtype = dtype
        self._device_fn = device_fn
        self._directory: Path | None = None

    @property
    def has_tower(self) -> bool:
        return self._tower is not None and self._spec is not None

    @property
    def is_loaded(self) -> bool:
        return self._tower is not None and self._tower.is_loaded

    def weight_bytes(self) -> int:
        return self._tower.weight_bytes() if self._tower is not None else 0

    def activation_bytes(self) -> int:
        return self._tower.activation_bytes() if self._tower is not None else 0

    def attach(self, directory: Path, *, lazy: bool) -> None:
        self._directory = directory
        if not lazy and self.has_tower:
            self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._tower is None or self._spec is None:
            raise ModelLoadError("vision tower is not configured for this model")
        if self._tower.is_loaded:
            return
        if self._directory is None:
            raise ModelLoadError("vision checkpoint directory is not attached")
        self._tower.load_from_safetensors(
            self._directory, device=self._device_fn(), dtype=self._dtype
        )

    def encode(self, images: Sequence[ImageInput]) -> tuple[torch.Tensor, ...]:
        self.ensure_loaded()
        if self._tower is None or self._spec is None:
            raise ModelLoadError("vision tower unavailable")
        return self._tower.encode_images(images, self._spec)

    def release(self) -> None:
        self._tower = None
        self._directory = None


class _SafeHandle(Protocol):
    def keys(self) -> Iterable[str]: ...
    def get_tensor(self, name: str) -> torch.Tensor: ...


def count_module_tensor_bytes(module: nn.Module, element_bytes: int) -> int:
    """Return total byte size of all parameters and persistent buffers on ``module``."""
    total = 0
    for param in module.parameters():
        total += param.numel() * element_bytes
    for name, buf in module.named_buffers():
        del name
        total += buf.numel() * element_bytes
    return total


def _safetensors_files(directory: Path, key_prefixes: tuple[str, ...]) -> list[Path]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        payload = cast("dict[str, object]", json.loads(index_path.read_text(encoding="utf-8")))
        weight_map = cast("dict[str, str]", payload["weight_map"])
        needed: list[Path] = []
        seen: set[str] = set()
        for key, filename in weight_map.items():
            if any(key.startswith(prefix) for prefix in key_prefixes) and filename not in seen:
                seen.add(filename)
                needed.append(directory / filename)
        return needed
    single = directory / "model.safetensors"
    if single.is_file():
        return [single]
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise ModelLoadError(f"no safetensors files found in {directory}")
    return shards


def load_prefixed_weights(
    module: nn.Module,
    directory: Path,
    *,
    prefixes: tuple[str, ...],
    strip_prefixes: tuple[str, ...],
    device: torch.device | str | None,
    dtype: torch.dtype,
    reset_buffers: nn.Module | None = None,
) -> None:
    """Load parameters and persistent buffers matching ``prefixes`` into ``module``.

    If ``module`` currently lives on the ``meta`` device, it is materialized on
    ``device`` first, non-persistent buffers are initialized via ``reset_buffers``
    if provided, and every checkpoint tensor is copied with no grad tracking.
    """
    target_device = torch.device(device or "cpu")
    first_tensor = next(module.parameters(), None)
    if first_tensor is not None and first_tensor.device.type == "meta":
        module.to_empty(device=target_device)

    expected_params = dict(module.named_parameters())
    expected_buffers = {
        name: buf for name, buf in module.named_buffers() if not name.endswith(".inv_freq")
    }
    loaded: set[str] = set()

    open_st = cast("Callable[[Path, str, str], AbstractContextManager[_SafeHandle]]", safe_open)
    for path in _safetensors_files(directory, prefixes):
        with open_st(path, "pt", "cpu") as handle:
            for key in handle.keys():  # noqa: SIM118  # safetensors handle is not a dict
                if not any(key.startswith(prefix) for prefix in prefixes):
                    continue
                local = key
                for strip in strip_prefixes:
                    if local.startswith(strip):
                        local = local[len(strip) :]
                        break
                target: torch.Tensor | None = expected_params.get(local)
                if target is None:
                    target = expected_buffers.get(local)
                if target is None:
                    continue
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != tuple(target.shape):
                    raise ModelLoadError(
                        f"vision weight {key} has shape {list(tensor.shape)}, "
                        f"expected {list(target.shape)}"
                    )
                target.data.copy_(tensor.to(device=target_device, dtype=dtype))
                loaded.add(local)

    missing = (set(expected_params) | set(expected_buffers)) - loaded
    if missing:
        sample = ", ".join(sorted(missing)[:6])
        raise ModelLoadError(
            f"vision checkpoint in {directory} missing {len(missing)} tensors: {sample}"
        )

    if reset_buffers is not None and hasattr(reset_buffers, "reset_cache"):
        reset_buffers.reset_cache(target_device)  # type: ignore[operator]  # structural helper call
