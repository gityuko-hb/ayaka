"""From a source to a model that is ready to run: build, check the checkpoint, stream the weights.

::

    resolve_source          model name -> directory                       (source.py)
    config.json             -> ModelConfig -> the model class             (ayaka.models.registry)
    build on "meta"         the architecture, with no memory behind it
    bind                    every checkpoint tensor against every parameter, headers only
    materialise             memory on the device, tied parameters tied again
    stream                  one tensor at a time, in file order: map, convert, copy into place

The checkpoint is checked against the model **before** any tensor is read and before any
device memory is allocated, so a wrong checkpoint fails in moments, and the error lists
everything that is wrong at once: parameters with no tensor, tensors no parameter claims,
parameters whose stored shape differs, parameters claimed twice, and stored dtypes that
cannot be converted. "Unexpected" tensors are what the engine does not model; "missing"
ones are what the checkpoint does not have. Either way nothing is skipped silently.

The bytes are mapped, not read: each tensor is copied from the mapped file straight into
its parameter, converted to the compute dtype by that copy, so the process never holds a
second copy of the model.

The model, not the loader, knows its checkpoint's layout (names, transposed weights, later
fused projections): see :mod:`ayaka.models.base`.
"""

from __future__ import annotations

import contextlib
import logging
import time
import warnings
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import torch

from ayaka.configs.model import ModelConfig
from ayaka.configs.model_source import RequestFormat
from ayaka.core.errors import ModelLoadError
from ayaka.model_loader.source import resolve_source
from ayaka.models.registry import resolve_model
from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType
from ayaka.utils.parsing import parse_json_strict
from ayaka.utils.torch_utils import resolve_device
from ayaka.weights.manifest import WeightManifest
from ayaka.weights.safetensors import SafetensorsFile

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.configs.device import DeviceConfig
    from ayaka.configs.model_source import ModelSourceConfig
    from ayaka.models.base import CausalLM
    from ayaka.ops.backend import OpsBackend
    from ayaka.weights.tensor import TensorRef

__all__ = ["load_model", "load_weights"]

logger = logging.getLogger(__name__)

_CONFIG_FILE: Final = "config.json"
_CONFIG_LIMIT: Final = 16 * 2**20
#: Stored dtypes that a copy converts correctly. Quantised and 8-bit floating formats carry
#: scales that the copy would drop, so they are refused, not misread.
_PLAIN_FLOATS: Final = (DType.FP64, DType.FP32, DType.FP16, DType.BF16)
_UNWRITABLE: Final = "The given buffer is not writable"


def load_model(
    source: ModelSourceConfig, device: DeviceConfig, *, ops: OpsBackend | None = None
) -> CausalLM:
    """Build the model a source describes and load its weights.

    Steps: resolve the source to a directory; read ``config.json`` and look its
    architecture up in the registry; build the model on the meta device; check the
    checkpoint against it; allocate on ``device``; tie shared parameters; stream the
    tensors in, converted to ``device.dtype``. The model is returned in evaluation mode.
    The architecture it was built from is ``model.config`` and its ops are ``model.ops``.

    Args:
        source: Where the model is. Only safetensors checkpoints are read: ``format`` may be
            ``AUTO`` or ``SAFETENSORS``.
        device: Where to put it and which compute dtype and ops backends to use.
        ops: The ops backend the model calls. Pass the one the rest of the engine uses so
            the model, the KV layout and the attention plan agree; ``None`` builds it from
            ``device.backends``.

    Returns:
        The model, with every parameter materialised on the device and filled.

    Raises:
        ModelLoadError: If the format is not one this build reads, the source cannot be
            resolved, ``config.json`` is missing or invalid, the architecture is not
            registered, a file is corrupt, or the checkpoint does not match the model.
            The last lists every mismatch (see :func:`load_weights`).
        ConfigError: If ``device`` names a backend that cannot serve this device and dtype.
        CapabilityError: If torch, the requested GPU or ``huggingface_hub`` (for a repo id)
            is not available.
    """
    _require_readable_format(source)
    started = time.monotonic()
    directory = resolve_source(source)
    config = _read_model_config(directory)
    model_class = resolve_model(config.architecture)
    backend = ops or load_backend(device.backends, device=device.device_type, dtype=device.dtype)
    target = resolve_device(device.device, capability="model_loader")

    model = model_class(config, ops=backend, dtype=device.dtype, device="meta")
    bound = _bind(model, directory)
    model.to_empty(device=target)
    model.tie_weights()
    model.initialize_buffers()
    _stream(model, bound)
    model.eval()
    logger.info(
        "loaded %s from %s onto %s as %s in %.1fs",
        config.architecture,
        directory,
        target,
        device.dtype.label,
        time.monotonic() - started,
    )
    return model


def load_weights(model: CausalLM, directory: str | Path) -> None:
    """Fill a built model's parameters from the safetensors checkpoint in ``directory``.

    The model must already own memory (not be on the meta device) and have its shared
    parameters tied. The checkpoint is checked against the model first; nothing is read
    unless it matches.

    Raises:
        ModelLoadError: If ``directory`` holds no readable checkpoint (a corrupt or truncated
            file, an index naming absent shards), or the checkpoint does not match the model.
            One error lists, in full: ``missing`` (parameters with no tensor), ``unexpected``
            (tensors no parameter claims), ``wrong_shape``, ``ambiguous`` (parameters claimed
            by several tensors) and ``unsupported_dtype``. The same lists are in
            ``error.context``.
    """
    _stream(model, _bind(model, Path(directory)))


def _require_readable_format(source: ModelSourceConfig) -> None:
    if source.format in (RequestFormat.AUTO, RequestFormat.SAFETENSORS):
        return
    reason = (
        "pickle checkpoints are never read, because loading one can run arbitrary code"
        if source.format is RequestFormat.PT
        else "random weights are not implemented"
    )
    raise ModelLoadError(
        f"format {source.format.value!r} is not supported: {reason}; use a safetensors checkpoint",
        context={"format": source.format.value},
    )


def _read_model_config(directory: Path) -> ModelConfig:
    path = directory / _CONFIG_FILE
    try:
        with path.open("rb") as handle:
            raw = handle.read(_CONFIG_LIMIT + 1)
    except OSError as exc:
        raise ModelLoadError(
            f"{directory}: cannot read {_CONFIG_FILE}: {exc}", context={"path": str(path)}
        ) from exc
    if len(raw) > _CONFIG_LIMIT:
        raise ModelLoadError(
            f"{path}: larger than {_CONFIG_LIMIT} bytes", context={"path": str(path)}
        )
    try:
        document = parse_json_strict(raw.decode("utf-8"))
    except ValueError as exc:  # not UTF-8, not JSON, or a repeated key
        raise ModelLoadError(f"{path}: not valid JSON: {exc}", context={"path": str(path)}) from exc
    if not isinstance(document, dict):
        raise ModelLoadError(
            f"{path}: expected a JSON object, got {type(document).__name__}",
            context={"path": str(path)},
        )
    return ModelConfig.from_hf_config(cast("Mapping[str, object]", document))


def _bind(model: CausalLM, directory: Path) -> Mapping[str, TensorRef]:
    """Match the checkpoint to the model from headers alone; raise one error for every problem."""
    manifest = WeightManifest.from_directory(directory)
    bound = type(model).weight_mapping(model.config).bind(manifest, model.checkpoint_specs())
    unsupported = {
        name: ref.dtype.label for name, ref in bound.items() if ref.dtype not in _PLAIN_FLOATS
    }
    if unsupported:
        listed = ", ".join(f"{name} is {label}" for name, label in list(unsupported.items())[:12])
        raise ModelLoadError(
            f"checkpoint {directory} stores {len(unsupported)} tensors in a dtype this build "
            f"cannot convert to {model.compute_dtype.label} (quantised formats are not "
            f"supported yet): {listed}",
            context={"unsupported_dtype": unsupported},
        )
    return bound


def _stream(model: CausalLM, bound: Mapping[str, TensorRef]) -> None:
    """Copy every bound tensor into its parameter, one at a time, in file order."""
    started = time.monotonic()
    total = sum(ref.nbytes for ref in bound.values())
    in_file_order = sorted(bound.items(), key=lambda item: (str(item[1].path), item[1].offset))
    with torch.no_grad():
        for path, group in groupby(in_file_order, key=lambda item: item[1].path):
            file = SafetensorsFile(path)
            try:
                for parameter, ref in group:
                    _copy_tensor(model, file, parameter, ref)
            except BaseException:
                # The traceback of the failure still holds views of the mapped file, which
                # keeps it mapped until the traceback is dropped; do not let that mask the error.
                with contextlib.suppress(BufferError):
                    file.close()
                raise
            file.close()
    logger.debug(
        "copied %d tensors, %d bytes in %.2fs", len(bound), total, time.monotonic() - started
    )


def _copy_tensor(model: CausalLM, file: SafetensorsFile, parameter: str, ref: TensorRef) -> None:
    stored = cast("torch.dtype", ref.dtype.torch_dtype)
    if ref.spec.numel == 0:
        model.load_weight(parameter, torch.empty(ref.shape, dtype=stored))
        return
    view = file.read(ref.name)
    with warnings.catch_warnings():
        # The mapping is read-only and nothing here writes through the tensor: it is copied.
        warnings.filterwarnings("ignore", message=_UNWRITABLE, category=UserWarning)
        tensor = torch.frombuffer(view, dtype=stored).reshape(ref.shape)
    model.load_weight(parameter, tensor)
