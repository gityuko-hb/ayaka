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
import dataclasses
import logging
import re
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
from ayaka.weights.gguf import gguf_hf_config, read_gguf
from ayaka.weights.manifest import WeightManifest
from ayaka.weights.quant import read_quant_format
from ayaka.weights.safetensors import SafetensorsFile

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayaka.configs.device import DeviceConfig
    from ayaka.configs.model_source import ModelSourceConfig
    from ayaka.models.base import CausalLM
    from ayaka.ops.backend import OpsBackend
    from ayaka.weights.mapping import WeightMapping
    from ayaka.weights.quant import QuantFormat
    from ayaka.weights.tensor import TensorRef

__all__ = [
    "describe_model",
    "load_model",
    "load_weights",
    "read_model_config",
    "stored_weights",
]

logger = logging.getLogger(__name__)

_CONFIG_FILE: Final = "config.json"
_CONFIG_LIMIT: Final = 16 * 2**20
#: Stored dtypes that a copy converts correctly. Quantised and 8-bit floating formats carry
#: scales that the copy would drop, so they load only into a parameter of their own dtype.
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
    gguf = _gguf_path(source)
    if gguf is not None:
        raise ModelLoadError(
            f"{gguf}: running GGUF tensors is not implemented; `ayaka plan` reads the file's "
            "header and plans it, but the engine loads safetensors checkpoints only",
            context={"path": str(gguf), "format": "gguf"},
        )
    started = time.monotonic()
    directory = resolve_source(source)
    config, quant = _read_checkpoint_config(directory)
    model_class = resolve_model(config.architecture)
    backend = ops or load_backend(
        device.backends, device=device.device_type, dtype=device.dtype, head_dim=config.head_dim
    )
    target = resolve_device(device.device, capability="model_loader")

    model = model_class(config, ops=backend, dtype=device.dtype, device="meta", quant=quant)
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


def describe_model(
    source: ModelSourceConfig, device: DeviceConfig, *, ops: OpsBackend | None = None
) -> tuple[Path, CausalLM]:
    """The model a source describes, built on the meta device: shapes and code, no weights.

    The checkpoint's ``config.json`` is read and the model built as :func:`load_model` builds it,
    shared parameters tied, but nothing is allocated and no tensor is read, so a plan can ask the
    model for its size and its KV geometry before deciding whether to load it.

    Returns:
        The checkpoint directory and the model.

    Raises:
        As :func:`load_model`, except for errors that need the tensors.
    """
    _require_readable_format(source)
    gguf = _gguf_path(source)
    if gguf is not None:
        directory, config, quant = gguf, _gguf_config(gguf), None
    else:
        directory = resolve_source(source)
        config, quant = _read_checkpoint_config(directory)
    model_class = resolve_model(config.architecture)
    backend = ops or load_backend(
        device.backends, device=device.device_type, dtype=device.dtype, head_dim=config.head_dim
    )
    model = model_class(config, ops=backend, dtype=device.dtype, device="meta", quant=quant)
    model.tie_weights()
    if quant is not None:
        logger.warning(
            "%s weights (%s): kept 8 bit and dequantized in slices before each product, "
            "which is slower than 16-bit weights",
            quant.name,
            quant.producer,
        )
    return directory, model


def read_model_config(source: ModelSourceConfig) -> ModelConfig:
    """The architecture a source describes, from its ``config.json``: nothing else is read.

    The engine reads it before choosing its ops backends, which depend on the head width.

    Raises:
        As :func:`describe_model`.
    """
    _require_readable_format(source)
    gguf = _gguf_path(source)
    if gguf is not None:
        return _gguf_config(gguf)
    return _read_checkpoint_config(resolve_source(source))[0]


def stored_weights(source: ModelSourceConfig) -> tuple[int, dict[str, int]] | None:
    """The bytes of a GGUF file's tensors and how many tensors each GGML type stores.

    A GGUF file mixes quantization types whose blocks have no PyTorch dtype, so its size comes
    from the tensor table (:attr:`~ayaka.weights.gguf.GGUFFile.weights_bytes`), not from a model.
    ``None`` for a safetensors checkpoint, whose weights are the model's parameters.

    Raises:
        ModelLoadError: If the GGUF header is not valid.
    """
    gguf = _gguf_path(source)
    if gguf is None:
        return None
    file = read_gguf(gguf)
    return file.weights_bytes, file.type_counts()


def _gguf_path(source: ModelSourceConfig) -> Path | None:
    """The file, when ``source.model`` names a local ``.gguf`` file and the format is ``auto``."""
    if source.format is not RequestFormat.AUTO:
        return None
    path = Path(source.model).expanduser()
    return path if path.suffix.lower() == ".gguf" and path.is_file() else None


def _gguf_config(path: Path) -> ModelConfig:
    return ModelConfig.from_hf_config(gguf_hf_config(read_gguf(path)))


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


def _read_checkpoint_config(directory: Path) -> tuple[ModelConfig, QuantFormat | None]:
    """The architecture and the quantization recipe that ``config.json`` describes."""
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
    raw = cast("Mapping[str, object]", document)
    return ModelConfig.from_hf_config(raw), read_quant_format(raw)


def _bind(model: CausalLM, directory: Path) -> Mapping[str, TensorRef]:
    """Match the checkpoint to the model from headers alone; raise one error for every problem."""
    manifest = WeightManifest.from_directory(directory)
    specs = {spec.name: spec for spec in model.checkpoint_specs()}
    bound = _mapping(model).bind(manifest, specs.values())
    unsupported = {
        name: ref.dtype.label
        for name, ref in bound.items()
        if not _converts(ref.dtype, specs[name].dtype)
    }
    if unsupported:
        listed = ", ".join(
            f"{name} is {label}, expected {specs[name].dtype.label}"
            for name, label in list(unsupported.items())[:12]
        )
        raise ModelLoadError(
            f"checkpoint {directory} stores {len(unsupported)} tensors in a dtype this build "
            f"cannot load as {model.compute_dtype.label} weights"
            f"{_quant_label(model)}: {listed}",
            context={"unsupported_dtype": unsupported},
        )
    return bound


def _mapping(model: CausalLM) -> WeightMapping:
    """The model's names, plus the scales and the unused tensors of its quantization recipe."""
    mapping = type(model).weight_mapping(model.config)
    quant = model.quant_format
    if quant is None or quant.producer == "mxfp4":
        return mapping
    return dataclasses.replace(
        mapping,
        suffixes=(*mapping.suffixes, (quant.scale_name, "weight_scale")),
        ignore=(*mapping.ignore, *(rf".*\.{re.escape(suffix)}" for suffix in quant.ignored)),
    )


def _converts(stored: DType, held: DType) -> bool:
    """Whether a stored dtype loads into a parameter of ``held``: a float copy, or the same 8 bits.

    An 8-bit tensor carries scales that a copy would drop, so it loads only into a parameter
    that holds it as is (a quantized weight), never into a 16-bit one.
    """
    if held in _PLAIN_FLOATS:
        return stored in _PLAIN_FLOATS
    return stored is held


def _quant_label(model: CausalLM) -> str:
    quant = model.quant_format
    return "" if quant is None else f" with {quant.producer} {quant.name} quantization"


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
