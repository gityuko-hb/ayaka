"""The quantization recipe of a checkpoint: how its 8-bit weights and their scales are stored.

A quantized checkpoint says how it was made in ``config.json`` ``quantization_config``.
:func:`read_quant_format` turns that block into a :class:`QuantFormat`, a data record that says
what the loader will find next to each quantized ``weight`` and how to read it back:

=======================  ===============================  ====================================
Producer                 Stored next to ``X.weight``      Scale layout
=======================  ===============================  ====================================
``fp8`` (Hugging Face)   ``X.weight_scale_inv``           one scale per ``[b0, b1]`` block
``compressed-tensors``   ``X.weight_scale`` (and          one scale per tensor (``[1]``), per
                         ``X.input_scale``, unused)       output row (``[out, 1]``) or per block
=======================  ===============================  ====================================

In every layout the weight is ``scale * weight`` once dequantized: the ``_inv`` of Hugging Face's
name is the inverse of the quantization step, which is the dequantization factor.

Only what Ayaka runs is accepted; anything else is refused with the reason, never misread:
8-bit floating weights (e4m3), symmetric, with static scales; one recipe for every linear layer
of the decoder, the output head left in 16 bit. Activation scales are read past: activations
stay in the compute dtype.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast

from ayaka.core.errors import ModelLoadError
from ayaka.utils.dtypes import DType

__all__ = ["QuantFormat", "ScaleLayout", "read_quant_format"]

type ScaleLayout = Literal["tensor", "channel", "block"]
"""How many scales a weight has: one, one per output row, or one per ``[b0, b1]`` block."""

type QuantProducer = Literal["fp8", "compressed-tensors", "mxfp4"]

_LAYOUTS: Final[tuple[ScaleLayout, ...]] = ("tensor", "channel", "block")
#: Modules a recipe may leave unquantized: the output head is not a decoder linear layer.
_UNQUANTIZED_OK: Final = frozenset({"lm_head"})
_COMPRESSED_FORMATS: Final = frozenset({"float-quantized", "naive-quantized"})
_FP8: Final = frozenset({DType.FP8_E4M3, DType.FP8_E5M2})


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """How a checkpoint stores its quantized linear weights.

    Attributes:
        producer: The tool that wrote the checkpoint: ``"fp8"`` (Hugging Face) or
            ``"compressed-tensors"``.
        weight_dtype: The stored dtype of every quantized ``weight``.
        scale_layout: How the scales are laid out next to each weight.
        block: ``(rows, columns)`` of one scale block for the ``"block"`` layout, else ``None``.
        scale_name: The suffix the scale is stored under, in place of ``weight``.
        ignored: Suffixes stored next to a quantized weight that Ayaka does not use
            (``input_scale``: activations stay in the compute dtype).

    Raises:
        ValueError: If the fields do not describe one layout: a block without the ``"block"``
            layout, or a block side that is not positive.
    """

    producer: QuantProducer
    weight_dtype: DType
    scale_layout: ScaleLayout
    block: tuple[int, int] | None
    scale_name: str
    ignored: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scale_layout not in _LAYOUTS:
            raise ValueError(f"scale_layout must be one of {_LAYOUTS}, got {self.scale_layout!r}")
        if (self.block is not None) != (self.scale_layout == "block"):
            raise ValueError("block is given exactly when scale_layout is 'block'")
        if self.block is not None and min(self.block) < 1:
            raise ValueError(f"block sides must be positive, got {self.block}")
        if self.producer == "mxfp4":
            if self.weight_dtype is not DType.FP4_E2M1:
                raise ValueError(
                    f"mxfp4 weight_dtype must be fp4_e2m1, got {self.weight_dtype.label}"
                )
        elif self.weight_dtype not in _FP8:
            raise ValueError(f"weight_dtype must be an 8-bit float, got {self.weight_dtype.label}")

    @property
    def bits(self) -> int:
        """Bits per stored weight element (the scales are not counted)."""
        return self.weight_dtype.bits

    @property
    def name(self) -> str:
        """A short label: ``fp8_e4m3/block-128x128``, ``fp8_e4m3/channel``."""
        if self.block is None:
            return f"{self.weight_dtype.label}/{self.scale_layout}"
        return f"{self.weight_dtype.label}/block-{self.block[0]}x{self.block[1]}"


def read_quant_format(config: Mapping[str, object]) -> QuantFormat | None:
    """The recipe of a checkpoint, from its ``config.json``; ``None`` when it is not quantized.

    Args:
        config: The parsed ``config.json``.

    Raises:
        ModelLoadError: If ``quantization_config`` names a method or a variant this build does
            not run. The message says which field and why.
    """
    raw = config.get("quantization_config")
    if raw is None:
        return None
    block = _mapping(raw, "quantization_config")
    method = block.get("quant_method")
    match method:
        case "fp8":
            return _hf_fp8(block)
        case "compressed-tensors":
            return _compressed_tensors(block)
        case "mxfp4":
            return QuantFormat(
                producer="mxfp4",
                weight_dtype=DType.FP4_E2M1,
                scale_layout="block",
                block=(1, 32),
                scale_name="scales",
            )
        case _:
            raise _refused(
                f"quant_method {method!r} is not supported (supported: 'fp8', "
                "'compressed-tensors' with 8-bit float weights, 'mxfp4')"
            )


def _hf_fp8(block: Mapping[str, object]) -> QuantFormat:
    if block.get("fmt", "e4m3") != "e4m3":
        raise _refused(f"fp8 fmt {block.get('fmt')!r} is not supported, only 'e4m3'")
    if block.get("activation_scheme", "dynamic") != "dynamic":
        raise _refused(
            f"fp8 activation_scheme {block.get('activation_scheme')!r} is not supported, "
            "only 'dynamic'"
        )
    _require_only_head_unquantized(block.get("modules_to_not_convert"), "modules_to_not_convert")
    size = block.get("weight_block_size")
    if size is None:
        raise _refused("fp8 without weight_block_size (one scale per tensor) is not supported")
    sides = _sides(size, "weight_block_size")
    return QuantFormat(
        producer="fp8",
        weight_dtype=DType.FP8_E4M3,
        scale_layout="block",
        block=sides,
        scale_name="weight_scale_inv",
    )


def _compressed_tensors(block: Mapping[str, object]) -> QuantFormat:
    fmt = block.get("format")
    if fmt not in _COMPRESSED_FORMATS:
        raise _refused(
            f"compressed-tensors format {fmt!r} is not supported "
            f"(supported: {sorted(_COMPRESSED_FORMATS)})"
        )
    status = block.get("quantization_status", "compressed")
    if status != "compressed":
        raise _refused(f"compressed-tensors quantization_status {status!r} is not 'compressed'")
    if block.get("kv_cache_scheme") is not None:
        raise _refused("a compressed-tensors kv_cache_scheme is not supported")
    _require_only_head_unquantized(block.get("ignore"), "ignore")
    groups = _mapping(block.get("config_groups"), "config_groups")
    if len(groups) != 1:
        raise _refused(f"config_groups has {len(groups)} groups; one recipe for every layer only")
    [group_value] = groups.values()
    group = _mapping(group_value, "config_groups group")
    targets = group.get("targets")
    if targets != ["Linear"]:
        raise _refused(f"config_groups targets {targets!r} are not ['Linear']")
    if group.get("output_activations") is not None:
        raise _refused("quantized output activations are not supported")
    weights = _mapping(group.get("weights"), "weights")
    problems: list[str] = []
    if weights.get("type") != "float":
        problems.append(f"type is {weights.get('type')!r}, not 'float'")
    if weights.get("num_bits") != 8:
        problems.append(f"num_bits is {weights.get('num_bits')!r}, not 8")
    if weights.get("symmetric") is not True:
        problems.append("the weights are not symmetric")
    if weights.get("dynamic") not in (None, False):
        problems.append("the weight scales are dynamic")
    if problems:
        raise _refused("compressed-tensors weights: " + "; ".join(problems))
    strategy = weights.get("strategy")
    sides: tuple[int, int] | None = None
    layout: ScaleLayout
    match strategy:
        case "tensor":
            layout = "tensor"
        case "channel":
            layout = "channel"
        case "block":
            layout = "block"
            sides = _sides(weights.get("block_structure"), "block_structure")
        case _:
            raise _refused(
                f"compressed-tensors weight strategy {strategy!r} is not supported "
                "(supported: 'tensor', 'channel', 'block')"
            )
    activations = group.get("input_activations")
    static_inputs = isinstance(activations, Mapping) and not cast(
        "Mapping[str, object]", activations
    ).get("dynamic", False)
    return QuantFormat(
        producer="compressed-tensors",
        weight_dtype=DType.FP8_E4M3,
        scale_layout=layout,
        block=sides,
        scale_name="weight_scale",
        ignored=("input_scale",) if static_inputs else (),
    )


def _require_only_head_unquantized(value: object, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise _refused(f"{field} must be a list of module names, got {value!r}")
    names = {str(item) for item in cast("Sequence[object]", value)}
    extra = sorted(names - _UNQUANTIZED_OK)
    if extra:
        raise _refused(
            f"{field} leaves {extra} in 16 bit; only the output head may stay unquantized"
        )


def _sides(value: object, field: str) -> tuple[int, int]:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and len(cast("Sequence[object]", value)) == 2
    ):
        rows, columns = cast("Sequence[object]", value)
        if (
            isinstance(rows, int)
            and isinstance(columns, int)
            and not isinstance(rows, bool)
            and not isinstance(columns, bool)
            and rows > 0
            and columns > 0
        ):
            return rows, columns
    raise _refused(f"{field} must be two positive integers, got {value!r}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _refused(f"{field} must be an object, got {type(value).__name__}")
    return cast("Mapping[str, object]", value)


def _refused(reason: str) -> ModelLoadError:
    return ModelLoadError(
        f"quantization_config: {reason}", context={"field": "quantization_config"}
    )
