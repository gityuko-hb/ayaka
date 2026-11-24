"""Quantize float32 tensors to 8-bit floats the way FP8 checkpoints store them, and write them.

Three layouts, as the producers write them:

* ``"tensor"``: one scale ``[1]`` (compressed-tensors ``strategy: tensor``);
* ``"channel"``: one scale per output row ``[out, 1]`` (``strategy: channel``);
* ``"block"``: one scale per ``[b0, b1]`` block (Hugging Face ``fp8`` ``weight_block_size``).

The scale of a group is ``amax / 448`` (the e4m3 maximum), so that ``scale * weight`` is the
dequantized value, as in both producers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from gpt2_support import TensorData

E4M3_MAX = 448.0

#: ``config.json`` ``quantization_config`` blocks with the fields the producers write.
HF_FP8_BLOCK: dict[str, object] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [16, 16],
}


def compressed_tensors(strategy: str, *, static_inputs: bool = False) -> dict[str, object]:
    """A compressed-tensors FP8 block like RedHatAI's, with ``strategy`` for the weights."""
    weights: dict[str, object] = {
        "block_structure": None,
        "dynamic": False,
        "group_size": None,
        "num_bits": 8,
        "observer": "minmax",
        "strategy": strategy,
        "symmetric": True,
        "type": "float",
    }
    inputs: dict[str, object] = {
        "dynamic": not static_inputs,
        "num_bits": 8,
        "strategy": "tensor" if static_inputs else "token",
        "symmetric": True,
        "type": "float",
    }
    return {
        "config_groups": {
            "group_0": {
                "input_activations": inputs,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": weights,
            }
        },
        "format": "naive-quantized" if static_inputs else "float-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def quantize(
    weight: torch.Tensor, layout: str, block: tuple[int, int] = (16, 16)
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(fp8 weight, float32 scale)`` with the scale in the layout's stored shape."""
    rows, columns = weight.shape
    match layout:
        case "tensor":
            scale = (weight.abs().amax() / E4M3_MAX).clamp_min(1e-12).reshape(1)
            expanded = scale.reshape(1, 1)
        case "channel":
            scale = (weight.abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp_min(1e-12)
            expanded = scale
        case "block":
            b0, b1 = block
            grid = torch.zeros(-(-rows // b0), -(-columns // b1))
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    tile = weight[i * b0 : (i + 1) * b0, j * b1 : (j + 1) * b1]
                    grid[i, j] = tile.abs().amax() / E4M3_MAX
            scale = grid.clamp_min(1e-12)
            expanded = scale.repeat_interleave(b0, 0)[:rows].repeat_interleave(b1, 1)[:, :columns]
        case _:
            raise ValueError(layout)
    stored = (weight / expanded).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return stored, scale


def dequantize(
    stored: torch.Tensor, scale: torch.Tensor, block: tuple[int, int] | None = None
) -> torch.Tensor:
    """``scale * weight`` in float32, for any of the three layouts."""
    rows, columns = stored.shape
    if block is None:
        expanded = scale.reshape(-1, 1) if scale.numel() > 1 else scale.reshape(1, 1)
    else:
        b0, b1 = block
        expanded = scale.repeat_interleave(b0, 0)[:rows].repeat_interleave(b1, 1)[:, :columns]
    return stored.to(torch.float32) * expanded


def fp8_data(tensor: torch.Tensor) -> TensorData:
    """The safetensors form of an ``float8_e4m3fn`` tensor."""
    stored = tensor.contiguous()
    return ("F8_E4M3", tuple(stored.shape), stored.flatten().view(torch.uint8).numpy().tobytes())


def quantize_state(
    state: dict[str, Any],
    layout: str,
    *,
    scale_name: str,
    block: tuple[int, int] = (16, 16),
    input_scales: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Quantize every decoder projection of a Hugging Face state.

    Returns ``(stored, dequantized)``: the tensors a checkpoint stores (8-bit weights, scales
    under ``scale_name``, an ``input_scale`` per layer when asked) and the float32 state the
    8-bit weights stand for, which the formulas run on.
    """
    stored: dict[str, Any] = {}
    dense: dict[str, Any] = {}
    for name, tensor in state.items():
        projection = name.endswith("_proj.weight") and ".layers." in name
        if not projection:
            stored[name] = tensor
            dense[name] = tensor
            continue
        weight, scale = quantize(tensor, layout, block)
        prefix = name.removesuffix("weight")
        stored[name] = weight
        stored[f"{prefix}{scale_name}"] = scale
        if input_scales:
            stored[f"{prefix}input_scale"] = torch.ones(1)
        dense[name] = dequantize(weight, scale, block if layout == "block" else None)
    return stored, dense
