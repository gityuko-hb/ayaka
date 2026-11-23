"""Phi-3: Llama's arithmetic on a checkpoint that stores ``qkv_proj`` and ``gate_up_proj`` fused.

The state is a Llama state (``llama_support``) with q, k, v concatenated into ``qkv_proj`` and
gate, up into ``gate_up_proj``, as a Phi-3 checkpoint stores them. The model must load it whole,
split nothing, and compute what Llama's formulas compute on the unfused state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.model_loader.loader import load_model  # noqa: E402
from ayaka.models.phi3 import Phi3ForCausalLM  # noqa: E402
from ayaka.models.registry import resolve_model  # noqa: E402
from gpt2_support import run_paged  # noqa: E402  # needs the importorskip above
from llama_support import (  # noqa: E402
    llama_config,
    llama_reference_hidden,
    llama_state,
    write_llama,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _fused(state: dict[str, Any], layers: int) -> dict[str, Any]:
    fused = dict(state)
    for layer in range(layers):
        attn, mlp = f"model.layers.{layer}.self_attn.", f"model.layers.{layer}.mlp."
        fused[f"{attn}qkv_proj.weight"] = torch.cat(
            [fused.pop(f"{attn}{part}_proj.weight") for part in "qkv"]
        )
        fused[f"{mlp}gate_up_proj.weight"] = torch.cat(
            [fused.pop(f"{mlp}{part}_proj.weight") for part in ("gate", "up")]
        )
    return fused


def test_the_registry_resolves_phi3() -> None:
    assert resolve_model("Phi3ForCausalLM") is Phi3ForCausalLM


def test_the_checkpoint_specs_are_the_fused_parameters() -> None:
    config = llama_config(architecture="Phi3ForCausalLM")
    model = Phi3ForCausalLM(
        config, ops=load_backend(["torch"], device="cpu", dtype=DType.FP32), dtype=DType.FP32
    )

    specs = {spec.name: spec.shape for spec in model.checkpoint_specs()}

    assert specs["layers.0.attn.qkv_proj.weight"] == ((4 + 2 * 2) * 8, 32)
    assert specs["layers.0.mlp.gate_up_proj.weight"] == (96, 32)
    assert "layers.0.attn.q_proj.weight" not in specs


def test_a_fused_checkpoint_loads_and_equals_the_formulas(tmp_path: Path) -> None:
    config = llama_config(architecture="Phi3ForCausalLM")
    state = llama_state(config, seed=7)
    directory = write_llama(tmp_path / "phi3", _fused(state, config.num_layers), config)
    model = load_model(ModelSourceConfig(str(directory)), DeviceConfig(dtype=DType.FP32))
    generator = torch.Generator().manual_seed(8)
    sequences = [torch.randint(0, 64, (n,), generator=generator) for n in (11, 6)]

    outputs = run_paged(model, sequences, prompt_lengths=[7, 6], chunk=3)

    assert isinstance(model, Phi3ForCausalLM)
    for sequence, hidden in zip(sequences, outputs, strict=True):
        expected = llama_reference_hidden(state, config, sequence)
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)
