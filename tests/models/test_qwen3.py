"""Qwen3: per-head query and key norms, and a paged forward that equals the formulas with them.

The oracle is ``llama_support.llama_reference_hidden`` on a state with ``q_norm`` and ``k_norm``
of width ``head_dim``: it normalises each head after the projection and before the rotary. The
``transformers`` comparison is in ``test_qwen3_transformers.py``.
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
from ayaka.models.qwen3 import Qwen3ForCausalLM  # noqa: E402
from ayaka.models.registry import resolve_model  # noqa: E402
from gpt2_support import load_hf_state, run_paged  # noqa: E402  # needs the importorskip above
from llama_support import (  # noqa: E402
    llama_config,
    llama_reference_hidden,
    llama_state,
    write_llama,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _model(**overrides: object) -> Qwen3ForCausalLM:
    config = llama_config(architecture="Qwen3ForCausalLM", head_dim=16, **overrides)
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    return Qwen3ForCausalLM(config, ops=ops, dtype=DType.FP32)


def test_the_registry_resolves_qwen3() -> None:
    assert resolve_model("Qwen3ForCausalLM") is Qwen3ForCausalLM


def test_each_layer_has_a_head_wide_query_and_key_norm() -> None:
    shapes = {name: tuple(p.shape) for name, p in _model().named_parameters()}

    assert shapes["layers.0.attn.q_norm.weight"] == (16,)
    assert shapes["layers.1.attn.k_norm.weight"] == (16,)
    assert shapes["layers.0.attn.qkv_proj.weight"] == ((4 + 2 * 2) * 16, 32)
    assert "layers.0.attn.qkv_proj.bias" not in shapes


@pytest.mark.parametrize(
    ("lengths", "prompts", "chunk"),
    [([13], [6], None), ([11, 8, 5], [7, 8, 1], 3)],
    ids=["prefill-then-decode", "three-requests-chunked"],
)
def test_paged_steps_equal_the_formulas(
    lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    model = _model()
    state = llama_state(model.config, seed=4, qk_norm="head")
    load_hf_state(model, state)
    generator = torch.Generator().manual_seed(6)
    sequences = [torch.randint(0, 64, (n,), generator=generator) for n in lengths]

    outputs = run_paged(model, sequences, prompt_lengths=prompts, chunk=chunk)

    for sequence, hidden in zip(sequences, outputs, strict=True):
        expected = llama_reference_hidden(state, model.config, sequence)
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)


def test_a_checkpoint_without_the_norms_is_refused_with_their_names(tmp_path: Path) -> None:
    config = llama_config(architecture="Qwen3ForCausalLM", head_dim=16)
    state = llama_state(config, qk_norm="head")
    del state["model.layers.0.self_attn.q_norm.weight"]
    directory = write_llama(tmp_path / "qwen3", state, config)

    with pytest.raises(Exception, match=r"missing 1 parameters: layers\.0\.attn\.q_norm\.weight"):
        load_model(ModelSourceConfig(str(directory)), DeviceConfig(dtype=DType.FP32))
