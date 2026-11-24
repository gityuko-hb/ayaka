"""Qwen2: biases on q, k and v only, and a paged forward that equals Llama's formulas with them.

The oracle is ``llama_support.llama_reference_hidden``, which adds a projection's bias when the
state has one; the state here has q, k and v biases and no ``o_proj`` bias, as a Qwen2
checkpoint does. The ``transformers`` comparison is in ``test_qwen2_transformers.py``.
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
from ayaka.models.qwen2 import Qwen2ForCausalLM  # noqa: E402
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


def _qwen2_state(config: Any, seed: int = 3) -> dict[str, Any]:
    with_biases = llama_state(
        llama_config(attention_bias=True, tie_word_embeddings=config.tie_word_embeddings),
        seed=seed,
    )
    return {name: t for name, t in with_biases.items() if not name.endswith("o_proj.bias")}


def _model(**overrides: object) -> Qwen2ForCausalLM:
    config = llama_config(architecture="Qwen2ForCausalLM", **overrides)
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    return Qwen2ForCausalLM(config, ops=ops, dtype=DType.FP32)


def test_the_registry_resolves_qwen2() -> None:
    assert resolve_model("Qwen2ForCausalLM") is Qwen2ForCausalLM


def test_qkv_has_a_bias_and_o_proj_has_none_whatever_the_config_says() -> None:
    names = {name for name, _ in _model(attention_bias=False).named_parameters()}

    assert "layers.0.attn.qkv_proj.bias" in names
    assert "layers.0.attn.o_proj.bias" not in names
    assert "layers.0.mlp.gate_up_proj.bias" not in names


@pytest.mark.parametrize(
    ("lengths", "prompts", "chunk"),
    [([13], [6], None), ([11, 8, 5], [7, 8, 1], 3)],
    ids=["prefill-then-decode", "three-requests-chunked"],
)
def test_paged_steps_equal_the_formulas(
    lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    model = _model()
    state = _qwen2_state(model.config)
    load_hf_state(model, state)
    generator = torch.Generator().manual_seed(5)
    sequences = [torch.randint(0, 64, (n,), generator=generator) for n in lengths]

    outputs = run_paged(model, sequences, prompt_lengths=prompts, chunk=chunk)

    for sequence, hidden in zip(sequences, outputs, strict=True):
        expected = llama_reference_hidden(state, model.config, sequence)
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)


def test_a_checkpoint_directory_loads_through_the_loader(tmp_path: Path) -> None:
    config = llama_config(architecture="Qwen2ForCausalLM", tie_word_embeddings=True)
    state = _qwen2_state(config)
    directory = write_llama(tmp_path / "qwen2", state, config)

    model = load_model(ModelSourceConfig(str(directory)), DeviceConfig(dtype=DType.FP32))

    assert isinstance(model, Qwen2ForCausalLM)
    assert model.lm_head.weight is model.embed_tokens.weight
    q_bias = state["model.layers.1.self_attn.q_proj.bias"]
    torch.testing.assert_close(
        model.get_parameter("layers.1.attn.qkv_proj.bias")[: q_bias.shape[0]], q_bias
    )
