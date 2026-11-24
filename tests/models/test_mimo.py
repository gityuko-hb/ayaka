"""MiMo-7B: a Qwen2 model whose checkpoint also stores a multi-token-prediction layer.

The authors' ``modeling_mimo.py`` subclasses ``Qwen2Model`` and ``Qwen2ForCausalLM`` and adds
``mtp_layers``; the main path is Qwen2's. So the model must compute what Qwen2's formulas compute,
read none of the 16 ``model.mtp_layers.0.*`` tensors, and still refuse any other tensor it does not
know. The ``transformers`` comparison is in ``test_mimo_transformers.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.model import ModelConfig
from ayaka.core.errors import ModelLoadError

torch: Any = pytest.importorskip("torch")

from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.model_loader.loader import describe_model, load_model  # noqa: E402
from ayaka.models.mimo import MiMoForCausalLM  # noqa: E402
from ayaka.models.qwen2 import Qwen2ForCausalLM  # noqa: E402
from ayaka.models.registry import resolve_model  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402
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

# ``config.json`` of XiaomiMiMo/MiMo-7B-RL, without ``auto_map`` (Hugging Face Hub, 2026-11-24).
MIMO_7B_RL: dict[str, object] = {
    "architectures": ["MiMoForCausalLM"],
    "attention_bias": True,
    "attention_dropout": 0.0,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "max_position_embeddings": 32768,
    "max_window_layers": 36,
    "model_type": "mimo",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "num_nextn_predict_layers": 1,
    "rms_norm_eps": 1e-05,
    "rope_theta": 640000,
    "sliding_window": 32768,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_mrope": False,
    "use_sliding_window": False,
    "vocab_size": 151680,
}

_MTP_PREFIX = "model.mtp_layers.0."


def _state(config: ModelConfig, seed: int = 3) -> dict[str, Any]:
    """A Qwen2 state: biases on q, k and v, none on ``o_proj``."""
    biased = llama_state(llama_config(attention_bias=True), seed=seed)
    return {name: t for name, t in biased.items() if not name.endswith("o_proj.bias")}


def _mtp_tensors(config: ModelConfig, seed: int = 9) -> dict[str, Any]:
    """The 16 tensors of MiMo's one multi-token-prediction layer, named and shaped as stored."""
    generator = torch.Generator().manual_seed(seed)
    hidden, inner = config.hidden_size, config.intermediate_size
    q_rows = config.num_attention_heads * config.head_dim
    kv_rows = config.num_kv_heads * config.head_dim

    def rand(*shape: int) -> Any:
        return torch.randn(shape, generator=generator)

    names = {
        "token_layernorm.weight": (hidden,),
        "hidden_layernorm.weight": (hidden,),
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "final_layernorm.weight": (hidden,),
        "input_proj.weight": (hidden, 2 * hidden),
        "self_attn.q_proj.weight": (q_rows, hidden),
        "self_attn.q_proj.bias": (q_rows,),
        "self_attn.k_proj.weight": (kv_rows, hidden),
        "self_attn.k_proj.bias": (kv_rows,),
        "self_attn.v_proj.weight": (kv_rows, hidden),
        "self_attn.v_proj.bias": (kv_rows,),
        "self_attn.o_proj.weight": (hidden, q_rows),
        "mlp.gate_proj.weight": (inner, hidden),
        "mlp.up_proj.weight": (inner, hidden),
        "mlp.down_proj.weight": (hidden, inner),
    }
    return {f"{_MTP_PREFIX}{name}": rand(*shape) for name, shape in names.items()}


def _load(directory: Path) -> Any:
    return load_model(ModelSourceConfig(str(directory)), DeviceConfig(dtype=DType.FP32))


def test_the_registry_resolves_mimo_to_a_qwen2_model() -> None:
    assert resolve_model("MiMoForCausalLM") is MiMoForCausalLM
    assert issubclass(MiMoForCausalLM, Qwen2ForCausalLM)


def test_mimo_7b_rl_reads_as_the_qwen2_shape_it_has() -> None:
    config = ModelConfig.from_hf_config(MIMO_7B_RL)

    assert config.architecture == "MiMoForCausalLM"
    assert (config.hidden_size, config.num_layers, config.intermediate_size) == (4096, 36, 11008)
    assert (config.num_attention_heads, config.num_kv_heads, config.head_dim) == (32, 8, 128)
    assert (config.vocab_size, config.rope_theta, config.tie_word_embeddings) == (
        151680,
        640000.0,
        False,
    )
    # ``sliding_window`` 32768 is switched off by ``use_sliding_window: false``.
    assert (config.sliding_window, config.context_length) == (None, 32768)


def test_a_config_without_the_keys_gets_the_defaults_of_its_config_class() -> None:
    raw = {k: v for k, v in MIMO_7B_RL.items() if k not in ("rope_theta", "tie_word_embeddings")}

    config = ModelConfig.from_hf_config(raw)

    assert (config.rope_theta, config.tie_word_embeddings) == (10000.0, False)


def test_the_model_has_no_parameter_for_the_multi_token_prediction_layer() -> None:
    config = llama_config(architecture="MiMoForCausalLM")
    from ayaka.ops.backend import load_backend

    model = MiMoForCausalLM(
        config, ops=load_backend(["torch"], device="cpu", dtype=DType.FP32), dtype=DType.FP32
    )

    names = {name for name, _ in model.named_parameters()}
    assert not any("mtp" in name for name in names)
    assert "layers.0.attn.qkv_proj.bias" in names
    assert "layers.0.attn.o_proj.bias" not in names


@pytest.mark.parametrize(
    ("lengths", "prompts", "chunk"),
    [([13], [6], None), ([11, 8, 5], [7, 8, 1], 3)],
    ids=["prefill-then-decode", "three-requests-chunked"],
)
def test_a_checkpoint_with_its_mtp_layer_loads_and_equals_the_formulas(
    tmp_path: Path, lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    config = llama_config(architecture="MiMoForCausalLM")
    state = _state(config)
    directory = write_llama(tmp_path / "mimo", state | _mtp_tensors(config), config)

    model = _load(directory)
    generator = torch.Generator().manual_seed(5)
    sequences = [torch.randint(0, 64, (n,), generator=generator) for n in lengths]
    outputs = run_paged(model, sequences, prompt_lengths=prompts, chunk=chunk)

    assert isinstance(model, MiMoForCausalLM)
    for sequence, hidden in zip(sequences, outputs, strict=True):
        expected = llama_reference_hidden(state, config, sequence)
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)


def test_the_planned_weights_do_not_count_the_multi_token_prediction_layer(tmp_path: Path) -> None:
    config = llama_config(architecture="MiMoForCausalLM")
    state = _state(config)
    mtp = _mtp_tensors(config)
    directory = write_llama(tmp_path / "mimo", state | mtp, config)

    _, model = describe_model(ModelSourceConfig(str(directory)), DeviceConfig(dtype=DType.FP32))

    planned = sum(p.numel() for p in model.parameters())
    assert planned == sum(t.numel() for t in state.values())  # fusing partitions the main tensors
    assert sum(t.numel() for t in mtp.values()) > 0  # and the layer the file also holds is extra


def test_a_tensor_the_model_does_not_know_is_still_refused(tmp_path: Path) -> None:
    config = llama_config(architecture="MiMoForCausalLM")
    stray = {
        "model.mtp_extra.weight": torch.zeros(4),
        "model.layers.0.mtp_layers.x": torch.zeros(4),
    }
    directory = write_llama(tmp_path / "mimo", _state(config) | stray, config)

    with pytest.raises(ModelLoadError) as caught:
        _load(directory)

    message = str(caught.value)
    assert "model.mtp_extra.weight" in message
    assert "model.layers.0.mtp_layers.x" in message


def test_a_missing_main_tensor_is_still_refused_even_with_the_mtp_layer_present(
    tmp_path: Path,
) -> None:
    config = llama_config(architecture="MiMoForCausalLM")
    state = _state(config)
    del state["model.layers.1.mlp.down_proj.weight"]
    directory = write_llama(tmp_path / "mimo", state | _mtp_tensors(config), config)

    with pytest.raises(ModelLoadError, match=r"layers\.1\.mlp\.down_proj\.weight"):
        _load(directory)
