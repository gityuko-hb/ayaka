"""Our Qwen2 against ``transformers``' on tiny randomly initialised ``Qwen2Config``s.

Needs torch and the ``hf`` extra. The config goes through ``ModelConfig.from_hf_config`` as a
downloaded ``config.json`` would, and the weights through the model's own mapping, so the q, k and
v biases are split into ``qkv_proj`` exactly as a checkpoint is. Float32 hidden states and logits
over a paged KV cache must agree with ``transformers``' full-sequence forward to ``1e-4``.
"""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.models.qwen2 import Qwen2ForCausalLM  # noqa: E402
from hf_parity import (  # noqa: E402  # needs the importorskip above
    CASE_IDS,
    CASES,
    assert_matches_over_paged_kv,
    ours_like,
    randomise,
)

pytestmark = pytest.mark.integration


def _reference(*, tie: bool, sliding: bool = False) -> Any:
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=tie,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        use_sliding_window=sliding,
        sliding_window=32,
        max_window_layers=1,
        architectures=["Qwen2ForCausalLM"],
    )
    return randomise(transformers.Qwen2ForCausalLM(config))


@pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize(("lengths", "prompts", "chunk"), CASES, ids=CASE_IDS)
def test_forward_matches_transformers_over_paged_kv(
    tie: bool, lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    reference = _reference(tie=tie)
    model = ours_like(reference, Qwen2ForCausalLM)

    assert_matches_over_paged_kv(reference, model, lengths, prompts, chunk)


def test_the_qkv_biases_are_loaded_and_o_proj_has_none() -> None:
    reference = _reference(tie=False)
    model = ours_like(reference, Qwen2ForCausalLM)
    attn = reference.model.layers[0].self_attn

    fused = model.get_parameter("layers.0.attn.qkv_proj.bias")

    torch.testing.assert_close(
        fused, torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias])
    )
    assert model.get_submodule("layers.0.attn.o_proj").bias is None


def test_a_sliding_window_in_use_caps_the_served_context() -> None:
    config = ModelConfig.from_hf_config(_reference(tie=False, sliding=True).config.to_dict())

    assert config.sliding_window == 32
    assert config.context_length == 32
    assert config.max_position_embeddings == 128


def test_a_sliding_window_switched_off_is_ignored() -> None:
    config = ModelConfig.from_hf_config(_reference(tie=False).config.to_dict())

    assert config.sliding_window is None
    assert config.context_length == 128
