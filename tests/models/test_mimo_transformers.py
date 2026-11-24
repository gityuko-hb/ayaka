"""Our MiMo against ``transformers``' on a tiny randomly initialised ``Qwen2Config``.

``transformers`` has no MiMo-7B class: the checkpoints' ``config.json`` names a ``mimo`` model type
that only the repository's own ``modeling_mimo.py`` defines, and the project never runs remote
code. That file subclasses ``Qwen2Model`` and ``Qwen2ForCausalLM`` and overrides no forward method,
so ``Qwen2ForCausalLM`` is the reference for the main path. The state dict is extended with the 16
random ``model.mtp_layers.0.*`` tensors a MiMo checkpoint stores, which our model must set aside.
Float32 hidden states and logits over a paged KV cache must agree with ``transformers``'
full-sequence forward to ``1e-4``.
"""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.models.mimo import MiMoForCausalLM  # noqa: E402
from hf_parity import (  # noqa: E402  # needs the importorskip above
    CASE_IDS,
    CASES,
    assert_matches_over_paged_kv,
    ours_like,
    randomise,
)

pytestmark = pytest.mark.integration

_HIDDEN, _INNER, _HEADS, _KV_HEADS, _HEAD_DIM = 64, 96, 4, 2, 16


def _reference(*, tie: bool) -> Any:
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=_HIDDEN,
        intermediate_size=_INNER,
        num_hidden_layers=2,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=tie,
        rope_parameters={"rope_type": "default", "rope_theta": 640000.0},
        use_sliding_window=False,
        sliding_window=None,
        architectures=["MiMoForCausalLM"],
    )
    return randomise(transformers.Qwen2ForCausalLM(config))


def _with_mtp_layer(state: dict[str, Any]) -> dict[str, Any]:
    """The state dict plus the 16 tensors of MiMo's multi-token-prediction layer."""
    generator = torch.Generator().manual_seed(21)
    q_rows, kv_rows = _HEADS * _HEAD_DIM, _KV_HEADS * _HEAD_DIM
    shapes = {
        "token_layernorm.weight": (_HIDDEN,),
        "hidden_layernorm.weight": (_HIDDEN,),
        "input_layernorm.weight": (_HIDDEN,),
        "post_attention_layernorm.weight": (_HIDDEN,),
        "final_layernorm.weight": (_HIDDEN,),
        "input_proj.weight": (_HIDDEN, 2 * _HIDDEN),
        "self_attn.q_proj.weight": (q_rows, _HIDDEN),
        "self_attn.q_proj.bias": (q_rows,),
        "self_attn.k_proj.weight": (kv_rows, _HIDDEN),
        "self_attn.k_proj.bias": (kv_rows,),
        "self_attn.v_proj.weight": (kv_rows, _HIDDEN),
        "self_attn.v_proj.bias": (kv_rows,),
        "self_attn.o_proj.weight": (_HIDDEN, q_rows),
        "mlp.gate_proj.weight": (_INNER, _HIDDEN),
        "mlp.up_proj.weight": (_INNER, _HIDDEN),
        "mlp.down_proj.weight": (_HIDDEN, _INNER),
    }
    extra = {
        f"model.mtp_layers.0.{name}": torch.randn(shape, generator=generator)
        for name, shape in shapes.items()
    }
    return {**state, **extra}


@pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize(("lengths", "prompts", "chunk"), CASES, ids=CASE_IDS)
def test_forward_matches_transformers_over_paged_kv(
    tie: bool, lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    reference = _reference(tie=tie)
    model = ours_like(reference, MiMoForCausalLM, state=_with_mtp_layer)

    assert_matches_over_paged_kv(reference, model, lengths, prompts, chunk)


def test_the_config_of_the_reference_is_read_as_mimo() -> None:
    config = ModelConfig.from_hf_config(_reference(tie=False).config.to_dict())

    assert config.architecture == "MiMoForCausalLM"
    assert (config.head_dim, config.rope_theta, config.sliding_window) == (16, 640000.0, None)
