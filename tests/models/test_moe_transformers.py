"""Our OLMoE and Qwen3-MoE against ``transformers``' on tiny randomly initialised configs.

Needs torch and the ``hf`` extra. ``transformers`` 5 keeps each layer's experts as two 3-D
tensors (``mlp.experts.gate_up_proj`` ``[E, 2I, H]``, ``down_proj`` ``[E, H, I]``); a checkpoint
on disk stores them one by one (``mlp.experts.{e}.gate_proj.weight``, ...), which is what our
models read, so the state dict is split into those names first. There are 8 experts, 3 per
token, so the paged cases reach every strategy of the torch ``experts`` op (gathered, dense,
grouped). Float32 hidden states and logits over a paged KV cache must agree with
``transformers``' full-sequence forward to ``1e-4``, as #113 fixed before the run.
"""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.models.olmoe import OlmoeForCausalLM  # noqa: E402
from ayaka.models.qwen3_moe import Qwen3MoeForCausalLM  # noqa: E402
from hf_parity import (  # noqa: E402  # needs the importorskip above
    CASE_IDS,
    CASES,
    assert_matches_over_paged_kv,
    ours_like,
    randomise,
)

pytestmark = pytest.mark.integration

_EXPERTS, _TOP_K = 8, 3


def split_experts(state: dict[str, Any]) -> dict[str, Any]:
    """``transformers``' stacked experts written back as a checkpoint stores them."""
    split: dict[str, Any] = {}
    for name, tensor in state.items():
        if name.endswith(".mlp.experts.gate_up_proj"):
            base = name.removesuffix("gate_up_proj")
            inner = tensor.shape[1] // 2
            for expert in range(tensor.shape[0]):
                split[f"{base}{expert}.gate_proj.weight"] = tensor[expert, :inner]
                split[f"{base}{expert}.up_proj.weight"] = tensor[expert, inner:]
        elif name.endswith(".mlp.experts.down_proj"):
            base = name.removesuffix("down_proj")
            for expert in range(tensor.shape[0]):
                split[f"{base}{expert}.down_proj.weight"] = tensor[expert]
        else:
            split[name] = tensor
    return split


def _olmoe(*, tie: bool, renormalize: bool) -> Any:
    config = transformers.OlmoeConfig(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        num_experts=_EXPERTS,
        num_experts_per_tok=_TOP_K,
        norm_topk_prob=renormalize,
        tie_word_embeddings=tie,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        pad_token_id=0,
        architectures=["OlmoeForCausalLM"],
    )
    return randomise(transformers.OlmoeForCausalLM(config))


def _qwen3_moe(*, dense_layers: list[int], step: int) -> Any:
    config = transformers.Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=40,
        moe_intermediate_size=20,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        num_experts=_EXPERTS,
        num_experts_per_tok=_TOP_K,
        norm_topk_prob=True,
        decoder_sparse_step=step,
        mlp_only_layers=dense_layers,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        architectures=["Qwen3MoeForCausalLM"],
    )
    return randomise(transformers.Qwen3MoeForCausalLM(config))


@pytest.mark.parametrize(
    ("tie", "renormalize"), [(False, False), (True, True)], ids=["untied", "tied-renormalised"]
)
@pytest.mark.parametrize(("lengths", "prompts", "chunk"), CASES, ids=CASE_IDS)
def test_olmoe_matches_transformers_over_paged_kv(
    tie: bool, renormalize: bool, lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    reference = _olmoe(tie=tie, renormalize=renormalize)
    model = ours_like(reference, OlmoeForCausalLM, state=split_experts)

    assert_matches_over_paged_kv(reference, model, lengths, prompts, chunk)


@pytest.mark.parametrize(
    ("dense_layers", "step"), [([], 1), ([1], 1), ([], 2)], ids=["all-moe", "mlp-only-1", "step-2"]
)
@pytest.mark.parametrize(("lengths", "prompts", "chunk"), CASES, ids=CASE_IDS)
def test_qwen3_moe_matches_transformers_over_paged_kv(
    dense_layers: list[int], step: int, lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    reference = _qwen3_moe(dense_layers=dense_layers, step=step)
    model = ours_like(reference, Qwen3MoeForCausalLM, state=split_experts)

    assert_matches_over_paged_kv(reference, model, lengths, prompts, chunk)


def test_the_dense_layers_are_those_transformers_builds() -> None:
    reference = _qwen3_moe(dense_layers=[0], step=2)
    config = ModelConfig.from_hf_config(reference.config.to_dict())

    built = [
        layer
        for layer, block in enumerate(reference.model.layers)
        if not isinstance(
            block.mlp, transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
        )
    ]
    assert list(config.dense_layers) == built == [0, 2]
    assert (config.num_experts, config.experts_per_token, config.moe_intermediate_size) == (
        _EXPERTS,
        _TOP_K,
        20,
    )
