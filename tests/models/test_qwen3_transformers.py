"""Our Qwen3 against ``transformers``' on tiny randomly initialised ``Qwen3Config``s.

Needs torch and the ``hf`` extra. ``head_dim`` is set apart from ``hidden_size``, as in the real
checkpoints, so a model that derived it from the width would fail. The per-head ``q_norm`` and
``k_norm`` weights are randomised like every other norm, so a norm applied over the wrong width,
after the rotary, or not at all changes the result. Float32 hidden states and logits over a paged
KV cache must agree with ``transformers``' full-sequence forward to ``1e-4``.
"""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.models.qwen3 import Qwen3ForCausalLM  # noqa: E402
from hf_parity import (  # noqa: E402  # needs the importorskip above
    CASE_IDS,
    CASES,
    assert_matches_over_paged_kv,
    ours_like,
    randomise,
)

pytestmark = pytest.mark.integration


def _reference(*, tie: bool, bias: bool) -> Any:
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=tie,
        attention_bias=bias,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        architectures=["Qwen3ForCausalLM"],
    )
    return randomise(transformers.Qwen3ForCausalLM(config))


@pytest.mark.parametrize(
    ("tie", "bias"), [(False, False), (True, True)], ids=["untied", "tied-bias"]
)
@pytest.mark.parametrize(("lengths", "prompts", "chunk"), CASES, ids=CASE_IDS)
def test_forward_matches_transformers_over_paged_kv(
    tie: bool, bias: bool, lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    reference = _reference(tie=tie, bias=bias)
    model = ours_like(reference, Qwen3ForCausalLM)

    assert_matches_over_paged_kv(reference, model, lengths, prompts, chunk)


def test_the_config_keeps_head_dim_apart_from_the_width() -> None:
    config = ModelConfig.from_hf_config(_reference(tie=False, bias=False).config.to_dict())

    assert config.architecture == "Qwen3ForCausalLM"
    assert (config.hidden_size, config.num_attention_heads, config.head_dim) == (48, 4, 16)
    assert config.sliding_window is None
