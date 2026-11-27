"""Our Phi-3 against ``transformers``' on tiny randomly initialised ``Phi3Config``s.

Needs torch and the ``hf`` extra. ``transformers`` keeps ``qkv_proj`` and ``gate_up_proj`` fused,
as Phi-3 checkpoints store them, so the state dict is loaded under its own names with nothing
split. Float32 hidden states and logits over a paged KV cache must agree with ``transformers``'
full-sequence forward to ``1e-4``. Configs Ayaka still refuses (``longrope``) are refused with
the reason.
"""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.core.errors import ConfigError  # noqa: E402
from ayaka.models.phi3 import Phi3ForCausalLM  # noqa: E402
from hf_parity import (  # noqa: E402  # needs the importorskip above
    CASE_IDS,
    CASES,
    assert_matches_over_paged_kv,
    ours_like,
    randomise,
)

pytestmark = pytest.mark.integration


def _config(**overrides: object) -> Any:
    fields: dict[str, object] = {
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "original_max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "sliding_window": None,
        "pad_token_id": 0,
        "architectures": ["Phi3ForCausalLM"],
    }
    fields.update(overrides)
    return transformers.Phi3Config(**fields)


@pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize(("lengths", "prompts", "chunk"), CASES, ids=CASE_IDS)
def test_forward_matches_transformers_over_paged_kv(
    tie: bool, lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    reference = randomise(transformers.Phi3ForCausalLM(_config(tie_word_embeddings=tie)))
    model = ours_like(reference, Phi3ForCausalLM)

    assert_matches_over_paged_kv(reference, model, lengths, prompts, chunk)


def test_the_window_of_phi_3_mini_4k_caps_the_context() -> None:
    raw = _config(max_position_embeddings=4096, sliding_window=2047).to_dict()

    config = ModelConfig.from_hf_config(raw)

    assert config.context_length == 2047


def test_partial_rotary_matches_transformers() -> None:
    reference = randomise(transformers.Phi3ForCausalLM(_config(partial_rotary_factor=0.75)))
    model = ours_like(reference, Phi3ForCausalLM)

    assert_matches_over_paged_kv(reference, model, [16], [8], 4)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "max_position_embeddings": 512,
                "rope_parameters": {
                    "rope_type": "longrope",
                    "rope_theta": 10000.0,
                    "short_factor": [1.0] * 8,
                    "long_factor": [2.0] * 8,
                    "original_max_position_embeddings": 128,
                },
            },
            "'longrope' is not implemented",
        ),
    ],
    ids=["longrope"],
)
def test_what_is_not_implemented_is_refused(overrides: dict[str, object], message: str) -> None:
    raw = _config(**overrides).to_dict()

    with pytest.raises(ConfigError, match=message):
        ModelConfig.from_hf_config(raw)
