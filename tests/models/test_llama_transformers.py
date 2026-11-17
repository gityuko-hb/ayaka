"""Our Llama against ``transformers``' on tiny randomly initialised ``LlamaConfig``s.

Needs torch and the ``hf`` extra (``uv sync --extra cpu --extra hf``). The weights are
re-randomised at a scale that makes every layer matter, the config goes through
``ModelConfig.from_hf_config`` as a downloaded ``config.json`` would (rotary scaling and grouped
heads included), and the weights go through the model's own mapping from the names
``transformers`` gives them, so the fused projections are split exactly as a checkpoint is. The
model then sees each sequence in pieces over a paged KV cache: float32 hidden states and logits
must agree with ``transformers``' full-sequence forward to ``1e-4``.
"""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.models.llama import LlamaForCausalLM  # noqa: E402
from ayaka.ops.backend import load_backend  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402
from gpt2_support import load_hf_state, run_paged  # noqa: E402  # needs the importorskip above

pytestmark = pytest.mark.integration

ROPE = {
    "default": {"rope_type": "default", "rope_theta": 10000.0},
    "llama3": {
        "rope_type": "llama3",
        "rope_theta": 500000.0,
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 16,
    },
    "linear": {"rope_type": "linear", "rope_theta": 10000.0, "factor": 2.0},
}


def _reference(rope: dict[str, object], *, tie: bool, bias: bool) -> Any:
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=tie,
        attention_bias=bias,
        rope_parameters=rope,
        architectures=["LlamaForCausalLM"],
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
            else:
                parameter.normal_(0.0, 0.3)
    return model.float()


def _ours(reference: Any) -> LlamaForCausalLM:
    config = ModelConfig.from_hf_config(reference.config.to_dict())
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    model = LlamaForCausalLM(config, ops=ops, dtype=DType.FP32)
    load_hf_state(model, dict(reference.state_dict()))
    return model


@pytest.mark.parametrize("rope", list(ROPE), ids=list(ROPE))
@pytest.mark.parametrize(
    ("tie", "bias"), [(False, False), (True, True)], ids=["untied", "tied-bias"]
)
@pytest.mark.parametrize(
    ("lengths", "prompts", "chunk"),
    [([13], [6], None), ([11, 8, 5], [7, 8, 1], 3), ([60, 17], [40, 17], 8)],
    ids=["prefill-then-decode", "three-requests-chunked", "past-the-trained-context"],
)
def test_forward_matches_transformers_over_paged_kv(
    rope: str,
    tie: bool,
    bias: bool,
    lengths: list[int],
    prompts: list[int],
    chunk: int | None,
) -> None:
    reference = _reference(ROPE[rope], tie=tie, bias=bias)
    model = _ours(reference)
    generator = torch.Generator().manual_seed(11)
    sequences = [torch.randint(0, 64, (n,), generator=generator) for n in lengths]

    outputs = run_paged(model, sequences, prompt_lengths=prompts, chunk=chunk)

    for sequence, hidden in zip(sequences, outputs, strict=True):
        with torch.no_grad():
            expected = reference.model(sequence[None]).last_hidden_state[0]
            expected_logits = reference(sequence[None]).logits[0]
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            model.compute_logits(hidden), expected_logits, rtol=1e-4, atol=1e-4
        )


def test_the_config_of_transformers_is_read_as_the_architecture_we_build() -> None:
    config = ModelConfig.from_hf_config(
        _reference(ROPE["llama3"], tie=False, bias=False).config.to_dict()
    )

    assert config.architecture == "LlamaForCausalLM"
    assert (config.num_attention_heads, config.num_kv_heads, config.head_dim) == (4, 2, 16)
    assert config.rope_theta == 500000.0
    assert config.rope_scaling is not None
    assert config.rope_scaling.kind == "llama3"
    assert not config.tie_word_embeddings
