"""Our GPT-2 against ``transformers``' on a tiny randomly initialised ``GPT2Config``.

Needs torch and the ``hf`` extra (``uv sync --extra cpu --extra hf``). The weights are
re-randomised at a scale that makes every layer matter (the default initialisation is near
zero), the config goes through ``ModelConfig.from_hf_config`` exactly as a downloaded
``config.json`` would, and the weights go through the model's own mapping from the checkpoint
names ``transformers`` gives them. The model then sees each sequence in pieces over a paged KV
cache: float32 hidden states and logits must agree with the full-sequence forward of
``transformers`` to ``1e-4``.
"""

from __future__ import annotations

from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.models.gpt2 import GPT2LMHeadModel  # noqa: E402
from ayaka.ops.backend import load_backend  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402
from gpt2_support import load_hf_state, run_paged  # noqa: E402  # needs the importorskip above

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def reference() -> Any:
    config = transformers.GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_positions=64,
        vocab_size=50,
        bos_token_id=0,
        eos_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        architectures=["GPT2LMHeadModel"],
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.3)
    return model


def _ours(reference: Any) -> GPT2LMHeadModel:
    config = ModelConfig.from_hf_config(reference.config.to_dict())
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    model = GPT2LMHeadModel(config, ops=ops, dtype=DType.FP32)
    load_hf_state(model, dict(reference.state_dict()))
    return model


def test_the_config_of_transformers_is_read_as_the_architecture_we_build(reference: Any) -> None:
    config = ModelConfig.from_hf_config(reference.config.to_dict())

    assert config.architecture == "GPT2LMHeadModel"
    assert (config.num_layers, config.num_attention_heads, config.head_dim) == (2, 4, 8)
    assert config.intermediate_size == 128  # n_inner is None: 4 * n_embd
    assert config.hidden_act == "gelu_new"
    assert config.tie_word_embeddings


@pytest.mark.parametrize(
    ("lengths", "prompts", "chunk"),
    [
        ([13], [6], None),
        ([13], [13], None),
        ([11, 8, 5], [7, 8, 1], 3),
        ([40, 17], [30, 17], 8),
    ],
    ids=["prefill-then-decode", "prefill-only", "three-requests-chunked", "longer-than-blocks"],
)
def test_forward_matches_transformers_over_paged_kv(
    reference: Any, lengths: list[int], prompts: list[int], chunk: int | None
) -> None:
    model = _ours(reference)
    generator = torch.Generator().manual_seed(11)
    sequences = [torch.randint(0, 50, (n,), generator=generator) for n in lengths]

    outputs = run_paged(model, sequences, prompt_lengths=prompts, chunk=chunk)

    for sequence, hidden in zip(sequences, outputs, strict=True):
        with torch.no_grad():
            expected = reference.transformer(sequence[None]).last_hidden_state[0]
            expected_logits = reference(sequence[None]).logits[0]
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            model.compute_logits(hidden), expected_logits, rtol=1e-4, atol=1e-4
        )
