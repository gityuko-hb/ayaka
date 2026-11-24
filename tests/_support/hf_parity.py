"""Our models against ``transformers``' on tiny random configs, through a paged KV cache.

The helpers here are what every architecture's parity test does the same way:

* :func:`randomise` re-draws a ``transformers`` model's weights at a scale that makes every layer
  matter (norm weights near one, everything else ``N(0, 0.3)``);
* :func:`ours_like` builds our model from the reference's ``config.to_dict()``, as a downloaded
  ``config.json`` is read, and loads the reference's state dict through our model's own mapping;
* :func:`assert_matches_over_paged_kv` feeds sequences to our model in pieces over a scrambled,
  NaN-filled paged cache and compares hidden states and logits with the reference's
  full-sequence forward.

Import this module only from a test that has already done ``pytest.importorskip("torch")`` and
``pytest.importorskip("transformers")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import torch

from ayaka.configs.model import ModelConfig
from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType
from gpt2_support import load_hf_state, run_paged

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayaka.models.base import CausalLM

__all__ = ["CASES", "CASE_IDS", "assert_matches_over_paged_kv", "ours_like", "randomise"]

#: ``(lengths, prompt lengths, chunk)`` of the paged runs, and their test ids: one request
#: prefilled then decoded; three requests in chunks; two requests longer than several blocks.
CASES: Final = (
    ([13], [6], None),
    ([11, 8, 5], [7, 8, 1], 3),
    ([60, 17], [40, 17], 8),
)
CASE_IDS: Final = ("prefill-then-decode", "three-requests-chunked", "several-blocks")


def randomise(model: Any, *, seed: int = 0, scale: float = 0.3) -> Any:
    """Re-draw every parameter of a ``transformers`` model and return it in float32, eval mode."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
            else:
                parameter.normal_(0.0, scale)
    return model.float().eval()


def ours_like(
    reference: Any,
    model_class: type[CausalLM],
    *,
    state: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
) -> CausalLM:
    """Our ``model_class`` in float32 on the CPU, with the reference's config and weights.

    ``state`` rewrites the reference's state dict into the names a checkpoint on disk uses, for
    the architectures whose ``transformers`` module keeps parameters in another layout.
    """
    config = ModelConfig.from_hf_config(reference.config.to_dict())
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    model = model_class(config, ops=ops, dtype=DType.FP32)
    tensors = dict(reference.state_dict())
    load_hf_state(model, tensors if state is None else state(tensors))
    return model


def assert_matches_over_paged_kv(
    reference: Any,
    model: CausalLM,
    lengths: Sequence[int],
    prompts: Sequence[int],
    chunk: int | None,
    *,
    tolerance: float = 1e-4,
    seed: int = 11,
) -> None:
    """Hidden states and logits of every position equal the reference's to ``tolerance``."""
    generator = torch.Generator().manual_seed(seed)
    vocab = model.config.vocab_size
    sequences = [torch.randint(0, vocab, (n,), generator=generator) for n in lengths]

    outputs = run_paged(model, sequences, prompt_lengths=prompts, chunk=chunk)

    for sequence, hidden in zip(sequences, outputs, strict=True):
        with torch.no_grad():
            expected = reference.model(sequence[None]).last_hidden_state[0]
            expected_logits = reference(sequence[None]).logits[0]
        torch.testing.assert_close(hidden, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            model.compute_logits(hidden), expected_logits, rtol=tolerance, atol=tolerance
        )
