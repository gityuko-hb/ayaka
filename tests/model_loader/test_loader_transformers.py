"""Loading a checkpoint that ``transformers`` itself wrote, and running it against ``transformers``.

Needs torch and the ``hf`` extra. ``save_pretrained`` writes the real ``config.json`` and
``model.safetensors`` of a tiny random GPT-2 (the tied ``lm_head`` is stored once, as in every
published GPT-2), ``load_model`` reads them back, and the forward over a paged KV cache must
match ``transformers`` on the original model.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.device import DeviceConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

from ayaka.model_loader.loader import load_model  # noqa: E402
from gpt2_support import run_paged  # noqa: E402  # needs the importorskip above

if TYPE_CHECKING:
    from pathlib import Path

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
    )
    torch.manual_seed(1)
    model = transformers.GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.3)
    return model


def _compare(model: Any, reference: Any, length: int = 20, prompt: int = 12) -> None:
    sequence = torch.randint(0, 50, (length,), generator=torch.Generator().manual_seed(3))
    [hidden] = run_paged(model, [sequence], prompt_lengths=[prompt], chunk=5)
    with torch.no_grad():
        expected = reference(sequence[None]).logits[0]
    torch.testing.assert_close(model.compute_logits(hidden), expected, rtol=1e-4, atol=1e-4)


def test_a_checkpoint_saved_by_transformers_loads_and_matches_it(
    reference: Any, tmp_path: Path
) -> None:
    reference.save_pretrained(tmp_path, safe_serialization=True)

    model = load_model(ModelSourceConfig(str(tmp_path)), DeviceConfig())

    assert model.config.architecture == "GPT2LMHeadModel"
    _compare(model, reference)


def test_a_sharded_checkpoint_saved_by_transformers(reference: Any, tmp_path: Path) -> None:
    reference.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="20KB")
    assert (tmp_path / "model.safetensors.index.json").is_file()

    model = load_model(ModelSourceConfig(str(tmp_path)), DeviceConfig())

    _compare(model, reference)


def test_a_half_precision_checkpoint_loads_into_float32(reference: Any, tmp_path: Path) -> None:
    half = copy.deepcopy(reference).half()
    half.save_pretrained(tmp_path, safe_serialization=True)

    model = load_model(ModelSourceConfig(str(tmp_path)), DeviceConfig(dtype=DType.FP32))

    assert {p.dtype for p in model.parameters()} == {torch.float32}
    _compare(model, half.float(), length=14, prompt=9)
