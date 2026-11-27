"""GPT-2: parameters, checkpoint layout, and a forward over paged KV that equals the formulas.

The oracle is ``tests/_support/gpt2_support.reference_hidden``: GPT-2 written out from its
published formulas in float32 torch, run over a whole sequence at once, on weights in the layout
of a Hugging Face checkpoint (``Conv1D``, ``[in, out]``). The model sees the same sequences in
pieces (a prefill chunk, then one token per step) through a paged cache whose blocks are in
scrambled order and start as NaN. The real ``transformers`` comparison is in
``test_gpt2_transformers.py``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from ayaka.core.errors import ModelLoadError
from ayaka.core.types import KVLayoutKind
from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.models.base import CausalLM  # noqa: E402
from ayaka.models.gpt2 import GPT2LMHeadModel  # noqa: E402
from ayaka.weights.spec import TensorSpec  # noqa: E402
from gpt2_support import (  # noqa: E402  # needs the importorskip above
    BLOCK_SIZE,
    hf_state,
    load_hf_state,
    reference_hidden,
    run_paged,
    tiny_config,
)

pytestmark = pytest.mark.unit


def _model(dtype: DType = DType.FP32, /, **overrides: object) -> GPT2LMHeadModel:
    config = tiny_config(**overrides)
    ops = load_backend(["torch"], device="cpu", dtype=dtype)
    return GPT2LMHeadModel(config, ops=ops, dtype=dtype)


def _tokens(length: int, seed: int, vocab: int = 50) -> Any:
    return torch.randint(0, vocab, (length,), generator=torch.Generator().manual_seed(seed))


class TestParameters:
    def test_names_and_shapes(self) -> None:
        model = _model()

        shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}

        assert shapes["wte.weight"] == (50, 32)
        assert shapes["wpe.weight"] == (64, 32)
        assert shapes["ln_f.weight"] == (32,)
        assert shapes["layers.1.attn.qkv_proj.weight"] == (96, 32)
        assert shapes["layers.1.attn.qkv_proj.bias"] == (96,)
        assert shapes["layers.0.attn.o_proj.weight"] == (32, 32)
        assert shapes["layers.0.mlp.up_proj.weight"] == (128, 32)
        assert shapes["layers.0.mlp.down_proj.weight"] == (32, 128)
        assert len(shapes) == 2 + 2 + 2 * 12  # embeddings, ln_f, two blocks of 12

    def test_the_head_shares_the_token_table(self) -> None:
        model = _model()

        assert model.lm_head.weight is model.wte.weight
        assert "lm_head.weight" not in dict(model.named_parameters())

    def test_an_untied_head_has_its_own_weight(self) -> None:
        model = _model(tie_word_embeddings=False)

        assert model.lm_head.weight is not model.wte.weight
        assert tuple(dict(model.named_parameters())["lm_head.weight"].shape) == (50, 32)

    def test_parameters_have_the_compute_dtype_and_no_gradient(self) -> None:
        model = _model(DType.BF16)

        assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
        assert not any(p.requires_grad for p in model.parameters())

    def test_it_can_be_built_on_the_meta_device_without_memory(self) -> None:
        config = tiny_config()
        ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)

        model = GPT2LMHeadModel(config, ops=ops, dtype=DType.FP32, device="meta")

        assert {p.device.type for p in model.parameters()} == {"meta"}
        assert model.lm_head.weight is model.wte.weight

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"num_kv_heads": 2}, "multi-head attention"),
            ({"head_dim": 4}, "not hidden_size 32"),
            ({"hidden_act": "silu"}, "hidden_act 'silu'"),
        ],
    )
    def test_a_config_it_cannot_run_is_refused(
        self, overrides: dict[str, object], fragment: str
    ) -> None:
        with pytest.raises(ModelLoadError, match=re.escape(fragment)):
            _model(**overrides)

    def test_the_gelu_variants_it_knows(self) -> None:
        for act in ("gelu_new", "gelu_pytorch_tanh", "gelu"):
            assert _model(hidden_act=act).config.hidden_act == act


class TestCheckpointLayout:
    def test_projections_are_listed_with_the_shape_the_checkpoint_stores(self) -> None:
        specs = {spec.name: spec for spec in _model().checkpoint_specs()}

        assert specs["layers.0.attn.qkv_proj.weight"].shape == (32, 96)
        assert specs["layers.0.attn.o_proj.weight"].shape == (32, 32)
        assert specs["layers.0.mlp.up_proj.weight"].shape == (32, 128)
        assert specs["layers.0.mlp.down_proj.weight"].shape == (128, 32)
        assert specs["layers.0.attn.qkv_proj.bias"].shape == (96,)
        assert specs["wte.weight"].shape == (50, 32)
        assert all(isinstance(spec, TensorSpec) for spec in specs.values())

    def test_a_tied_head_is_listed_once(self) -> None:
        names = [spec.name for spec in _model().checkpoint_specs()]

        assert "wte.weight" in names
        assert "lm_head.weight" not in names

    @pytest.mark.parametrize("prefix", ["transformer.", ""])
    @pytest.mark.parametrize("masks", [False, True])
    @pytest.mark.parametrize("head", [False, True])
    def test_every_tensor_of_a_hugging_face_state_dict_is_claimed_and_every_parameter_filled(
        self, prefix: str, masks: bool, head: bool
    ) -> None:
        model = _model()
        state = hf_state(model.config, prefix=prefix, with_masks=masks, with_head=head)
        mapping = GPT2LMHeadModel.weight_mapping(model.config)

        targets = {name: mapping.target(name) for name in state}
        filled = [target for target in targets.values() if target is not None]
        ignored = sorted(name for name, target in targets.items() if target is None)

        assert sorted(filled) == sorted(spec.name for spec in model.checkpoint_specs())
        assert len(set(filled)) == len(filled)
        expected_ignored = [n for n in state if n.endswith(".attn.bias") or n == "lm_head.weight"]
        assert ignored == sorted(expected_ignored)

    def test_an_untied_head_is_not_ignored(self) -> None:
        config = tiny_config(tie_word_embeddings=False)
        mapping = GPT2LMHeadModel.weight_mapping(config)

        assert mapping.target("lm_head.weight") == "lm_head.weight"

    def test_a_tensor_the_model_does_not_have_is_not_renamed_away(self) -> None:
        mapping = GPT2LMHeadModel.weight_mapping(tiny_config())

        assert mapping.target("transformer.h.0.attn.c_attn.scales") == (
            "transformer.h.0.attn.c_attn.scales"
        )
        assert mapping.target("score.weight") == "score.weight"

    def test_load_weight_transposes_a_conv1d_weight_and_converts_the_dtype(self) -> None:
        model = _model(DType.BF16)
        stored = torch.randn(32, 96)

        model.load_weight("layers.0.attn.qkv_proj.weight", stored)

        parameter = model.get_parameter("layers.0.attn.qkv_proj.weight")
        assert parameter.dtype == torch.bfloat16
        assert torch.equal(parameter, stored.t().to(torch.bfloat16))

    def test_load_weight_leaves_a_bias_and_an_embedding_as_stored(self) -> None:
        model = _model()
        bias, table = torch.randn(96), torch.randn(50, 32)

        model.load_weight("layers.1.attn.qkv_proj.bias", bias)
        model.load_weight("wte.weight", table)

        assert torch.equal(model.get_parameter("layers.1.attn.qkv_proj.bias"), bias)
        assert torch.equal(model.get_parameter("wte.weight"), table)
        assert torch.equal(model.lm_head.weight, table)

    def test_load_weight_refuses_a_tensor_in_the_parameters_own_layout(self) -> None:
        model = _model()

        with pytest.raises(
            ModelLoadError, match=r"parameter layers.0.attn.qkv_proj.weight is \[96, 32\]"
        ):
            model.load_weight("layers.0.attn.qkv_proj.weight", torch.zeros(96, 32))


class TestForward:
    @pytest.mark.parametrize("prefix", ["transformer.", ""])
    def test_prefill_then_decode_over_paged_kv_equals_the_reference(self, prefix: str) -> None:
        model = _model()
        state = hf_state(model.config, seed=3, prefix=prefix)
        load_hf_state(model, state)
        sequence = _tokens(13, seed=1)

        [hidden] = run_paged(model, [sequence], prompt_lengths=[6])

        assert hidden.shape == (13, 32)
        torch.testing.assert_close(
            hidden, reference_hidden(state, model.config, sequence), rtol=1e-4, atol=1e-4
        )

    def test_several_requests_with_chunked_prefill_decode_together(self) -> None:
        model = _model()
        state = hf_state(model.config, seed=4)
        load_hf_state(model, state)
        sequences = [_tokens(11, seed=2), _tokens(8, seed=3), _tokens(5, seed=4)]

        outputs = run_paged(model, sequences, prompt_lengths=[7, 8, 1], chunk=3)

        for sequence, hidden in zip(sequences, outputs, strict=True):
            torch.testing.assert_close(
                hidden, reference_hidden(state, model.config, sequence), rtol=1e-4, atol=1e-4
            )

    def test_logits_are_the_hidden_states_times_the_token_table(self) -> None:
        model = _model()
        state = hf_state(model.config, seed=5)
        load_hf_state(model, state)
        sequence = _tokens(9, seed=5)
        [hidden] = run_paged(model, [sequence], prompt_lengths=[9])

        logits = model.compute_logits(hidden[-1:])

        assert logits.shape == (1, 50)
        torch.testing.assert_close(
            logits, hidden[-1:] @ state["transformer.wte.weight"].t(), rtol=1e-5, atol=1e-5
        )

    def test_an_untied_head_is_used_for_the_logits(self) -> None:
        model = _model(tie_word_embeddings=False)
        state = hf_state(model.config, seed=6, with_head=True)
        state["lm_head.weight"] = torch.randn(50, 32)
        load_hf_state(model, state)
        hidden = torch.randn(3, 32)

        logits = model.compute_logits(hidden)

        torch.testing.assert_close(
            logits, hidden @ state["lm_head.weight"].t(), rtol=1e-5, atol=1e-5
        )

    def test_positions_come_from_the_context_not_from_the_row_index(self) -> None:
        # Decoding position 5 alone, after a prefill of 5, must use wpe[5], not wpe[0].
        model = _model()
        state = hf_state(model.config, seed=7)
        load_hf_state(model, state)
        sequence = _tokens(6, seed=6)

        [hidden] = run_paged(model, [sequence], prompt_lengths=[5])

        torch.testing.assert_close(
            hidden[5], reference_hidden(state, model.config, sequence)[5], rtol=1e-4, atol=1e-4
        )

    def test_the_output_is_in_the_compute_dtype(self) -> None:
        model = _model(DType.BF16)
        load_hf_state(model, hf_state(model.config, seed=8))

        [hidden] = run_paged(model, [_tokens(5, seed=7)], prompt_lengths=[5])

        assert hidden.dtype == torch.bfloat16
        assert torch.isfinite(hidden.float()).all()


class TestKVSpecs:
    @pytest.mark.parametrize("layout", [KVLayoutKind.NHD, KVLayoutKind.HND])
    def test_one_group_of_every_layer(self, layout: KVLayoutKind) -> None:
        model = _model(DType.FP16)

        [group] = model.kv_specs(block_size=BLOCK_SIZE, layout=layout)
        from ayaka.kvcache.storage.spec import KVGroup

        assert isinstance(group, KVGroup)
        assert group.layer_ids == (0, 1)
        assert group.spec.num_kv_heads == 4
        assert group.spec.head_dim == 8
        assert group.spec.block_size == BLOCK_SIZE
        assert group.spec.dtype is DType.FP16
        assert group.spec.layout is layout


def test_the_contract_is_abstract() -> None:
    config = tiny_config()
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)

    with pytest.raises(TypeError, match="abstract"):
        CausalLM(config, ops=ops, dtype=DType.FP32)  # type: ignore[abstract]  # that is the test
