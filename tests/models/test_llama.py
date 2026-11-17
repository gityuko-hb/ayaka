"""Llama: parameters, the fused checkpoint layout, and a paged forward that equals the formulas.

The oracle is ``tests/_support/llama_support.llama_reference_hidden``: Llama written out from its
published formulas in float32 torch over a whole sequence, on weights in the layout of a Hugging
Face checkpoint (separate q/k/v and gate/up projections). The model sees the same sequences in
pieces through a paged cache whose blocks are scrambled and start as NaN. The ``transformers``
comparison is in ``test_llama_transformers.py``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from ayaka.configs.model import RopeScaling
from ayaka.core.errors import ModelLoadError
from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.model_loader.loader import load_model  # noqa: E402
from ayaka.models.llama import LlamaForCausalLM  # noqa: E402
from ayaka.models.registry import resolve_model  # noqa: E402
from gpt2_support import load_hf_state, run_paged  # noqa: E402  # needs the importorskip above
from llama_support import (  # noqa: E402
    llama_config,
    llama_reference_hidden,
    llama_state,
    write_llama,
)

pytestmark = pytest.mark.unit


def _model(dtype: DType = DType.FP32, /, **overrides: object) -> LlamaForCausalLM:
    config = llama_config(**overrides)
    ops = load_backend(["torch"], device="cpu", dtype=dtype)
    return LlamaForCausalLM(config, ops=ops, dtype=dtype)


def _tokens(length: int, seed: int, vocab: int = 64) -> Any:
    return torch.randint(0, vocab, (length,), generator=torch.Generator().manual_seed(seed))


class TestParameters:
    def test_names_and_shapes_are_the_fused_ones(self) -> None:
        model = _model()

        shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}

        assert shapes["embed_tokens.weight"] == (64, 32)
        assert shapes["layers.0.attn.qkv_proj.weight"] == ((4 + 2 * 2) * 8, 32)
        assert shapes["layers.0.attn.o_proj.weight"] == (32, 32)
        assert shapes["layers.1.mlp.gate_up_proj.weight"] == (96, 32)
        assert shapes["layers.1.mlp.down_proj.weight"] == (32, 48)
        assert shapes["lm_head.weight"] == (64, 32)
        assert "layers.0.attn.qkv_proj.bias" not in shapes
        assert len(shapes) == 3 + 2 * 6  # embed, norm, head; two blocks of six

    def test_the_checkpoint_specs_list_the_parts(self) -> None:
        specs = {spec.name: spec.shape for spec in _model().checkpoint_specs()}

        assert specs["layers.0.attn.q_proj.weight"] == (32, 32)
        assert specs["layers.0.attn.k_proj.weight"] == (16, 32)
        assert specs["layers.0.attn.v_proj.weight"] == (16, 32)
        assert specs["layers.0.mlp.gate_proj.weight"] == (48, 32)
        assert specs["layers.0.mlp.up_proj.weight"] == (48, 32)
        assert "layers.0.attn.qkv_proj.weight" not in specs
        assert "layers.0.mlp.gate_up_proj.weight" not in specs

    def test_a_part_fills_its_rows(self) -> None:
        model = _model()
        model.get_parameter("layers.0.attn.qkv_proj.weight").data.zero_()

        model.load_weight("layers.0.attn.k_proj.weight", torch.ones(16, 32))

        fused = model.get_parameter("layers.0.attn.qkv_proj.weight")
        assert torch.count_nonzero(fused[:32]) == 0
        assert torch.equal(fused[32:48], torch.ones(16, 32))
        assert torch.count_nonzero(fused[48:]) == 0

    def test_a_part_of_the_wrong_shape_is_refused(self) -> None:
        with pytest.raises(ModelLoadError, match=r"k_proj\.weight \(rows 32\.\.48"):
            _model().load_weight("layers.0.attn.k_proj.weight", torch.ones(8, 32))

    def test_tied_embeddings_share_the_table(self) -> None:
        model = _model(tie_word_embeddings=True)

        assert model.lm_head.weight is model.embed_tokens.weight

    def test_every_layer_shares_one_rotary_table(self) -> None:
        model = _model()

        assert all(layer.attn.rotary is model.rotary for layer in cast("Any", model.layers))
        assert len(list(model.buffers())) == 1

    def test_the_registry_resolves_llama(self) -> None:
        assert resolve_model("LlamaForCausalLM") is LlamaForCausalLM

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"hidden_act": "gelu"}, "hidden_act is 'gelu'"),
            ({"rope_theta": None}, "no rope_theta"),
        ],
    )
    def test_configs_it_cannot_run_are_refused(
        self, overrides: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ModelLoadError, match=message):
            _model(**overrides)


class TestForward:
    @pytest.mark.parametrize(
        ("lengths", "prompts", "chunk"),
        [([13], [6], None), ([11, 8, 5], [7, 8, 1], 3), ([40, 17], [30, 17], 8)],
        ids=["prefill-then-decode", "three-requests-chunked", "longer-than-blocks"],
    )
    def test_paged_steps_equal_the_formulas(
        self, lengths: list[int], prompts: list[int], chunk: int | None
    ) -> None:
        model = _model()
        state = llama_state(model.config, seed=3)
        load_hf_state(model, state)
        sequences = [_tokens(n, seed=i) for i, n in enumerate(lengths)]

        outputs = run_paged(model, sequences, prompt_lengths=prompts, chunk=chunk)

        for sequence, hidden in zip(sequences, outputs, strict=True):
            expected = llama_reference_hidden(state, model.config, sequence)
            torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "scaling",
        [
            RopeScaling(kind="linear", factor=2.0),
            RopeScaling(
                kind="llama3",
                factor=8.0,
                original_max_position_embeddings=16,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
            ),
        ],
        ids=["linear", "llama3"],
    )
    def test_scaled_rotary_positions_equal_the_formulas(self, scaling: RopeScaling) -> None:
        model = _model(rope_scaling=scaling, head_dim=16, num_attention_heads=2, num_kv_heads=1)
        state = llama_state(model.config, seed=5)
        load_hf_state(model, state)
        sequence = _tokens(30, seed=9)

        [hidden] = run_paged(model, [sequence], prompt_lengths=[20], chunk=7)

        expected = llama_reference_hidden(state, model.config, sequence)
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)

    def test_attention_biases_are_loaded_and_used(self) -> None:
        model = _model(attention_bias=True)
        state = llama_state(model.config, seed=7)
        load_hf_state(model, state)
        sequence = _tokens(9, seed=2)

        [hidden] = run_paged(model, [sequence], prompt_lengths=[5])

        expected = llama_reference_hidden(state, model.config, sequence)
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)


class TestLoading:
    def test_the_loader_builds_and_fills_llama_from_a_checkpoint(self, tmp_path: Any) -> None:
        config = llama_config(tie_word_embeddings=True)
        state = llama_state(config, seed=1)
        write_llama(tmp_path, state, config)

        model = load_model(
            ModelSourceConfig(str(tmp_path), offline=True), DeviceConfig(dtype=DType.FP32)
        )

        assert isinstance(model, LlamaForCausalLM)
        assert model.config.rope_theta == 10000.0
        sequence = _tokens(12, seed=4)
        [hidden] = run_paged(model, [sequence], prompt_lengths=[8])
        expected = llama_reference_hidden(state, config, sequence)
        torch.testing.assert_close(hidden, expected, rtol=1e-4, atol=1e-4)
        logits = model.compute_logits(hidden)
        torch.testing.assert_close(
            logits, expected @ state["model.embed_tokens.weight"].T, rtol=1e-4, atol=1e-4
        )

    def test_a_checkpoint_missing_a_part_lists_it(self, tmp_path: Any) -> None:
        config = llama_config()
        state = llama_state(config)
        del state["model.layers.1.self_attn.v_proj.weight"]
        write_llama(tmp_path, state, config)

        with pytest.raises(ModelLoadError, match=r"missing 1 parameters: layers\.1\.attn\.v_proj"):
            load_model(
                ModelSourceConfig(str(tmp_path), offline=True), DeviceConfig(dtype=DType.FP32)
            )

    def test_the_rotary_table_is_computed_after_the_meta_build(self, tmp_path: Any) -> None:
        config = llama_config()
        write_llama(tmp_path, llama_state(config), config)

        model = load_model(
            ModelSourceConfig(str(tmp_path), offline=True), DeviceConfig(dtype=DType.FP32)
        )

        reference = _model().rotary.cache
        assert torch.equal(cast("Any", model).rotary.cache, reference)
