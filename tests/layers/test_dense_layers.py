"""Linear, norm, activation, embedding and MLP layers against the matching ``torch.nn`` modules.

The reference is the ``torch.nn`` module (or function) in float32, with its parameters set to
the same values as the layer's after rounding them to the layer's dtype, run on the same
rounded input. The layer rounds each op's result to its dtype once, so the expected gap is
that rounding per op: the bound is ``factor * eps * max|y|`` with ``eps = finfo(dtype).eps``
and ``factor`` the number of rounding steps between input and output (one for a single op,
a few more for a block of three).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast

import pytest

from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from torch import nn  # noqa: E402  # needs the importorskip above

from ayaka.layers.activation import GELU, GELUAndMul, SiLUAndMul  # noqa: E402
from ayaka.layers.embedding.token import LMHead, TokenEmbedding  # noqa: E402
from ayaka.layers.ffn.dense import MLP, GatedMLP  # noqa: E402
from ayaka.layers.linear import Linear  # noqa: E402
from ayaka.layers.norm import LayerNorm, RMSNorm  # noqa: E402

if TYPE_CHECKING:
    from ayaka.ops.backend import OpsBackend

pytestmark = pytest.mark.unit

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DTYPE_IDS = ["fp16", "bf16", "fp32"]
_REGISTRY_DTYPE = {torch.float16: DType.FP16, torch.bfloat16: DType.BF16, torch.float32: DType.FP32}


def _ops(dtype: Any) -> OpsBackend:
    return load_backend(["torch"], device="cpu", dtype=_REGISTRY_DTYPE[dtype])


def _fill(module: nn.Module, seed: int = 0) -> None:
    """Give every parameter random values, as a loader would."""
    generator = torch.Generator().manual_seed(seed)
    for parameter in module.parameters():
        parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)


def _randn(*shape: int, dtype: Any, seed: int = 1) -> Any:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).to(dtype)


def _close(actual: Any, expected: Any, dtype: Any, *, factor: float = 2.0) -> None:
    expected = expected.detach()
    eps = torch.finfo(dtype).eps
    tolerance = factor * eps * max(float(expected.abs().max()), 1.0)
    torch.testing.assert_close(actual.float(), expected, rtol=0.0, atol=tolerance)


def _mirror(layer: nn.Module, reference: nn.Module) -> None:
    """Copy the layer's (dtype-rounded) parameters into a float32 reference of the same names."""
    reference.load_state_dict({name: tensor.float() for name, tensor in layer.state_dict().items()})


class TestConstruction:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_parameters_have_the_requested_dtype_and_hold_no_gradient(self, dtype: Any) -> None:
        layer = Linear(8, 4, ops=_ops(dtype), dtype=dtype, device="cpu")

        assert [(name, p.dtype, p.requires_grad) for name, p in layer.named_parameters()] == [
            ("weight", dtype, False),
            ("bias", dtype, False),
        ]

    def test_names_and_shapes_match_torch_nn(self) -> None:
        ops, dtype = _ops(torch.float32), torch.float32
        pairs: list[tuple[nn.Module, nn.Module]] = [
            (Linear(6, 3, ops=ops, dtype=dtype), nn.Linear(6, 3)),
            (Linear(6, 3, bias=False, ops=ops, dtype=dtype), nn.Linear(6, 3, bias=False)),
            (LayerNorm(6, eps=1e-5, ops=ops, dtype=dtype), nn.LayerNorm(6)),
            (LayerNorm(6, eps=1e-5, bias=False, ops=ops, dtype=dtype), nn.LayerNorm(6, bias=False)),
            (RMSNorm(6, eps=1e-6, ops=ops, dtype=dtype), nn.RMSNorm(6)),
            (TokenEmbedding(10, 6, ops=ops, dtype=dtype), nn.Embedding(10, 6)),
            (LMHead(6, 10, ops=ops, dtype=dtype), nn.Linear(6, 10, bias=False)),
        ]

        for layer, reference in pairs:
            assert {k: v.shape for k, v in layer.state_dict().items()} == {
                k: v.shape for k, v in reference.state_dict().items()
            }

    def test_a_layer_without_a_bias_has_none(self) -> None:
        layer = Linear(4, 2, bias=False, ops=_ops(torch.float32), dtype=torch.float32)

        assert layer.bias is None
        assert [name for name, _ in layer.named_parameters()] == ["weight"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestLinear:
    @pytest.mark.parametrize("bias", [True, False])
    def test_matches_nn_linear(self, dtype: Any, bias: bool) -> None:
        layer = Linear(32, 20, bias=bias, ops=_ops(dtype), dtype=dtype)
        _fill(layer)
        x = _randn(3, 5, 32, dtype=dtype)
        reference = nn.Linear(32, 20, bias=bias)
        _mirror(layer, reference)

        out = layer(x)

        assert out.shape == (3, 5, 20)
        assert out.dtype == dtype
        _close(out, reference(x.float()), dtype, factor=4.0)

    def test_an_empty_batch_gives_an_empty_result(self, dtype: Any) -> None:
        layer = Linear(8, 6, ops=_ops(dtype), dtype=dtype)

        assert layer(torch.empty(0, 8, dtype=dtype)).shape == (0, 6)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestNorm:
    def test_rms_norm_matches_nn_rms_norm(self, dtype: Any) -> None:
        layer = RMSNorm(64, eps=1e-6, ops=_ops(dtype), dtype=dtype)
        _fill(layer)
        x = _randn(3, 5, 64, dtype=dtype)
        reference = nn.RMSNorm(64, eps=1e-6)
        _mirror(layer, reference)

        _close(layer(x), reference(x.float()), dtype)

    @pytest.mark.parametrize("bias", [True, False])
    def test_layer_norm_matches_nn_layer_norm(self, dtype: Any, bias: bool) -> None:
        layer = LayerNorm(48, eps=1e-5, bias=bias, ops=_ops(dtype), dtype=dtype)
        _fill(layer)
        x = _randn(3, 5, 48, dtype=dtype)
        reference = nn.LayerNorm(48, eps=1e-5, bias=bias)
        _mirror(layer, reference)

        _close(layer(x), reference(x.float()), dtype, factor=4.0)

    def test_fused_add_updates_the_residual_and_the_input(self, dtype: Any) -> None:
        layer = RMSNorm(64, eps=1e-6, ops=_ops(dtype), dtype=dtype)
        _fill(layer)
        x, residual = _randn(4, 64, dtype=dtype), _randn(4, 64, dtype=dtype, seed=2)
        reference = nn.RMSNorm(64, eps=1e-6)
        _mirror(layer, reference)
        total = x.float() + residual.float()

        layer.fused_add(x, residual)

        _close(residual, total, dtype)
        _close(x, reference(total), dtype)

    def test_forward_does_not_modify_its_input(self, dtype: Any) -> None:
        layer = RMSNorm(16, eps=1e-6, ops=_ops(dtype), dtype=dtype)
        _fill(layer)
        x = _randn(2, 16, dtype=dtype)
        before = x.clone()

        out = layer(x)

        assert torch.equal(x, before)
        assert out.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestActivation:
    @pytest.mark.parametrize("tanh", [False, True])
    def test_gelu_matches_nn_gelu(self, dtype: Any, tanh: bool) -> None:
        x = _randn(6, 33, dtype=dtype) * 3

        out = GELU(tanh=tanh, ops=_ops(dtype))(x)

        expected = nn.GELU(approximate="tanh" if tanh else "none")(x.float())
        _close(out, expected, dtype)

    def test_silu_and_mul_matches_silu_times_the_second_half(self, dtype: Any) -> None:
        x = _randn(3, 4, 40, dtype=dtype)

        out = SiLUAndMul(ops=_ops(dtype))(x)

        a, b = x.float().chunk(2, dim=-1)
        assert out.shape == (3, 4, 20)
        _close(out, nn.SiLU()(a) * b, dtype, factor=4.0)

    @pytest.mark.parametrize("tanh", [False, True])
    def test_gelu_and_mul_matches_gelu_times_the_second_half(self, dtype: Any, tanh: bool) -> None:
        x = _randn(5, 24, dtype=dtype)

        out = GELUAndMul(tanh=tanh, ops=_ops(dtype))(x)

        a, b = x.float().chunk(2, dim=-1)
        _close(out, nn.GELU(approximate="tanh" if tanh else "none")(a) * b, dtype, factor=4.0)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestEmbedding:
    def test_lookup_matches_nn_embedding_exactly(self, dtype: Any) -> None:
        layer = TokenEmbedding(11, 8, ops=_ops(dtype), dtype=dtype)
        _fill(layer)
        reference = nn.Embedding(11, 8)
        reference.load_state_dict({"weight": layer.weight.detach().clone().float()})
        ids = torch.tensor([[0, 10, 3], [3, 3, 7]])

        out = layer(ids)

        assert out.dtype == dtype
        assert torch.equal(out.float(), reference(ids))

    def test_lm_head_matches_nn_linear_without_a_bias(self, dtype: Any) -> None:
        head = LMHead(32, 50, ops=_ops(dtype), dtype=dtype)
        _fill(head)
        x = _randn(4, 32, dtype=dtype)
        reference = nn.Linear(32, 50, bias=False)
        _mirror(head, reference)

        _close(head(x), reference(x.float()), dtype, factor=4.0)

    def test_tied_head_uses_the_embedding_table(self, dtype: Any) -> None:
        ops = _ops(dtype)
        embedding, head = (
            TokenEmbedding(20, 16, ops=ops, dtype=dtype),
            LMHead(16, 20, ops=ops, dtype=dtype),
        )
        _fill(embedding)

        head.tie_weights(embedding)

        assert head.weight is embedding.weight
        assert len(list(nn.ModuleList([embedding, head]).parameters())) == 1
        x = _randn(3, 16, dtype=dtype)
        _close(head(x), x.float() @ embedding.weight.float().T, dtype, factor=4.0)


class TestTieWeights:
    def test_a_table_of_another_shape_is_refused(self) -> None:
        ops, dtype = _ops(torch.float32), torch.float32

        with pytest.raises(ValueError, match="cannot tie"):
            LMHead(16, 20, ops=ops, dtype=dtype).tie_weights(
                TokenEmbedding(21, 16, ops=ops, dtype=dtype)
            )

    def test_a_table_of_another_dtype_is_refused(self) -> None:
        ops = _ops(torch.float32)

        with pytest.raises(ValueError, match=r"cannot tie a torch\.float32 head"):
            LMHead(16, 20, ops=ops, dtype=torch.float32).tie_weights(
                TokenEmbedding(20, 16, ops=ops, dtype=torch.float16)
            )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestMLP:
    @pytest.mark.parametrize("tanh", [False, True])
    def test_matches_a_sequential_of_nn_modules(self, dtype: Any, tanh: bool) -> None:
        layer = MLP(32, 64, tanh=tanh, ops=_ops(dtype), dtype=dtype)
        _fill(layer)
        x = _randn(3, 5, 32, dtype=dtype)
        reference = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(approximate="tanh" if tanh else "none"), nn.Linear(64, 32)
        )
        state = layer.state_dict()
        reference.load_state_dict(
            {
                "0.weight": state["up_proj.weight"].float(),
                "0.bias": state["up_proj.bias"].float(),
                "2.weight": state["down_proj.weight"].float(),
                "2.bias": state["down_proj.bias"].float(),
            }
        )

        out = layer(x)

        assert out.shape == x.shape
        _close(out, reference(x.float()), dtype, factor=16.0)

    @pytest.mark.parametrize(
        ("activation", "reference_activation"),
        [
            ("silu", nn.SiLU()),
            ("gelu", nn.GELU()),
            ("gelu_tanh", nn.GELU(approximate="tanh")),
        ],
    )
    def test_gated_mlp_matches_separate_gate_and_up_projections(
        self, dtype: Any, activation: Any, reference_activation: nn.Module
    ) -> None:
        layer = GatedMLP(32, 48, activation=activation, ops=_ops(dtype), dtype=dtype)
        _fill(layer)
        x = _randn(3, 5, 32, dtype=dtype)
        gate_up = layer.gate_up_proj.weight.float()
        gate, up = nn.Linear(32, 48, bias=False), nn.Linear(32, 48, bias=False)
        gate.weight.data, up.weight.data = gate_up[:48], gate_up[48:]  # gate rows come first
        down = nn.Linear(48, 32, bias=False)
        down.weight.data = layer.down_proj.weight.float()

        out = layer(x)

        expected = down(reference_activation(gate(x.float())) * up(x.float()))
        assert out.shape == x.shape
        _close(out, expected, dtype, factor=16.0)


class TestGatedMLPRejections:
    def test_an_unknown_activation_is_refused(self) -> None:
        with pytest.raises(ValueError, match="activation must be 'silu', 'gelu' or 'gelu_tanh'"):
            GatedMLP(
                8,
                8,
                activation=cast("Any", "relu"),
                ops=_ops(torch.float32),
                dtype=torch.float32,
            )


class _Recorder:
    """Stands in for one op family; writes a recognisable value and notes the call."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        def op(*args: Any, **_: Any) -> None:
            self.calls.append(name)
            args[-1].fill_(42.0)  # every op the tests route here takes ``out`` last

        return op


class TestLayersCallTheBackend:
    @pytest.mark.parametrize(
        ("family", "build", "run", "op"),
        [
            (
                "linear",
                lambda ops: Linear(4, 3, ops=ops, dtype=torch.float32),
                lambda layer: layer(torch.ones(2, 4)),
                "linear",
            ),
            (
                "norm",
                lambda ops: RMSNorm(4, eps=1e-6, ops=ops, dtype=torch.float32),
                lambda layer: layer(torch.ones(2, 4)),
                "rms_norm",
            ),
            (
                "norm",
                lambda ops: LayerNorm(4, eps=1e-6, ops=ops, dtype=torch.float32),
                lambda layer: layer(torch.ones(2, 4)),
                "layer_norm",
            ),
            (
                "activation",
                lambda ops: GELU(tanh=True, ops=ops),
                lambda layer: layer(torch.ones(2, 4)),
                "gelu",
            ),
            (
                "activation",
                lambda ops: SiLUAndMul(ops=ops),
                lambda layer: layer(torch.ones(2, 4)),
                "silu_and_mul",
            ),
            (
                "embedding",
                lambda ops: TokenEmbedding(5, 4, ops=ops, dtype=torch.float32),
                lambda layer: layer(torch.zeros(2, dtype=torch.int64)),
                "lookup",
            ),
        ],
        ids=["linear", "rms_norm", "layer_norm", "gelu", "silu_and_mul", "lookup"],
    )
    def test_the_work_is_done_by_the_injected_op(
        self, family: str, build: Any, run: Any, op: str
    ) -> None:
        recorder = _Recorder()
        ops = dataclasses.replace(_ops(torch.float32), **{family: recorder})

        out = run(build(ops))

        assert recorder.calls == [op]
        assert (out == 42.0).all()

    def test_a_block_reaches_the_backend_through_its_projections(self) -> None:
        recorder = _Recorder()
        ops = dataclasses.replace(_ops(torch.float32), linear=recorder)

        GatedMLP(4, 6, ops=ops, dtype=torch.float32)(torch.ones(2, 4))

        assert recorder.calls == ["linear", "linear"]
