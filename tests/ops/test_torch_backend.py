"""The torch reference backend against fp64 oracles written for these tests.

Tolerances. The oracle runs in float64 on the exact values the op receives, so
the only expected difference is the op's own rounding. Elementwise ops round
once, so the error is at most half an ulp of the result: ``rtol = eps``
(twice that bound) with a small ``atol`` for results near zero. The matrix
product accumulates, so its bound is relative to the largest result:
``atol = 2 * eps * max|y|``. ``eps`` is ``torch.finfo(dtype).eps``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.core.errors import BackendError
from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

if TYPE_CHECKING:
    from ayaka.ops.backend import OpsBackend

pytestmark = pytest.mark.unit

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DTYPE_IDS = ["fp16", "bf16", "fp32"]
_REGISTRY_DTYPE = {torch.float16: DType.FP16, torch.bfloat16: DType.BF16, torch.float32: DType.FP32}


def _ops(dtype: Any) -> OpsBackend:
    return load_backend(["torch"], device="cpu", dtype=_REGISTRY_DTYPE[dtype])


def _randn(*shape: int, dtype: Any, seed: int = 0) -> Any:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(dtype)


def _close(actual: Any, expected: Any, *, dtype: Any, scaled: bool = False) -> None:
    eps = torch.finfo(dtype).eps
    if scaled:
        rtol, atol = 0.0, 2 * eps * float(expected.abs().max())
    else:
        rtol, atol = eps, 1e-6
    torch.testing.assert_close(actual.double(), expected, rtol=rtol, atol=atol)


def _rms_norm_oracle(x: Any, weight: Any, eps: float) -> Any:
    x64 = x.double()
    return x64 / torch.sqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * weight.double()


def _layer_norm_oracle(x: Any, weight: Any, bias: Any | None, eps: float) -> Any:
    x64 = x.double()
    centred = x64 - x64.mean(-1, keepdim=True)
    result = centred / torch.sqrt(centred.pow(2).mean(-1, keepdim=True) + eps) * weight.double()
    return result if bias is None else result + bias.double()


def _gelu_oracle(x: Any, *, tanh: bool) -> Any:
    x64 = x.double()
    if tanh:
        inner = (2 / torch.pi) ** 0.5 * (x64 + 0.044715 * x64**3)
        return 0.5 * x64 * (1 + torch.tanh(inner))
    return 0.5 * x64 * (1 + torch.erf(x64 / 2**0.5))


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestNorm:
    def test_rms_norm_matches_the_oracle(self, dtype: Any) -> None:
        x, weight = _randn(3, 5, 64, dtype=dtype), _randn(64, dtype=dtype, seed=1)
        out = torch.empty_like(x)

        _ops(dtype).norm.rms_norm(x, weight, out, eps=1e-6)

        _close(out, _rms_norm_oracle(x, weight, 1e-6), dtype=dtype)

    def test_rms_norm_can_overwrite_its_input(self, dtype: Any) -> None:
        x, weight = _randn(4, 32, dtype=dtype), _randn(32, dtype=dtype, seed=1)
        expected = _rms_norm_oracle(x, weight, 1e-5)

        _ops(dtype).norm.rms_norm(x, weight, x, eps=1e-5)

        _close(x, expected, dtype=dtype)

    def test_rms_norm_of_large_fp16_values_does_not_overflow(self, dtype: Any) -> None:
        # 300**2 exceeds the fp16 maximum, so squaring in fp16 would give inf.
        x = torch.full((2, 8), 300.0).to(dtype)
        weight, out = torch.ones(8, dtype=dtype), torch.empty_like(x)

        _ops(dtype).norm.rms_norm(x, weight, out, eps=1e-6)

        _close(out, _rms_norm_oracle(x, weight, 1e-6), dtype=dtype)

    @pytest.mark.parametrize("with_bias", [True, False])
    def test_layer_norm_matches_the_oracle(self, dtype: Any, with_bias: bool) -> None:
        x, weight = _randn(3, 5, 48, dtype=dtype), _randn(48, dtype=dtype, seed=1)
        bias = _randn(48, dtype=dtype, seed=2) if with_bias else None
        out = torch.empty_like(x)

        _ops(dtype).norm.layer_norm(x, weight, bias, out, eps=1e-5)

        # Centring subtracts nearly equal numbers, so the bound is relative to the largest result.
        _close(out, _layer_norm_oracle(x, weight, bias, 1e-5), dtype=dtype, scaled=True)

    def test_fused_add_rms_norm_updates_both_tensors(self, dtype: Any) -> None:
        x, residual = _randn(4, 64, dtype=dtype), _randn(4, 64, dtype=dtype, seed=1)
        weight = _randn(64, dtype=dtype, seed=2)
        total = x.double() + residual.double()

        _ops(dtype).norm.fused_add_rms_norm(x, residual, weight, eps=1e-6)

        _close(residual, total, dtype=dtype)
        _close(x, _rms_norm_oracle(total, weight, 1e-6), dtype=dtype)

    def test_empty_batch_is_a_no_op(self, dtype: Any) -> None:
        x, weight = torch.empty(0, 16, dtype=dtype), torch.ones(16, dtype=dtype)
        out = torch.empty_like(x)

        _ops(dtype).norm.rms_norm(x, weight, out, eps=1e-6)

        assert out.shape == (0, 16)


class TestNormRejections:
    def test_integer_input_is_not_promoted(self) -> None:
        x = torch.ones(2, 4, dtype=torch.int32)
        weight, out = torch.ones(4), torch.empty(2, 4)

        with pytest.raises(BackendError, match=r"x is int32.*float16, bfloat16, float32"):
            _ops(torch.float32).norm.rms_norm(x, weight, out, eps=1e-6)

    def test_mismatched_weight_dtype_is_not_cast(self) -> None:
        x, out = torch.ones(2, 4), torch.empty(2, 4)

        with pytest.raises(BackendError, match=r"weight must be float32 \[4\], got float16 \[4\]"):
            _ops(torch.float32).norm.rms_norm(x, torch.ones(4, dtype=torch.float16), out, eps=1e-6)

    def test_wrong_out_shape_is_not_broadcast_into_and_stays_untouched(self) -> None:
        x, weight = torch.ones(2, 4), torch.ones(4)
        out = torch.full((2, 1), 7.0)

        with pytest.raises(BackendError, match=r"out must be float32 \[2, 4\]"):
            _ops(torch.float32).norm.rms_norm(x, weight, out, eps=1e-6)

        assert out.tolist() == [[7.0], [7.0]]

    def test_fused_add_names_the_residual_in_its_error(self) -> None:
        x, weight = torch.ones(2, 4), torch.ones(4)

        with pytest.raises(BackendError, match=r"residual must be float32 \[2, 4\]"):
            _ops(torch.float32).norm.fused_add_rms_norm(x, torch.ones(3, 4), weight, eps=1e-6)

    def test_layer_norm_bias_must_match(self) -> None:
        x, weight, out = torch.ones(2, 4), torch.ones(4), torch.empty(2, 4)

        with pytest.raises(BackendError, match=r"bias must be float32 \[4\]"):
            _ops(torch.float32).norm.layer_norm(x, weight, torch.ones(5), out, eps=1e-5)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestActivation:
    @pytest.mark.parametrize("tanh", [False, True])
    def test_gelu_matches_the_oracle(self, dtype: Any, tanh: bool) -> None:
        x = _randn(6, 33, dtype=dtype) * 3
        out = torch.empty_like(x)

        _ops(dtype).activation.gelu(x, out, tanh=tanh)

        _close(out, _gelu_oracle(x, tanh=tanh), dtype=dtype)

    def test_gelu_can_overwrite_its_input(self, dtype: Any) -> None:
        x = _randn(4, 16, dtype=dtype)
        expected = _gelu_oracle(x, tanh=False)

        _ops(dtype).activation.gelu(x, x, tanh=False)

        _close(x, expected, dtype=dtype)

    def test_silu_and_mul_uses_the_first_half_as_the_activated_half(self, dtype: Any) -> None:
        x = _randn(3, 4, 40, dtype=dtype)
        out = torch.empty(3, 4, 20, dtype=dtype)
        activated, gate = x.double().chunk(2, dim=-1)

        _ops(dtype).activation.silu_and_mul(x, out)

        _close(out, activated * torch.sigmoid(activated) * gate, dtype=dtype, scaled=True)

    @pytest.mark.parametrize("tanh", [False, True])
    def test_gelu_and_mul_matches_the_oracle(self, dtype: Any, tanh: bool) -> None:
        x = _randn(5, 24, dtype=dtype)
        out = torch.empty(5, 12, dtype=dtype)
        activated, gate = x.chunk(2, dim=-1)

        _ops(dtype).activation.gelu_and_mul(x, out, tanh=tanh)

        _close(out, _gelu_oracle(activated, tanh=tanh) * gate.double(), dtype=dtype, scaled=True)


class TestActivationRejections:
    def test_odd_packed_width_is_refused(self) -> None:
        with pytest.raises(BackendError, match=r"two halves .* last dimension, which is 7"):
            _ops(torch.float32).activation.silu_and_mul(torch.ones(2, 7), torch.empty(2, 3))

    def test_out_must_have_the_half_width(self) -> None:
        with pytest.raises(BackendError, match=r"out must be float32 \[2, 4\]"):
            _ops(torch.float32).activation.gelu_and_mul(
                torch.ones(2, 8), torch.empty(2, 8), tanh=False
            )

    def test_gelu_rejects_integer_input(self) -> None:
        with pytest.raises(BackendError, match="x is int64"):
            _ops(torch.float32).activation.gelu(
                torch.ones(2, dtype=torch.int64), torch.empty(2), tanh=False
            )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestEmbedding:
    @pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
    def test_lookup_copies_rows_exactly(self, dtype: Any, id_dtype: Any) -> None:
        weight = _randn(11, 8, dtype=dtype)
        ids = torch.tensor([[0, 10, 3], [3, 3, 7]], dtype=id_dtype)
        out = torch.empty(2, 3, 8, dtype=dtype)

        _ops(dtype).embedding.lookup(ids, weight, out)

        assert torch.equal(out, weight[ids.long()])

    def test_lookup_of_no_tokens_writes_nothing(self, dtype: Any) -> None:
        weight, ids = _randn(5, 4, dtype=dtype), torch.empty(0, dtype=torch.int64)
        out = torch.empty(0, 4, dtype=dtype)

        _ops(dtype).embedding.lookup(ids, weight, out)

        assert out.shape == (0, 4)


class TestEmbeddingRejections:
    def test_float_ids_are_refused(self) -> None:
        with pytest.raises(BackendError, match="token_ids must be int32 or int64, got float32"):
            _ops(torch.float32).embedding.lookup(
                torch.zeros(2), torch.ones(3, 4), torch.empty(2, 4)
            )

    def test_weight_must_be_a_table(self) -> None:
        with pytest.raises(BackendError, match=r"weight must be \[vocab, hidden\]"):
            _ops(torch.float32).embedding.lookup(
                torch.zeros(2, dtype=torch.int64), torch.ones(4), torch.empty(2, 4)
            )

    def test_out_must_have_the_hidden_width(self) -> None:
        with pytest.raises(BackendError, match=r"out must be float32 \[2, 4\]"):
            _ops(torch.float32).embedding.lookup(
                torch.zeros(2, dtype=torch.int64), torch.ones(3, 4), torch.empty(2, 5)
            )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestLinear:
    @pytest.mark.parametrize("with_bias", [True, False])
    def test_linear_matches_the_oracle(self, dtype: Any, with_bias: bool) -> None:
        x, weight = _randn(3, 5, 32, dtype=dtype), _randn(20, 32, dtype=dtype, seed=1)
        bias = _randn(20, dtype=dtype, seed=2) if with_bias else None
        out = torch.empty(3, 5, 20, dtype=dtype)
        expected = x.double() @ weight.double().T
        if bias is not None:
            expected = expected + bias.double()

        _ops(dtype).linear.linear(x, weight, bias, out)

        _close(out, expected, dtype=dtype, scaled=True)

    def test_empty_batch_gives_an_empty_result(self, dtype: Any) -> None:
        x, weight = torch.empty(0, 8, dtype=dtype), _randn(6, 8, dtype=dtype)
        out = torch.empty(0, 6, dtype=dtype)

        _ops(dtype).linear.linear(x, weight, None, out)

        assert out.shape == (0, 6)


class TestLinearRejections:
    def test_weight_must_match_the_input_width(self) -> None:
        with pytest.raises(BackendError, match=r"weight must be float32 \[6, 8\]"):
            _ops(torch.float32).linear.linear(
                torch.ones(2, 8), torch.ones(6, 7), None, torch.empty(2, 6)
            )

    def test_weight_must_be_a_matrix(self) -> None:
        with pytest.raises(BackendError, match=r"weight must be \[out_features, in_features\]"):
            _ops(torch.float32).linear.linear(torch.ones(2, 8), torch.ones(8), None, torch.empty(2))

    def test_bias_must_match_the_output_width(self) -> None:
        with pytest.raises(BackendError, match=r"bias must be float32 \[6\]"):
            _ops(torch.float32).linear.linear(
                torch.ones(2, 8), torch.ones(6, 8), torch.ones(5), torch.empty(2, 6)
            )

    def test_out_dtype_is_not_cast_into(self) -> None:
        with pytest.raises(BackendError, match=r"out must be float32 \[2, 6\]"):
            _ops(torch.float32).linear.linear(
                torch.ones(2, 8),
                torch.ones(6, 8),
                None,
                torch.empty(2, 6, dtype=torch.float16),
            )
