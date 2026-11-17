"""Backend parity: the same cases on every backend that can run them, against one oracle.

Each case is written once and parametrised by *target*: a backend list and a device. The CPU
reference (``torch`` on ``cpu``) runs everywhere and is ``unit``; every other target needs a GPU
and is ``gpu``. A target whose backend is not installed is skipped with the reason.

Tolerances. Attention is compared with a float64 oracle: a backend that accumulates in at
least fp32 and rounds once is within ``4 * eps(dtype) * max|y|``, the bound the contract suite uses
for the reference backend itself. In fp32 the bound is ``16 * eps``: on the dev machine the torch
backend on CUDA reached about 5 eps, because its fused kernel keeps a running softmax and rounds
more than once. The other families are compared with the CPU reference on the same inputs, within
8 units in the last place of the dtype, because two devices sum in different orders. Sampling with
the same random numbers must give the same tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.ops.backend import OpsBackend, load_backend
from ayaka.ops.kvcache import RAW_CODEC, KVBuffers
from ayaka.ops.sampling import DrawStream
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

BLOCK = 16
HEAD_DIM = 64  # every backend has kernels for it
_DTYPE = {DType.FP16: torch.float16, DType.BF16: torch.bfloat16, DType.FP32: torch.float32}


@dataclass(frozen=True)
class Target:
    """Where a parity case runs: a backend list (most preferred first) and a device."""

    name: str
    backends: tuple[str, ...]
    device: str
    dtypes: tuple[DType, ...]
    module: str | None = None  # a library the backend needs, skipped when missing

    def ops(self, dtype: DType) -> OpsBackend:
        if self.module is not None:
            pytest.importorskip(self.module, reason=f"{self.name} needs {self.module}")
        if dtype not in self.dtypes:
            pytest.skip(f"{self.name} does not compute in {dtype.label}")
        return load_backend(list(self.backends), device=self.device, dtype=dtype)


TARGETS = [
    pytest.param(
        Target("torch-cpu", ("torch",), "cpu", (DType.FP32, DType.FP16, DType.BF16)),
        marks=pytest.mark.unit,
        id="torch-cpu",
    ),
    pytest.param(
        Target("torch-cuda", ("torch",), "cuda", (DType.FP32, DType.FP16, DType.BF16)),
        marks=pytest.mark.gpu,
        id="torch-cuda",
    ),
    pytest.param(
        Target(
            "flashinfer-cuda",
            ("flashinfer", "torch"),
            "cuda",
            (DType.FP16, DType.BF16),
            module="flashinfer",
        ),
        marks=pytest.mark.gpu,
        id="flashinfer-cuda",
    ),
]
DTYPES = [DType.FP16, DType.BF16, DType.FP32]


def _reference() -> OpsBackend:
    return load_backend(["torch"], device="cpu", dtype=DType.FP32)


# -- attention ---------------------------------------------------------------------------------


@dataclass
class Paged:
    buffers: KVBuffers
    tables: list[list[int]]
    keys: list[Any]
    values: list[Any]


def _paged(
    lengths: list[int], *, dtype: Any, device: str, layout: KVLayoutKind, kv_heads: int, seed: int
) -> Paged:
    generator = torch.Generator().manual_seed(seed)
    needed = [-(-n // BLOCK) for n in lengths]
    blocks = 1 + sum(needed) + 2
    shape = (
        (blocks, BLOCK, kv_heads, HEAD_DIM)
        if layout is KVLayoutKind.NHD
        else (blocks, kv_heads, BLOCK, HEAD_DIM)
    )
    k = torch.full(shape, float("nan"), dtype=dtype)
    v = torch.full(shape, float("nan"), dtype=dtype)
    nhd_k = k if layout is KVLayoutKind.NHD else k.permute(0, 2, 1, 3)
    nhd_v = v if layout is KVLayoutKind.NHD else v.permute(0, 2, 1, 3)
    free = (torch.randperm(blocks - 1, generator=generator) + 1).tolist()
    tables, keys, values = [], [], []
    for length, count in zip(lengths, needed, strict=True):
        table = [free.pop() for _ in range(count)]
        ck = torch.randn(length, kv_heads, HEAD_DIM, generator=generator).to(dtype)
        cv = torch.randn(length, kv_heads, HEAD_DIM, generator=generator).to(dtype)
        for position in range(length):
            nhd_k[table[position // BLOCK], position % BLOCK] = ck[position]
            nhd_v[table[position // BLOCK], position % BLOCK] = cv[position]
        tables.append(table)
        keys.append(ck)
        values.append(cv)
    # Block 0 and the unused tails stay NaN except where a backend may read them as padding.
    k[0] = 0
    v[0] = 0
    buffers = KVBuffers(k.to(device), v.to(device), layout, RAW_CODEC)
    return Paged(buffers, tables, keys, values)


def _oracle(q: Any, keys: Any, values: Any, scale: float) -> Any:
    rows, heads, _ = q.shape
    length, kv_heads, _ = keys.shape
    out = torch.empty(q.shape, dtype=torch.float64)
    for row in range(rows):
        visible = length - rows + row + 1
        k = keys[:visible].double().repeat_interleave(heads // kv_heads, dim=1)
        v = values[:visible].double().repeat_interleave(heads // kv_heads, dim=1)
        weights = torch.softmax(torch.einsum("hd,thd->ht", q[row].double(), k) * scale, dim=-1)
        out[row] = torch.einsum("ht,thd->hd", weights, v)
    return out


def _attention_case(
    target: Target,
    dtype: DType,
    steps: list[tuple[int, int]],
    *,
    layout: KVLayoutKind = KVLayoutKind.NHD,
    heads: int = 8,
    kv_heads: int = 2,
    graph: bool = False,
) -> None:
    ops = target.ops(dtype)
    torch_dtype = _DTYPE[dtype]
    rows = [count for count, _ in steps]
    lengths = [length for _, length in steps]
    paged = _paged(
        [n for n in lengths if n],
        dtype=torch_dtype,
        device=target.device,
        layout=layout,
        kv_heads=kv_heads,
        seed=sum(lengths),
    )
    tables, real = [], iter(paged.tables)
    for length in lengths:
        tables.append(next(real) if length else [])
    width = max(max((len(t) for t in tables), default=1), 1)
    table = torch.tensor([t + [0] * (width - len(t)) for t in tables], dtype=torch.int32)
    q = torch.randn(sum(rows), heads, HEAD_DIM, generator=torch.Generator().manual_seed(7))
    q = q.to(torch_dtype)
    scale = HEAD_DIM**-0.5
    device = torch.device(target.device)
    if graph:
        meta = ops.attention.graph_metadata(len(steps), width, BLOCK, device)
        ops.attention.update_graph_metadata(meta, torch.tensor(lengths, dtype=torch.int32), table)
    else:
        meta = ops.attention.plan(
            ForwardMode.DECODE if set(rows) == {1} else ForwardMode.MIXED,
            torch.tensor(rows, dtype=torch.int32, device=device),
            torch.tensor(lengths, dtype=torch.int32, device=device),
            table.to(device),
            BLOCK,
        )
    out = torch.full(q.shape, 7.0, dtype=torch_dtype, device=device)
    ops.attention.attend(q.to(device), paged.buffers, meta, out, scale=scale)

    expected = torch.zeros(q.shape, dtype=torch.float64)
    first, real_index = 0, 0
    for count, length in steps:
        if length:
            expected[first : first + count] = _oracle(
                q[first : first + count], paged.keys[real_index], paged.values[real_index], scale
            )
            real_index += 1
        first += count
    eps = torch.finfo(torch_dtype).eps
    # fp32 on a GPU: the fused kernels keep a running softmax and round more than once, so the
    # single-rounding bound of the CPU reference becomes 16 eps (measured: about 5 eps).
    units = 16 if dtype is DType.FP32 else 4
    atol = units * eps * float(expected.abs().max()) + (1e-6 if dtype is DType.FP32 else 0.0)
    torch.testing.assert_close(out.cpu().double(), expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", DTYPES, ids=[d.label for d in DTYPES])
class TestAttention:
    def test_decode_rows(self, target: Target, dtype: DType) -> None:
        _attention_case(target, dtype, [(1, 1), (1, 16), (1, 17), (1, 70)])

    def test_prefill_chunks_and_verify_windows(self, target: Target, dtype: DType) -> None:
        _attention_case(target, dtype, [(20, 20), (5, 37), (1, 9), (4, 16), (16, 48)])

    def test_padding_requests(self, target: Target, dtype: DType) -> None:
        _attention_case(target, dtype, [(3, 10), (2, 0), (1, 5), (1, 0)])

    @pytest.mark.parametrize(("heads", "kv_heads"), [(4, 4), (8, 1), (12, 4)])
    def test_grouped_heads(self, target: Target, dtype: DType, heads: int, kv_heads: int) -> None:
        _attention_case(target, dtype, [(6, 19), (1, 33)], heads=heads, kv_heads=kv_heads)

    def test_hnd_layout(self, target: Target, dtype: DType) -> None:
        _attention_case(target, dtype, [(7, 7), (1, 40)], layout=KVLayoutKind.HND)

    def test_graph_metadata(self, target: Target, dtype: DType) -> None:
        _attention_case(target, dtype, [(1, 5), (1, 0), (1, 64), (1, 0)], graph=True)


# -- the other families: the CPU reference on the same inputs ------------------------------------


def _close(got: Any, expected: Any, dtype: DType) -> None:
    eps = torch.finfo(_DTYPE[dtype]).eps
    scale = max(float(expected.abs().max()), 1.0)
    torch.testing.assert_close(
        got.cpu().float(), expected.float(), rtol=8 * eps, atol=8 * eps * scale
    )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("dtype", DTYPES, ids=[d.label for d in DTYPES])
class TestElementwise:
    def test_rms_norm_and_fused_add(self, target: Target, dtype: DType) -> None:
        ops, reference, dt = target.ops(dtype), _reference(), _DTYPE[dtype]
        generator = torch.Generator().manual_seed(1)
        x = torch.randn(9, 96, generator=generator).to(dt)
        residual = torch.randn(9, 96, generator=generator).to(dt)
        weight = (1 + 0.1 * torch.randn(96, generator=generator)).to(dt)
        out, expected = torch.empty_like(x, device=target.device), torch.empty_like(x)
        ops.norm.rms_norm(x.to(target.device), weight.to(target.device), out, eps=1e-5)
        reference.norm.rms_norm(x, weight, expected, eps=1e-5)
        _close(out, expected, dtype)

        # Copies: on the CPU, ``to`` returns the tensor itself, and both ops write in place.
        xd, rd = x.to(target.device, copy=True), residual.to(target.device, copy=True)
        xc, rc = x.clone(), residual.clone()
        ops.norm.fused_add_rms_norm(xd, rd, weight.to(target.device), eps=1e-5)
        reference.norm.fused_add_rms_norm(xc, rc, weight, eps=1e-5)
        _close(xd, xc, dtype)
        _close(rd, rc, dtype)

    def test_silu_and_mul_and_gelu(self, target: Target, dtype: DType) -> None:
        ops, reference, dt = target.ops(dtype), _reference(), _DTYPE[dtype]
        x = torch.randn(5, 64, generator=torch.Generator().manual_seed(2)).to(dt)
        out, expected = (
            torch.empty(5, 32, dtype=dt, device=target.device),
            torch.empty(5, 32, dtype=dt),
        )
        ops.activation.silu_and_mul(x.to(target.device), out)
        reference.activation.silu_and_mul(x, expected)
        _close(out, expected, dtype)
        out, expected = torch.empty_like(x, device=target.device), torch.empty_like(x)
        ops.activation.gelu(x.to(target.device), out, tanh=True)
        reference.activation.gelu(x, expected, tanh=True)
        _close(out, expected, dtype)

    def test_store_and_gather(self, target: Target, dtype: DType) -> None:
        ops, dt = target.ops(dtype), _DTYPE[dtype]
        paged = _paged(
            [20], dtype=dt, device=target.device, layout=KVLayoutKind.NHD, kv_heads=2, seed=4
        )
        device = torch.device(target.device)
        # One decode row at position 20: offset 4 of the request's second block.
        meta = ops.attention.plan(
            ForwardMode.DECODE,
            torch.ones(1, dtype=torch.int32, device=device),
            torch.tensor([21], dtype=torch.int32, device=device),
            torch.tensor([paged.tables[0]], dtype=torch.int32, device=device),
            BLOCK,
        )
        k = torch.full((1, 2, HEAD_DIM), 3.0, dtype=dt, device=device)
        ops.kvcache.store(k, k + 1, paged.buffers, meta)
        block = paged.tables[0][1]
        out_k = torch.empty(1, BLOCK, 2, HEAD_DIM, dtype=dt, device=device)
        out_v = torch.empty_like(out_k)
        ops.kvcache.gather(paged.buffers, torch.tensor([block], device=device), out_k, out_v)
        assert torch.equal(out_k[0, 4].cpu(), k[0].cpu())
        assert torch.equal(out_v[0, 4].cpu(), (k + 1)[0].cpu())


@pytest.mark.parametrize("target", TARGETS)
class TestSampling:
    def test_the_same_random_numbers_give_the_same_tokens(self, target: Target) -> None:
        ops, reference = target.ops(DType.FP32), _reference()
        generator = torch.Generator().manual_seed(5)
        rows = 64
        logits = torch.randn(rows, 500, generator=generator) * 3
        seeds = torch.randint(0, 2**40, (rows,), generator=generator)
        positions = torch.randint(0, 4096, (rows,), generator=generator)
        sequences = torch.zeros(rows, dtype=torch.int32)
        temperature = torch.rand(rows, generator=generator) + 0.3
        top_k = torch.randint(0, 50, (rows,), generator=generator, dtype=torch.int32)
        top_p = torch.rand(rows, generator=generator) * 0.5 + 0.5
        min_p = torch.rand(rows, generator=generator) * 0.1
        device = target.device

        u, expected_u = torch.empty(rows, device=device), torch.empty(rows)
        ops.sampling.uniform(
            seeds.to(device), positions.to(device), sequences.to(device), u, stream=DrawStream.PICK
        )
        reference.sampling.uniform(seeds, positions, sequences, expected_u, stream=DrawStream.PICK)
        assert torch.equal(u.cpu(), expected_u)

        out = torch.empty(rows, dtype=torch.int64, device=device)
        expected = torch.empty(rows, dtype=torch.int64)
        ops.sampling.sample(
            logits.to(device),
            temperature.to(device),
            top_k.to(device),
            top_p.to(device),
            min_p.to(device),
            u,
            out,
        )
        reference.sampling.sample(logits, temperature, top_k, top_p, min_p, expected_u, expected)
        assert torch.equal(out.cpu(), expected)

        ops.sampling.argmax(logits.to(device), out)
        reference.sampling.argmax(logits, expected)
        assert torch.equal(out.cpu(), expected)
