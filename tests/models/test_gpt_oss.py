"""Unit and transformers parity tests for GptOss and MXFP4 dequantization."""

from __future__ import annotations

import math
from typing import Any

import pytest

from ayaka.configs.model import ModelConfig, RopeScaling
from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.kvcache.storage.pool import KVPool
from ayaka.layers.context import ForwardContext
from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType
from ayaka.weights.mxfp4 import FP4_E2M1_LUT, dequantize_mxfp4
from ayaka.weights.quant import QuantFormat

torch: Any = pytest.importorskip("torch")

from ayaka.models.gpt_oss import GptOssForCausalLM  # noqa: E402
from gpt2_support import load_hf_state  # noqa: E402


def _tiny_gpt_oss_config() -> dict[str, Any]:
    return {
        "architectures": ["GptOssForCausalLM"],
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "sliding_window": 4,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "attention_bias": True,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "swiglu_limit": 7.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 2.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 32,
            "truncate": False,
        },
    }


@pytest.mark.unit
def test_mxfp4_dequantize_exact_values() -> None:
    # Pack nibbles: byte = lo | (hi << 4) -> lo=idx2 (1.0), hi=idx10 (-1.0)
    byte_val = 2 | (10 << 4)
    blocks = torch.full((2, 1, 16), byte_val, dtype=torch.uint8)
    # Scale exponent 128 => 2^(128-127) = 2.0 on row 0; 126 => 0.5 on row 1
    scales = torch.tensor([[128], [126]], dtype=torch.uint8)

    out = dequantize_mxfp4(blocks, scales, dtype=torch.float32)
    assert out.shape == (2, 32)
    assert out[0, 0::2].tolist() == [2.0] * 16
    assert out[0, 1::2].tolist() == [-2.0] * 16
    assert out[1, 0::2].tolist() == [0.5] * 16
    assert out[1, 1::2].tolist() == [-0.5] * 16
    assert len(FP4_E2M1_LUT) == 16


@pytest.mark.unit
def test_gpt_oss_mxfp4_gathered_vs_grouped_paths_agree() -> None:
    cfg = ModelConfig.from_hf_config(_tiny_gpt_oss_config())
    assert cfg.rope_scaling == RopeScaling(
        kind="yarn",
        factor=2.0,
        original_max_position_embeddings=32,
        truncate=False,
    )
    assert cfg.context_length == 64

    ops = load_backend(["torch"], device="cpu", dtype=DType.BF16)
    quant = QuantFormat(
        producer="mxfp4",
        weight_dtype=DType.FP4_E2M1,
        scale_layout="block",
        block=(1, 32),
        scale_name="scales",
    )
    model = GptOssForCausalLM(cfg, ops=ops, dtype=DType.BF16, quant=quant)
    torch.manual_seed(3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype == torch.uint8:
                if name.endswith("_blocks"):
                    param.random_(0, 255)
                else:
                    param.fill_(126)  # small positive scale 0.5
            elif "norm" in name:
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.1)

    groups = model.kv_specs(block_size=4, layout=KVLayoutKind.NHD)
    assert [(g.layer_ids, g.spec.window) for g in groups] == [((0,), 4), ((1,), None)]

    pools = [KVPool(g, 16, device="cpu") for g in groups]
    kv_buffers = {lid: pool.buffers(lid) for pool in pools for lid in pool.group.layer_ids}
    tokens = torch.tensor(list(range(1, 13)), dtype=torch.int64)  # 12 > 8 => grouped prefill
    table = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    meta = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([12], dtype=torch.int32),
        torch.tensor([12], dtype=torch.int32),
        table,
        4,
    )
    ctx = ForwardContext(attention=meta, kv_buffers=kv_buffers)
    with torch.no_grad():
        grouped = model.compute_logits(model(tokens, ctx))

    # Re-run chunked by 3 tokens (<= 8 => gathered path) on fresh KV pools
    pools2 = [KVPool(g, 16, device="cpu") for g in groups]
    kv_buf2 = {lid: pool.buffers(lid) for pool in pools2 for lid in pool.group.layer_ids}
    chunks = []
    for start in range(0, 12, 3):
        end = start + 3
        pages = math.ceil(end / 4)
        m2 = ops.attention.plan(
            ForwardMode.PREFILL,
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([end], dtype=torch.int32),
            torch.tensor([list(range(1, pages + 1))], dtype=torch.int32),
            4,
        )
        with torch.no_grad():
            chunks.append(
                model.compute_logits(model(tokens[start:end], ForwardContext(m2, kv_buf2)))
            )
    gathered = torch.cat(chunks, dim=0)
    torch.testing.assert_close(grouped, gathered, rtol=3e-2, atol=3e-2)


@pytest.mark.integration
def test_gpt_oss_matches_transformers_and_mxfp4_dequant() -> None:
    transformers: Any = pytest.importorskip("transformers")
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    # Verify dequantize_mxfp4 matches transformers bit-for-bit
    torch.manual_seed(5)
    blocks = torch.randint(0, 256, (4, 64, 2, 16), dtype=torch.uint8)
    scales = torch.randint(120, 132, (4, 64, 2), dtype=torch.uint8)
    ref_deq = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)  # [E, in, out]
    our_deq = dequantize_mxfp4(blocks, scales, dtype=torch.bfloat16).transpose(1, 2).contiguous()
    torch.testing.assert_close(our_deq, ref_deq, rtol=0.0, atol=0.0)

    raw = _tiny_gpt_oss_config()
    hf_cfg = transformers.GptOssConfig(**raw, attn_implementation="eager")
    ref = transformers.GptOssForCausalLM(hf_cfg).float().eval()
    torch.manual_seed(19)
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "norm" in name:
                param.normal_(1.0, 0.1)
            else:
                param.normal_(0.0, 0.15)

    cfg = ModelConfig.from_hf_config(raw)
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    ours = GptOssForCausalLM(cfg, ops=ops, dtype=DType.FP32)
    load_hf_state(ours, dict(ref.state_dict()))

    tokens = torch.tensor([2, 7, 13, 19, 25, 31, 40, 44], dtype=torch.int64)
    block_size = 4
    groups = ours.kv_specs(block_size=block_size, layout=ops.kvcache.layout())
    pools = [KVPool(g, 8, device="cpu") for g in groups]
    kv_buf = {lid: pool.buffers(lid) for pool in pools for lid in pool.group.layer_ids}
    table = torch.tensor([[1, 2]], dtype=torch.int32)
    meta = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([tokens.shape[0]], dtype=torch.int32),
        torch.tensor([tokens.shape[0]], dtype=torch.int32),
        table,
        block_size,
    )
    with torch.no_grad():
        expected = ref(input_ids=tokens[None]).logits[0]
        actual = ours.compute_logits(ours(tokens, ForwardContext(meta, kv_buf)))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
