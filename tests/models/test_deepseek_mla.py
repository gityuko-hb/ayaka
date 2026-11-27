"""Unit and transformers parity tests for DeepSeek V2/V3 Multi-Head Latent Attention."""

from __future__ import annotations

import math
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.core.types import ForwardMode, KVLayoutKind  # noqa: E402
from ayaka.kvcache.storage.pool import LatentKVPool  # noqa: E402
from ayaka.kvcache.storage.spec import LatentKVGroup, LatentKVSpec  # noqa: E402
from ayaka.layers.context import ForwardContext  # noqa: E402
from ayaka.models.deepseek_mla import DeepseekV2ForCausalLM  # noqa: E402
from ayaka.ops.backend import load_backend  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402
from gpt2_support import load_hf_state  # noqa: E402


def _tiny_ds2_config(*, moe: bool) -> dict[str, Any]:
    return {
        "architectures": ["DeepseekV2ForCausalLM"],
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "q_lora_rank": 12,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "rope_interleave": True,
        "n_routed_experts": 4 if moe else 0,
        "num_experts_per_tok": 2 if moe else 0,
        "moe_intermediate_size": 24 if moe else 0,
        "n_shared_experts": 1 if moe else 0,
        "first_k_dense_replace": 1 if moe else 2,
    }


@pytest.mark.unit
def test_deepseek_mla_halves_kv_cache_and_matches_chunked_decode() -> None:
    # Full-scale DeepSeek V2/V3 compression: (512 + 64) vs 2 * 16 * 128 = 4096 => 0.14x
    spec_v2 = LatentKVSpec(
        num_layers=60, kv_lora_rank=512, qk_rope_head_dim=64, block_size=16, dtype=DType.BF16
    )
    assert spec_v2.compression_ratio_vs_kv(num_kv_heads=16, head_dim=128) <= 0.5

    cfg = ModelConfig.from_hf_config(_tiny_ds2_config(moe=False))
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    model = DeepseekV2ForCausalLM(cfg, ops=ops, dtype=DType.FP32)
    torch.manual_seed(9)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.2)

    groups = model.kv_specs(block_size=4, layout=KVLayoutKind.NHD)
    assert len(groups) == 1
    assert isinstance(groups[0], LatentKVGroup)
    assert groups[0].spec.compression_ratio_vs_kv(cfg.num_kv_heads, cfg.head_dim) <= 0.5

    tokens = torch.tensor([2, 5, 11, 17, 23, 29, 35], dtype=torch.int64)
    pool_full = LatentKVPool(groups[0], 8, device="cpu")
    m_full = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([[1, 2]], dtype=torch.int32),
        4,
    )
    with torch.no_grad():
        h_full = model(
            tokens,
            ForwardContext(
                attention=m_full,
                kv_buffers={},
                latent_kv_buffers={0: pool_full.buffers(0), 1: pool_full.buffers(1)},
            ),
        )

    pool_chunk = LatentKVPool(groups[0], 8, device="cpu")
    bufs = {0: pool_chunk.buffers(0), 1: pool_chunk.buffers(1)}
    parts: list[Any] = []
    for pos, end in ((0, 4), (4, 6), (6, 7)):
        count = end - pos
        pages = math.ceil(end / 4)
        m = ops.attention.plan(
            ForwardMode.DECODE if count == 1 else ForwardMode.PREFILL,
            torch.tensor([count], dtype=torch.int32),
            torch.tensor([end], dtype=torch.int32),
            torch.tensor([list(range(1, pages + 1))], dtype=torch.int32),
            4,
        )
        with torch.no_grad():
            parts.append(
                model(
                    tokens[pos:end],
                    ForwardContext(attention=m, kv_buffers={}, latent_kv_buffers=bufs),
                )
            )
    h_chunk = torch.cat(parts, dim=0)
    torch.testing.assert_close(h_full, h_chunk, rtol=1e-5, atol=1e-5)


@pytest.mark.integration
def test_deepseek_v2_matches_transformers_tiny_cpu() -> None:
    transformers: Any = pytest.importorskip("transformers")
    raw = _tiny_ds2_config(moe=False)
    hf_cfg = transformers.DeepseekV2Config(**raw, attn_implementation="eager")
    ref = transformers.DeepseekV2ForCausalLM(hf_cfg).float().eval()
    torch.manual_seed(31)
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "norm" in name:
                param.normal_(1.0, 0.1)
            else:
                param.normal_(0.0, 0.2)

    cfg = ModelConfig.from_hf_config(raw)
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    ours = DeepseekV2ForCausalLM(cfg, ops=ops, dtype=DType.FP32)
    load_hf_state(ours, dict(ref.state_dict()))

    tokens = torch.tensor([1, 8, 15, 22, 30, 41], dtype=torch.int64)
    group = ours.kv_specs(block_size=4, layout=ops.kvcache.layout())[0]
    pool = LatentKVPool(group, 8, device="cpu")
    meta = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([6], dtype=torch.int32),
        torch.tensor([6], dtype=torch.int32),
        torch.tensor([[1, 2]], dtype=torch.int32),
        4,
    )
    ctx = ForwardContext(
        attention=meta,
        kv_buffers={},
        latent_kv_buffers={lid: pool.buffers(lid) for lid in group.layer_ids},
    )
    with torch.no_grad():
        expected = ref(input_ids=tokens[None]).logits[0]
        actual = ours.compute_logits(ours(tokens, ctx))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
