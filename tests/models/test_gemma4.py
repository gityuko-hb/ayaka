"""Unit and transformers parity tests for Gemma 4 sliding+full shared KV and PLE."""

from __future__ import annotations

import math
from typing import Any

import pytest

from ayaka.configs.model import ModelConfig
from ayaka.core.types import ForwardMode, KVLayoutKind
from ayaka.kvcache.storage.pool import KVPool
from ayaka.layers.context import ForwardContext
from ayaka.ops.backend import load_backend
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.models.gemma4 import Gemma4ForConditionalGeneration  # noqa: E402
from gpt2_support import load_hf_state  # noqa: E402


def _tiny_gemma4_config() -> dict[str, Any]:
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "tie_word_embeddings": True,
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 6,
            "num_kv_shared_layers": 2,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "global_head_dim": 32,
            "sliding_window": 4,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 64,
            "use_double_wide_mlp": True,
            "hidden_size_per_layer_input": 8,
            "vocab_size_per_layer_input": 64,
            "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        },
    }


def _run_paged(model: Gemma4ForConditionalGeneration, tokens: Any, *, chunk: int | None) -> Any:
    ops = model.ops
    block_size = 4
    groups = model.kv_specs(block_size=block_size, layout=ops.kvcache.layout())
    pools = [KVPool(g, 16, device="cpu") for g in groups]
    kv_buffers = {lid: pool.buffers(lid) for pool in pools for lid in pool.group.layer_ids}
    seq_len = tokens.shape[0]
    step_size = seq_len if chunk is None else chunk
    outs: list[Any] = []
    pos = 0
    while pos < seq_len:
        end = min(pos + step_size, seq_len)
        count = end - pos
        pages = math.ceil(end / block_size)
        table = torch.tensor([list(range(1, pages + 1))], dtype=torch.int32)
        meta = ops.attention.plan(
            ForwardMode.DECODE if count == 1 and pos > 0 else ForwardMode.PREFILL,
            torch.tensor([count], dtype=torch.int32),
            torch.tensor([end], dtype=torch.int32),
            table,
            block_size,
        )
        ctx = ForwardContext(attention=meta, kv_buffers=kv_buffers)
        with torch.no_grad():
            outs.append(model(tokens[pos:end], ctx))
        pos = end
    return torch.cat(outs, dim=0)


@pytest.mark.unit
def test_gemma4_specs_shared_kv_and_chunk_invariance() -> None:
    cfg = ModelConfig.from_hf_config(_tiny_gemma4_config())
    assert cfg.context_length == 64  # sliding window is natively executed
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    model = Gemma4ForConditionalGeneration(cfg, ops=ops, dtype=DType.FP32)
    torch.manual_seed(11)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "scalar" in name or "norm" in name:
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.2)

    groups = model.kv_specs(block_size=4, layout=KVLayoutKind.NHD)
    assert len(groups) == 2
    assert groups[0].layer_ids == (0, 2)
    assert groups[0].spec.head_dim == 16
    assert groups[0].spec.window == 4
    assert groups[1].layer_ids == (1, 3)
    assert groups[1].spec.head_dim == 32
    assert groups[1].spec.window is None

    tokens = torch.tensor([1, 4, 9, 12, 17, 25, 31, 42, 50], dtype=torch.int64)
    full_h = _run_paged(model, tokens, chunk=None)
    chunk_h = _run_paged(model, tokens, chunk=3)
    torch.testing.assert_close(full_h, chunk_h, rtol=1e-5, atol=1e-5)


@pytest.mark.integration
def test_gemma4_matches_transformers_tiny_cpu() -> None:
    transformers: Any = pytest.importorskip("transformers")
    raw = _tiny_gemma4_config()
    tc = dict(raw["text_config"])
    tc["attn_implementation"] = "eager"
    hf_text_cfg = transformers.Gemma4TextConfig(**tc)
    ref = transformers.Gemma4ForCausalLM(hf_text_cfg).float().eval()
    torch.manual_seed(23)
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "scalar" in name:
                param.fill_(0.85)
            elif "norm" in name:
                param.normal_(1.0, 0.1)
            else:
                param.normal_(0.0, 0.15)
        for name, buf in ref.named_buffers():
            if "layer_scalar" in name:
                buf.fill_(0.85)

    cfg = ModelConfig.from_hf_config(raw)
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    ours = Gemma4ForConditionalGeneration(cfg, ops=ops, dtype=DType.FP32)
    # HF Gemma4TextDecoderLayer stores layer_scalar as a persistent buffer in state_dict.
    state = dict(ref.state_dict())
    load_hf_state(ours, state)

    tokens = torch.tensor([3, 8, 19, 27, 45, 52, 6, 11], dtype=torch.int64)
    with torch.no_grad():
        expected = ref(input_ids=tokens[None]).logits[0]
        hidden = _run_paged(ours, tokens, chunk=3)
        actual = ours.compute_logits(hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
