"""Multi-token-prediction (MTP) drafter: unit tests and MiMo-7B-RL GPU gates (#128)."""

from __future__ import annotations

import gc
import json
import math
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model import ModelConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.speculative import SpeculativeConfig
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.types import ForwardMode
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.engine.build import plan_engine  # noqa: E402
from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.kvcache.storage.pool import KVPool  # noqa: E402
from ayaka.layers.context import ForwardContext  # noqa: E402
from ayaka.model_loader.loader import load_model  # noqa: E402
from ayaka.model_loader.source import resolve_source  # noqa: E402
from ayaka.models.mimo import (  # noqa: E402
    describe_mimo_mtp_bytes,
    load_mimo_mtp_layer,
)
from ayaka.ops.backend import load_backend  # noqa: E402
from llama_support import llama_config, llama_state, write_llama  # noqa: E402
from tiny_model import patch_tokenizer_loading, tiny_tokenizer  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _tiny_mimo_checkpoint(directory: Path) -> tuple[ModelConfig, dict[str, Any]]:
    cfg = llama_config(
        architecture="MiMoForCausalLM", attention_bias=True, num_layers=2, vocab_size=260
    )
    base_state = {
        k: v for k, v in llama_state(cfg, seed=13).items() if not k.endswith("o_proj.bias")
    }
    gen = torch.Generator().manual_seed(29)
    h, inner = cfg.hidden_size, cfg.intermediate_size
    qr = cfg.num_attention_heads * cfg.head_dim
    kvr = cfg.num_kv_heads * cfg.head_dim
    mtp = {
        "model.mtp_layers.0.token_layernorm.weight": torch.randn((h,), generator=gen),
        "model.mtp_layers.0.hidden_layernorm.weight": torch.randn((h,), generator=gen),
        "model.mtp_layers.0.input_layernorm.weight": torch.randn((h,), generator=gen),
        "model.mtp_layers.0.post_attention_layernorm.weight": torch.randn((h,), generator=gen),
        "model.mtp_layers.0.final_layernorm.weight": torch.randn((h,), generator=gen),
        "model.mtp_layers.0.input_proj.weight": torch.randn((h, 2 * h), generator=gen),
        "model.mtp_layers.0.self_attn.q_proj.weight": torch.randn((qr, h), generator=gen),
        "model.mtp_layers.0.self_attn.q_proj.bias": torch.randn((qr,), generator=gen),
        "model.mtp_layers.0.self_attn.k_proj.weight": torch.randn((kvr, h), generator=gen),
        "model.mtp_layers.0.self_attn.k_proj.bias": torch.randn((kvr,), generator=gen),
        "model.mtp_layers.0.self_attn.v_proj.weight": torch.randn((kvr, h), generator=gen),
        "model.mtp_layers.0.self_attn.v_proj.bias": torch.randn((kvr,), generator=gen),
        "model.mtp_layers.0.self_attn.o_proj.weight": torch.randn((h, qr), generator=gen),
        "model.mtp_layers.0.mlp.gate_proj.weight": torch.randn((inner, h), generator=gen),
        "model.mtp_layers.0.mlp.up_proj.weight": torch.randn((inner, h), generator=gen),
        "model.mtp_layers.0.mlp.down_proj.weight": torch.randn((h, inner), generator=gen),
    }
    full = base_state | mtp
    write_llama(directory, full, cfg)
    cfg_path = directory / "config.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw["num_nextn_predict_layers"] = 1
    cfg_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return ModelConfig.from_hf_config(raw), mtp


def _manual_mimo_mtp(
    mtp: dict[str, Any],
    input_embeds: Any,
    hidden_states: Any,
    config: ModelConfig,
) -> Any:
    def rms(x: Any, w: Any) -> Any:
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + config.norm_eps)).to(x.dtype) * w

    prefix = "model.mtp_layers.0."
    t = rms(input_embeds, mtp[f"{prefix}token_layernorm.weight"])
    h = rms(hidden_states, mtp[f"{prefix}hidden_layernorm.weight"])
    x = torch.cat([h, t], dim=-1) @ mtp[f"{prefix}input_proj.weight"].T
    res = x
    xn = rms(x, mtp[f"{prefix}input_layernorm.weight"])
    seq = x.shape[0]
    heads, kv_heads, dim = config.num_attention_heads, config.num_kv_heads, config.head_dim
    theta = float(config.rope_theta or 10000.0)
    pos = torch.arange(seq, device=x.device, dtype=torch.float32)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    ang = torch.outer(pos, inv)
    cos = torch.cat([ang, ang], dim=-1).cos().unsqueeze(1).to(x.dtype)
    sin = torch.cat([ang, ang], dim=-1).sin().unsqueeze(1).to(x.dtype)

    def rotate(u: Any) -> Any:
        first, second = u[..., : dim // 2], u[..., dim // 2 :]
        return u * cos + torch.cat([-second, first], dim=-1) * sin

    q = xn @ mtp[f"{prefix}self_attn.q_proj.weight"].T + mtp[f"{prefix}self_attn.q_proj.bias"]
    k = xn @ mtp[f"{prefix}self_attn.k_proj.weight"].T + mtp[f"{prefix}self_attn.k_proj.bias"]
    v = xn @ mtp[f"{prefix}self_attn.v_proj.weight"].T + mtp[f"{prefix}self_attn.v_proj.bias"]
    q = rotate(q.view(seq, heads, dim))
    k = rotate(k.view(seq, kv_heads, dim))
    v = v.view(seq, kv_heads, dim)
    k = k.repeat_interleave(heads // kv_heads, dim=1)
    v = v.repeat_interleave(heads // kv_heads, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) / math.sqrt(dim)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
    attn = torch.einsum(
        "hts,shd->thd", scores.masked_fill(mask, float("-inf")).softmax(-1), v.float()
    )
    x = res + attn.to(x.dtype).reshape(seq, -1) @ mtp[f"{prefix}self_attn.o_proj.weight"].T
    res = x
    xn = rms(x, mtp[f"{prefix}post_attention_layernorm.weight"])
    gated = torch.nn.functional.silu(xn @ mtp[f"{prefix}mlp.gate_proj.weight"].T) * (
        xn @ mtp[f"{prefix}mlp.up_proj.weight"].T
    )
    x = res + gated @ mtp[f"{prefix}mlp.down_proj.weight"].T
    return rms(x, mtp[f"{prefix}final_layernorm.weight"])


def test_mimo_mtp_layer_loading_and_forward_parity(tmp_path: Path) -> None:
    d = tmp_path / "mimo-mtp"
    cfg, mtp_state = _tiny_mimo_checkpoint(d)
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    model: Any = load_model(ModelSourceConfig(str(d)), DeviceConfig(dtype=DType.FP32), ops=ops)

    mtp_layer = load_mimo_mtp_layer(
        d,
        cfg,
        rotary=model.rotary,
        ops=ops,
        dtype=DType.FP32,
        device="cpu",
        layer_id=cfg.num_layers,
        mtp_index=0,
    )
    param_bytes = sum(p.numel() * p.element_size() for p in mtp_layer.parameters())
    assert param_bytes == describe_mimo_mtp_bytes(cfg, dtype=DType.FP32)

    [group] = model.kv_specs(block_size=4, layout=ops.kvcache.layout())
    import dataclasses

    from ayaka.kvcache.storage.spec import KVGroup

    full_group = KVGroup(
        layer_ids=(*group.layer_ids, cfg.num_layers),
        spec=dataclasses.replace(group.spec, num_layers=group.spec.num_layers + 1),
    )
    pool = KVPool(full_group, 8, device="cpu")
    bufs = {lid: pool.buffers(lid) for lid in full_group.layer_ids}
    seq = 6
    bt = torch.tensor([[1, 2]], dtype=torch.int32)
    meta = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([seq], dtype=torch.int32),
        torch.tensor([seq], dtype=torch.int32),
        bt,
        4,
    )
    emb = torch.randn((seq, cfg.hidden_size), generator=torch.Generator().manual_seed(1))
    hid = torch.randn((seq, cfg.hidden_size), generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        actual = mtp_layer(emb, hid, ForwardContext(meta, bufs))
        expected = _manual_mimo_mtp(mtp_state, emb, hid, cfg)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_engine_mtp_matches_plain_and_prefix_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_tokenizer_loading(monkeypatch)
    d = tmp_path / "mimo-mtp-e2e"
    _tiny_mimo_checkpoint(d)

    plain_cfg = EngineConfig(
        source=ModelSourceConfig(str(d), offline=True),
        cache=CacheConfig(block_size=4, num_blocks=65),
    )
    mtp_cfg = EngineConfig(
        source=ModelSourceConfig(str(d), offline=True),
        cache=CacheConfig(block_size=4, num_blocks=65),
        speculative=SpeculativeConfig(
            enabled=True, method="mtp", controller="fixed", fixed_k=1, max_draft_tokens=1
        ),
    )
    plan = plan_engine(mtp_cfg)
    assert plan.budget.draft_bytes > 0
    assert plan.budget.loaded_drafts == ("mtp",)
    assert plan.group.spec.num_layers == 3

    prompts = [(1, 3, 5, 7, 9, 11), (1, 3, 5, 7, 13, 15)]
    with LLM.from_config(plain_cfg, tokenizer=tiny_tokenizer()) as plain_llm:
        plain_outs = plain_llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=10))
        assert isinstance(plain_outs, list)
        expected = [o.outputs[0].token_ids for o in plain_outs]

    with LLM.from_config(mtp_cfg, tokenizer=tiny_tokenizer()) as mtp_llm:
        outs = mtp_llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=10))
        assert isinstance(outs, list)
        assert [o.outputs[0].token_ids for o in outs] == expected
        # Run a second prompt sharing block 0 (with same lookahead token) to test lookahead caching.
        again = mtp_llm.generate((1, 3, 5, 7, 9, 99), SamplingParams(temperature=0.0, max_tokens=6))
        assert not isinstance(again, list)
        assert len(again.outputs[0].token_ids) == 6


@pytest.mark.gpu
def test_mimo_7b_rl_mtp_gates_on_l4() -> None:
    """Gates 1-4 on XiaomiMiMo/MiMo-7B-RL: >=99% parity, >=0.70 acc, bit-exact, reconcile."""
    from tokenizers import Tokenizer

    model_id = "XiaomiMiMo/MiMo-7B-RL"
    snap = resolve_source(ModelSourceConfig(model_id, offline=True))
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    prompts = [
        tuple(
            tok.encode(
                "Write a Python function to compute Fibonacci numbers using dynamic programming:\n"
            ).ids
        ),
        tuple(
            tok.encode(
                "Solve step by step: If a train travels at 80 km/h for 2.5 hours, "
                "what distance does it cover?"
            ).ids
        ),
        tuple(tok.encode("Explain the difference between TCP and UDP in networking.").ids),
    ]

    tok_cfg = TokenizerConfig(tokenizer=str(snap), offline=True, skip_tokenizer_init=True)
    plain_cfg = EngineConfig(
        source=ModelSourceConfig(model_id, offline=True),
        tokenizer=tok_cfg,
        device=DeviceConfig(
            device="cuda", dtype=DType.BF16, backends=("flashinfer", "torch"), cuda_graph=True
        ),
        cache=CacheConfig(kv_reserve_tokens=2048),
    )
    plain_tokens: list[tuple[int, ...]] = []
    with LLM.from_config(plain_cfg) as plain_llm:
        for pr in prompts:
            res = plain_llm.generate(pr, SamplingParams(temperature=0.0, max_tokens=48))
            assert not isinstance(res, list)
            plain_tokens.append(res.outputs[0].token_ids)
    del plain_llm
    gc.collect()
    torch.cuda.empty_cache()

    mtp_cfg = EngineConfig(
        source=ModelSourceConfig(model_id, offline=True),
        tokenizer=tok_cfg,
        device=DeviceConfig(
            device="cuda", dtype=DType.BF16, backends=("flashinfer", "torch"), cuda_graph=True
        ),
        cache=CacheConfig(kv_reserve_tokens=2048),
        speculative=SpeculativeConfig(
            enabled=True, method="mtp", controller="fixed", fixed_k=1, max_draft_tokens=1
        ),
    )
    plan = plan_engine(mtp_cfg)
    assert "mtp" in plan.budget.loaded_drafts

    try:
        with LLM.from_config(mtp_cfg) as mtp_llm:
            assert mtp_llm.engine.startup is not None
            assert mtp_llm.engine.startup.reconciliation.matches

            mtp_tokens: list[tuple[int, ...]] = []
            for pr in prompts:
                res = mtp_llm.generate(pr, SamplingParams(temperature=0.0, max_tokens=48))
                assert not isinstance(res, list)
                mtp_tokens.append(res.outputs[0].token_ids)

            assert mtp_tokens == plain_tokens
            stats: Any = mtp_llm.engine.runner.controller
            assert stats is not None
            snap_stats = stats.snapshot()
            assert snap_stats["total_drafts"] > 0
            mean_acc = snap_stats["total_accepted"] / snap_stats["total_drafts"]
            assert mean_acc >= 0.70, f"MiMo-7B-RL mean acceptance {mean_acc:.3f} < 0.70"
            proposer = mtp_llm.engine.runner.proposer
            assert proposer is not None
            assert getattr(proposer, "graph_replays", 0) > 0
    finally:
        gc.collect()
        torch.cuda.empty_cache()
