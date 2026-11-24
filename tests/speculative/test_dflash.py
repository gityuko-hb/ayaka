"""DFlash block drafter: unit tests and Qwen3-8B / Llama-3.1-8B L4 GPU gates (#129)."""

from __future__ import annotations

import gc
import json
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.speculative import SpeculativeConfig
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.core.params import SamplingParams
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")
safetensors_torch: Any = pytest.importorskip("safetensors.torch")

from ayaka.engine.build import plan_engine  # noqa: E402
from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.model_loader.source import resolve_source  # noqa: E402
from ayaka.models.dflash import (  # noqa: E402
    DFlashConfig,
    describe_dflash_bytes,
    describe_dflash_kv_bytes_per_token,
    load_dflash_draft_model,
)
from ayaka.ops.backend import load_backend  # noqa: E402
from dflash_reference import dflash_reference_draft  # noqa: E402
from llama_support import llama_config, llama_state, write_llama  # noqa: E402
from tiny_model import patch_tokenizer_loading, tiny_tokenizer  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from ayaka.configs.model import ModelConfig

pytestmark = pytest.mark.unit


def _write_tiny_dflash(
    directory: Path, target_config: ModelConfig, *, block_size: int = 4, mask_token_id: int = 3
) -> tuple[DFlashConfig, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    target_layers = [0, max(0, target_config.num_layers - 1)]
    raw: dict[str, Any] = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "block_size": block_size,
        "hidden_size": target_config.hidden_size,
        "intermediate_size": target_config.intermediate_size,
        "num_hidden_layers": 2,
        "num_attention_heads": target_config.num_attention_heads,
        "num_key_value_heads": target_config.num_kv_heads,
        "head_dim": target_config.head_dim,
        "rms_norm_eps": target_config.norm_eps,
        "rope_theta": float(target_config.rope_theta or 10000.0),
        "max_position_embeddings": target_config.max_position_embeddings,
        "vocab_size": target_config.vocab_size,
        "dflash_config": {
            "mask_token_id": mask_token_id,
            "target_layer_ids": target_layers,
        },
    }
    (directory / "config.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    cfg = DFlashConfig.from_hf_config(raw)

    gen = torch.Generator().manual_seed(41)
    h, inner = cfg.hidden_size, cfg.intermediate_size
    qr = cfg.num_attention_heads * cfg.head_dim
    kvr = cfg.num_key_value_heads * cfg.head_dim

    def w(*shape: int) -> Any:
        return torch.randn(shape, generator=gen) * 0.05

    state: dict[str, Any] = {
        "fc.weight": w(h, len(target_layers) * h),
        "hidden_norm.weight": w(h),
        "norm.weight": w(h),
    }
    for idx in range(cfg.num_hidden_layers):
        p = f"layers.{idx}."
        state[f"{p}input_layernorm.weight"] = w(h)
        state[f"{p}post_attention_layernorm.weight"] = w(h)
        state[f"{p}self_attn.q_proj.weight"] = w(qr, h)
        state[f"{p}self_attn.k_proj.weight"] = w(kvr, h)
        state[f"{p}self_attn.v_proj.weight"] = w(kvr, h)
        state[f"{p}self_attn.o_proj.weight"] = w(h, qr)
        state[f"{p}self_attn.q_norm.weight"] = w(cfg.head_dim)
        state[f"{p}self_attn.k_norm.weight"] = w(cfg.head_dim)
        state[f"{p}mlp.gate_proj.weight"] = w(inner, h)
        state[f"{p}mlp.up_proj.weight"] = w(inner, h)
        state[f"{p}mlp.down_proj.weight"] = w(h, inner)
    safetensors_torch.save_file(state, str(directory / "model.safetensors"))
    return cfg, state


def test_dflash_model_loading_and_cached_kv_parity(tmp_path: Path) -> None:
    target_cfg = llama_config(num_layers=3, vocab_size=260)
    dflash_dir = tmp_path / "tiny-dflash"
    cfg, weights = _write_tiny_dflash(dflash_dir, target_cfg, block_size=5)

    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    draft = load_dflash_draft_model(dflash_dir, ops=ops, dtype=DType.FP32, device="cpu")
    param_bytes = sum(p.numel() * p.element_size() for p in draft.parameters())
    assert param_bytes == describe_dflash_bytes(cfg, dtype=DType.FP32)
    assert describe_dflash_kv_bytes_per_token(cfg, dtype=DType.FP32) > 0

    gen = torch.Generator().manual_seed(7)
    feats1 = torch.randn((4, len(cfg.target_layer_ids) * cfg.hidden_size), generator=gen)
    feats2 = torch.randn((2, len(cfg.target_layer_ids) * cfg.hidden_size), generator=gen)
    noise = torch.randn((cfg.block_size, cfg.hidden_size), generator=gen)
    lm_head = torch.randn((cfg.vocab_size, cfg.hidden_size), generator=gen)

    with torch.inference_mode():
        ks1, vs1 = draft.compute_context_kv(feats1, torch.arange(0, 4))
        ks2, vs2 = draft.compute_context_kv(feats2, torch.arange(4, 6))
        ks = [torch.cat([a, b], dim=1) for a, b in zip(ks1, ks2, strict=True)]
        vs = [torch.cat([a, b], dim=1) for a, b in zip(vs1, vs2, strict=True)]
        hid_actual = draft.forward_block(noise, torch.arange(6, 6 + cfg.block_size), ks, vs)
        logits_actual = hid_actual[1:] @ lm_head.T

        feats_full = torch.cat([feats1, feats2], dim=0)
        logits_ref, hid_ref = dflash_reference_draft(
            weights, cfg, feats_full, noise, torch.arange(0, 6 + cfg.block_size), lm_head
        )
    assert torch.allclose(hid_actual, hid_ref, atol=1e-5)
    assert torch.allclose(logits_actual, logits_ref, atol=1e-5)


@pytest.mark.parametrize(
    ("threshold", "cap", "override"),
    [(0.0, 0, 0), (0.5, 4, 0), (0.0, 0, 3)],
)
def test_engine_dflash_matches_plain_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threshold: float,
    cap: int,
    override: int,
) -> None:
    patch_tokenizer_loading(monkeypatch)
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    cfg = llama_config(num_layers=3, vocab_size=260)
    write_llama(target_dir, llama_state(cfg, seed=19), cfg)
    _write_tiny_dflash(draft_dir, cfg, block_size=5, mask_token_id=3)

    plain_cfg = EngineConfig(
        source=ModelSourceConfig(str(target_dir), offline=True),
        cache=CacheConfig(block_size=4, num_blocks=65),
    )
    dflash_cfg = EngineConfig(
        source=ModelSourceConfig(str(target_dir), offline=True),
        cache=CacheConfig(block_size=4, num_blocks=65),
        speculative=SpeculativeConfig(
            enabled=True,
            method="dflash",
            draft_model=str(draft_dir),
            max_draft_tokens=4,
            controller="fixed",
            fixed_k=4,
            draft_confidence_threshold=threshold,
            draft_context_cap=cap,
            lookup_override_match=override,
        ),
    )
    plan = plan_engine(dflash_cfg)
    assert plan.budget.draft_bytes > 0
    assert plan.budget.loaded_drafts == ("dflash",)

    prompts = [(1, 4, 5, 6, 4, 5, 6), (1, 7, 8, 9, 10)]
    with LLM.from_config(plain_cfg, tokenizer=tiny_tokenizer()) as plain_llm:
        plain_outs = plain_llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=10))
        assert isinstance(plain_outs, list)
        expected = [o.outputs[0].token_ids for o in plain_outs]

    with LLM.from_config(dflash_cfg, tokenizer=tiny_tokenizer()) as spec_llm:
        outs = spec_llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=10))
        assert isinstance(outs, list)
        assert [o.outputs[0].token_ids for o in outs] == expected


def _generate_tokens(cfg: EngineConfig, prompts: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    outs: list[tuple[int, ...]] = []
    llm = LLM.from_config(cfg)
    try:
        for pr in prompts:
            res = llm.generate(pr, SamplingParams(temperature=0.0, max_tokens=48))
            assert not isinstance(res, list)
            outs.append(res.outputs[0].token_ids)
    finally:
        llm.close()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return outs


def _run_dflash_and_collect(
    cfg: EngineConfig, prompts: list[tuple[int, ...]], draft_id: str
) -> tuple[list[tuple[int, ...]], float, dict[str, Any]]:
    """Run the DFlash engine, then return its tokens, tau and the Gate 1 payload on the CPU.

    Everything that holds GPU memory (engine, runner, target and draft model, K/V) is local to
    this function, so it is all unreachable when it returns and the caller can load the reference
    weights into a free GPU.
    """
    spec_llm: Any = None
    try:
        spec_llm = LLM.from_config(cfg)
        assert spec_llm.engine.startup is not None
        assert spec_llm.engine.startup.reconciliation.matches

        runner: Any = spec_llm.engine.runner
        proposer: Any = runner.proposer
        assert proposer is not None

        captured_features: list[Any] = []
        orig_observe = proposer.observe_features

        def _spy_observe(
            request_ids: Any,
            start_positions: Any,
            rows_per_request: Any,
            emitted_tokens: Any,
            features: Any,
        ) -> None:
            if len(captured_features) < 8:
                captured_features.append(features.detach().clone())
            orig_observe(request_ids, start_positions, rows_per_request, emitted_tokens, features)

        proposer.observe_features = _spy_observe
        try:
            spec_tokens: list[tuple[int, ...]] = []
            for pr in prompts:
                res = spec_llm.generate(pr, SamplingParams(temperature=0.0, max_tokens=48))
                assert not isinstance(res, list)
                spec_tokens.append(res.outputs[0].token_ids)
        finally:
            proposer.observe_features = orig_observe

        ctrl: Any = runner.controller
        assert ctrl is not None
        snap = ctrl.snapshot()
        assert snap["rounds"] > 0
        tau = 1.0 + snap["total_accepted"] / snap["rounds"]

        # Collect Ayaka DFlash predictions and real target hidden states before freeing VRAM.
        assert captured_features, f"{draft_id}: expected captured target features"
        real_feats = torch.cat(captured_features, dim=0)
        ctx_len = min(int(real_feats.shape[0]), 64)
        ctx_feats = real_feats[:ctx_len]
        draft_model = proposer._draft
        dcfg = draft_model.config
        target_model: Any = runner.model

        with torch.inference_mode():
            ks, vs = draft_model.compute_context_kv(
                ctx_feats, torch.arange(0, ctx_len, device="cuda")
            )
            step_k = dcfg.block_size - 1
            num_blocks = (2000 + step_k - 1) // step_k
            anchor_pool = prompts[0] + spec_tokens[0]
            blocks: list[tuple[Any, Any, Any, Any]] = []
            for blk_idx in range(num_blocks):
                anchor = anchor_pool[blk_idx % len(anchor_pool)]
                blk_ids = torch.tensor([anchor, *([dcfg.mask_token_id] * step_k)], device="cuda")
                noise = target_model.embed_tokens(blk_ids)
                start_p = (blk_idx * 7) % 128 + ctx_len
                blk_pos = torch.arange(start_p, start_p + dcfg.block_size, device="cuda")
                h_ayaka = draft_model.forward_block(noise, blk_pos, ks, vs)
                logits_ayaka = target_model.compute_logits(h_ayaka[1:])
                blocks.append(
                    (
                        noise.detach().cpu(),
                        blk_pos.detach().cpu(),
                        h_ayaka.detach().cpu(),
                        logits_ayaka.argmax(-1).detach().cpu(),
                    )
                )
            payload: dict[str, Any] = {
                "dcfg": dcfg,
                "ctx_feats": ctx_feats.detach().cpu(),
                "ctx_len": ctx_len,
                "lm_head_weight": target_model.lm_head.weight.detach().cpu(),
                "blocks": blocks,
            }
    finally:
        if spec_llm is not None:
            spec_llm.close()
    return spec_tokens, tau, payload


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("target_id", "draft_id", "max_k", "min_tau"),
    [
        ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", 15, 2.0),
        ("unsloth/Llama-3.1-8B-Instruct", "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat", 9, 2.0),
    ],
)
def test_dflash_real_checkpoints_gates_on_l4(
    target_id: str, draft_id: str, max_k: int, min_tau: float
) -> None:
    """Gates 1-4 for Qwen3-8B and Llama-3.1-8B DFlash on L4 GPU."""
    tokenizers: Any = pytest.importorskip("tokenizers")  # tests/tokenizers shadows it for pyright

    target_dir = resolve_source(ModelSourceConfig(target_id, offline=True))
    draft_dir = resolve_source(ModelSourceConfig(draft_id, offline=True))
    tok = tokenizers.Tokenizer.from_file(str(target_dir / "tokenizer.json"))
    prompts = [
        tuple(
            tok.encode(
                "def binary_search(nums: list[int], target: int) -> int:\n"
                "    lo, hi = 0, len(nums) - 1\n"
                "    while lo <= hi:\n"
            ).ids
        ),
        tuple(
            tok.encode(
                "HTTP/1.1 200 OK\nContent-Type: application/json\n\n"
                '{"status": "ok", "items": [{"id": 1, "name": "alpha"}, {"id": 2, '
            ).ids
        ),
    ]
    tok_cfg = TokenizerConfig(tokenizer=str(target_dir), offline=True, skip_tokenizer_init=True)

    plain_cfg = EngineConfig(
        source=ModelSourceConfig(target_id, offline=True),
        tokenizer=tok_cfg,
        device=DeviceConfig(
            device="cuda", dtype=DType.BF16, backends=("flashinfer", "torch"), cuda_graph=True
        ),
        cache=CacheConfig(kv_reserve_tokens=2048),
    )
    plain_tokens = _generate_tokens(plain_cfg, prompts)

    dflash_cfg = EngineConfig(
        source=ModelSourceConfig(target_id, offline=True),
        tokenizer=tok_cfg,
        device=DeviceConfig(
            device="cuda", dtype=DType.BF16, backends=("flashinfer", "torch"), cuda_graph=True
        ),
        cache=CacheConfig(kv_reserve_tokens=2048),
        speculative=SpeculativeConfig(
            enabled=True,
            method="dflash",
            draft_model=draft_id,
            max_draft_tokens=max_k,
            controller="fixed",
            fixed_k=max_k,
        ),
    )
    plan = plan_engine(dflash_cfg)
    assert "dflash" in plan.budget.loaded_drafts

    spec_tokens, tau, gate1_payload = _run_dflash_and_collect(dflash_cfg, prompts, draft_id)
    gc.collect()
    torch.cuda.empty_cache()
    assert spec_tokens == plain_tokens
    assert tau >= min_tau, f"{draft_id} Gate 2 tau={tau:.2f} < {min_tau}"

    # Gate 1: compare Ayaka's bf16 draft against the vendored reference run in float32, across
    # >= 2,000 positions. The reference is float32 on purpose: bf16 logits near 16-32 move in steps
    # of 0.0625-0.125, so a bf16 reference cannot say which of two tokens closer than that is the
    # right one (measured on L4: a bf16 reference disagrees with Ayaka at 2.4% and 1.8% of the
    # positions with a margin of 0.05, and with none at a margin of 0.25 against float32). A
    # near tie is therefore a float32 top-2 margin below 0.25, two bf16 steps.
    import torch.nn.functional as functional
    from safetensors import safe_open

    near_tie_margin = 0.25
    raw_weights: dict[str, Any] = {}
    try:
        with safe_open(str(draft_dir / "model.safetensors"), framework="pt", device="cuda") as sf:
            for k in list(sf.keys()):
                raw_weights[k] = sf.get_tensor(k).to(torch.float32)

        dcfg = gate1_payload["dcfg"]
        ctx_feats_gpu = gate1_payload["ctx_feats"].to("cuda", torch.float32)
        ctx_pos_gpu = torch.arange(0, gate1_payload["ctx_len"], device="cuda")
        lm_head_gpu = gate1_payload["lm_head_weight"].to("cuda", torch.float32)
        agreed_non_tie = 0
        evaluated_non_tie = 0
        min_cos = 1.0
        with torch.inference_mode():
            for noise_cpu, blk_pos_cpu, h_ay_cpu, ay_ids_cpu in gate1_payload["blocks"]:
                all_pos = torch.cat([ctx_pos_gpu, blk_pos_cpu.to("cuda")])
                logits_ref, h_ref = dflash_reference_draft(
                    raw_weights,
                    dcfg,
                    ctx_feats_gpu,
                    noise_cpu.to("cuda", torch.float32),
                    all_pos,
                    lm_head_gpu,
                )
                cos = (
                    functional.cosine_similarity(h_ay_cpu.to("cuda").float(), h_ref.float(), dim=-1)
                    .min()
                    .item()
                )
                min_cos = min(min_cos, cos)
                top2 = logits_ref.topk(2, dim=-1).values
                non_tie = (top2[:, 0] - top2[:, 1]) >= near_tie_margin
                evaluated_non_tie += int(non_tie.sum().item())
                agreed_non_tie += int(
                    ((ay_ids_cpu.to("cuda") == logits_ref.argmax(-1)) & non_tie).sum().item()
                )
        assert min_cos >= 0.995, f"{draft_id} Gate 1 min cosine {min_cos:.4f} < 0.995"
        assert evaluated_non_tie >= 1000
        agree = agreed_non_tie / evaluated_non_tie
        print(
            f"{draft_id} Gate 1: min cosine {min_cos:.4f}, parity {agree:.4f} "
            f"({agreed_non_tie}/{evaluated_non_tie} positions with a float32 margin >= "
            f"{near_tie_margin}), tau {tau:.2f}"
        )
        assert agree >= 0.99, (
            f"{draft_id} Gate 1 parity {agree:.4f} ({agreed_non_tie}/{evaluated_non_tie}) < 0.99"
        )
    finally:
        gate1_payload.clear()
        raw_weights.clear()
        del raw_weights
        gc.collect()
        torch.cuda.empty_cache()
