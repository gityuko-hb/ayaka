"""Clean local reference implementation of z-lab/dflash forward & decode loop.

Derived directly from inspected ``dflash.py`` and ``utils.py`` inside ``z-lab/Qwen3-8B-DFlash-b16``
(no ``trust_remote_code`` execution). The GPU test compares the engine's drafts with it, and the
external comparison benchmark runs its decode loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as functional


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    )
    ang = positions.float().unsqueeze(-1) * inv.unsqueeze(0)
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * weight


def dflash_reference_draft(
    weights: dict[str, torch.Tensor],
    config: Any,
    target_features: torch.Tensor,
    noise_embeddings: torch.Tensor,
    positions: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step stateless reference DFlash draft pass returning ``(logits, block_hidden)``.

    Args:
        weights: DFlash safetensors mapping (``fc.weight``, ``hidden_norm.weight``, ``norm.weight``,
            and ``layers.0..L-1.*``) on target device/dtype.
        config: ``DFlashConfig``-like object.
        target_features: ``[ctx_len, len(target_layer_ids) * hidden_size]``.
        noise_embeddings: ``[blk_len, hidden_size]`` (anchor token followed by masks).
        positions: ``[ctx_len + blk_len]`` concatenated context + block position ids.
        lm_head_weight: Target ``[vocab_size, hidden_size]`` weight matrix.
    """
    eps = float(config.rms_norm_eps)
    theta = float(config.rope_theta)
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    dim = int(config.head_dim)
    ctx_len = int(target_features.shape[0])
    q_len = int(noise_embeddings.shape[0])

    target_hid = _rmsnorm(
        target_features @ weights["fc.weight"].T, weights["hidden_norm.weight"], eps
    )
    cos, sin = _rope_cos_sin(positions, dim, theta, noise_embeddings.dtype)
    cos_u = cos.unsqueeze(1)  # [ctx+q, 1, D]
    sin_u = sin.unsqueeze(1)

    h = noise_embeddings
    for idx in range(int(config.num_hidden_layers)):
        p = f"layers.{idx}."
        residual = h
        hn = _rmsnorm(h, weights[f"{p}input_layernorm.weight"], eps)
        q = (hn @ weights[f"{p}self_attn.q_proj.weight"].T).view(q_len, heads, dim)
        q = _rmsnorm(q, weights[f"{p}self_attn.q_norm.weight"], eps)
        q = q * cos_u[-q_len:] + _rotate_half(q) * sin_u[-q_len:]

        k_ctx = target_hid @ weights[f"{p}self_attn.k_proj.weight"].T
        k_noise = hn @ weights[f"{p}self_attn.k_proj.weight"].T
        k = torch.cat([k_ctx, k_noise], dim=0).view(ctx_len + q_len, kv_heads, dim)
        k = _rmsnorm(k, weights[f"{p}self_attn.k_norm.weight"], eps)
        k = k * cos_u + _rotate_half(k) * sin_u

        v_ctx = target_hid @ weights[f"{p}self_attn.v_proj.weight"].T
        v_noise = hn @ weights[f"{p}self_attn.v_proj.weight"].T
        v = torch.cat([v_ctx, v_noise], dim=0).view(ctx_len + q_len, kv_heads, dim)

        attn = functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=False,
            enable_gqa=True,
        )
        attn_out = attn.squeeze(0).transpose(0, 1).reshape(q_len, -1)
        h = residual + attn_out @ weights[f"{p}self_attn.o_proj.weight"].T

        residual = h
        hn = _rmsnorm(h, weights[f"{p}post_attention_layernorm.weight"], eps)
        gate = functional.silu(hn @ weights[f"{p}mlp.gate_proj.weight"].T)
        up = hn @ weights[f"{p}mlp.up_proj.weight"].T
        h = residual + (gate * up) @ weights[f"{p}mlp.down_proj.weight"].T

    h = _rmsnorm(h, weights["norm.weight"], eps)
    logits = h[1:] @ lm_head_weight.T
    return logits, h


@dataclass(slots=True)
class HFReferenceResult:
    output_ids: list[int]
    tokens_per_sec: float
    mean_accepted_per_step: float
    steps: int
    peak_vram_bytes: int


def run_hf_dflash_reference(
    target_model: Any,
    draft_weights: dict[str, torch.Tensor],
    config: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_ids: set[int] | None = None,
) -> HFReferenceResult:
    """Reference greedy DFlash speculative decode over an HF ``transformers`` target model."""
    device = target_model.device
    dtype = next(target_model.parameters()).dtype
    embed = target_model.get_input_embeddings()
    lm_head_w = target_model.get_output_embeddings().weight
    block_size = int(config.block_size)
    mask_id = int(config.mask_token_id)
    layer_ids = list(config.target_layer_ids)
    eos = eos_ids or set()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = target_model(inp, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        first_tok = int(torch.argmax(out.logits[0, -1], dim=-1).item())
        generated: list[int] = [first_tok]
        if first_tok in eos or max_new_tokens <= 1:
            dt = max(time.perf_counter() - t0, 1e-9)
            return HFReferenceResult(
                generated, len(generated) / dt, 1.0, 1, torch.cuda.max_memory_allocated(device)
            )

        ctx_feats = torch.cat([out.hidden_states[lid + 1][0] for lid in layer_ids], dim=-1).to(
            dtype
        )
        # Keep dynamic cache or crop on target.
        accept_lens: list[int] = []
        cur_pos = len(prompt_ids)
        while len(generated) < max_new_tokens:
            blk_in = torch.tensor(
                [generated[-1]] + [mask_id] * (block_size - 1), dtype=torch.long, device=device
            )
            pos_all = torch.arange(cur_pos + block_size, dtype=torch.long, device=device)
            # Filter ctx_feats to cur_pos entries.
            logits, _ = dflash_reference_draft(
                draft_weights, config, ctx_feats[:cur_pos], embed(blk_in), pos_all, lm_head_w
            )
            drafts = torch.argmax(logits, dim=-1).tolist()
            cand = torch.tensor([[generated[-1], *drafts]], dtype=torch.long, device=device)
            t_out = target_model(
                cand, past_key_values=past, use_cache=True, output_hidden_states=True
            )
            posterior = torch.argmax(t_out.logits[0], dim=-1).tolist()
            acc = 0
            for j in range(block_size - 1):
                if posterior[j] == drafts[j]:
                    acc += 1
                else:
                    break
            bonus = posterior[acc]
            committed = [*drafts[:acc], bonus]
            accept_lens.append(acc + 1)
            new_feats = torch.cat(
                [t_out.hidden_states[lid + 1][0, : acc + 1] for lid in layer_ids], dim=-1
            ).to(dtype)
            ctx_feats = torch.cat([ctx_feats, new_feats], dim=0)
            cur_pos += acc + 1
            # transformers 5 crops by a negative count of tokens to remove, not to a length.
            excess = int(past.get_seq_length()) - cur_pos
            if excess > 0:
                past.crop(-excess)
            for t in committed:
                generated.append(int(t))
                if t in eos or len(generated) >= max_new_tokens:
                    break
            if generated[-1] in eos:
                break
    torch.cuda.synchronize(device)
    dt = max(time.perf_counter() - t0, 1e-9)
    mean_acc = sum(accept_lens) / max(len(accept_lens), 1)
    return HFReferenceResult(
        output_ids=generated[:max_new_tokens],
        tokens_per_sec=len(generated[:max_new_tokens]) / dt,
        mean_accepted_per_step=mean_acc,
        steps=len(accept_lens),
        peak_vram_bytes=int(torch.cuda.max_memory_allocated(device)),
    )
