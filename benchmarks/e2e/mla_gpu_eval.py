"""Evaluate DeepSeek V2/V3 MLA on NVIDIA A100 with FlashInfer CUDA graphs vs Hugging Face."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import struct
import sys
import tempfile
import time
from dataclasses import replace as replace_dc
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from _machine import machine, write

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model import ModelConfig
from ayaka.configs.model_source import ModelSourceConfig, RequestFormat
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.types import KVLayoutKind
from ayaka.engine.build import plan_engine
from ayaka.engine.llm import LLM
from ayaka.kvcache.storage.spec import KVSpec, LatentKVGroup
from ayaka.utils.dtypes import DType

_ITEMSIZE = {"BF16": 2, "F32": 4}


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        bf16 = tensor.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
        raw = bf16.flatten().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(bf16.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    hdr = json.dumps(header).encode("utf-8")
    hdr_padded = hdr + b" " * (-len(hdr) % 8)
    path.write_bytes(struct.pack("<Q", len(hdr_padded)) + hdr_padded + bytes(payload))


def _write_byte_tokenizer(directory: Path, vocab_size: int) -> None:
    tokenizers = importlib.import_module("tokenizers")
    models = importlib.import_module("tokenizers.models")
    pre_toks = importlib.import_module("tokenizers.pre_tokenizers")
    vocab = {f"tok{i}": i for i in range(vocab_size)}
    tok = tokenizers.Tokenizer(models.WordLevel(vocab, "tok0"))
    tok.pre_tokenizer = pre_toks.Whitespace()
    tok.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "tok2", "unk_token": "tok0"}), encoding="utf-8"
    )


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _mla_hf_config_dict() -> dict[str, Any]:
    return {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "vocab_size": 256,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "q_lora_rank": 256,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "rope_interleave": True,
        "n_routed_experts": 0,
        "num_experts_per_tok": 0,
        "moe_intermediate_size": 0,
        "n_shared_experts": 0,
        "first_k_dense_replace": 4,
    }


def _build_and_eval(ckpt_dir: Path) -> dict[str, Any]:
    transformers = importlib.import_module("transformers")
    raw_cfg = _mla_hf_config_dict()
    hf_cfg = transformers.DeepseekV2Config(**raw_cfg, attn_implementation="eager")
    torch.manual_seed(42)
    ref = transformers.DeepseekV2ForCausalLM(hf_cfg).to(torch.bfloat16)
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "norm" in name:
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.04)

    (ckpt_dir / "config.json").write_text(json.dumps(raw_cfg), encoding="utf-8")
    _write_byte_tokenizer(ckpt_dir, vocab_size=256)
    _write_safetensors(ckpt_dir / "model.safetensors", dict(ref.state_dict()))

    prompt = tuple(range(5, 37))
    ref_cuda = ref.to("cuda").eval()
    ids_t = torch.tensor([prompt], dtype=torch.int64, device="cuda")
    with torch.inference_mode():
        seq = ref_cuda.generate(
            input_ids=ids_t,
            attention_mask=torch.ones_like(ids_t),
            max_new_tokens=16,
            do_sample=False,
        )
    hf_tokens = [int(t) for t in seq[0, len(prompt) :].tolist()]
    del ref_cuda, ref
    _clear_cuda()

    engine_cfg = EngineConfig(
        source=ModelSourceConfig(str(ckpt_dir), format=RequestFormat.SAFETENSORS, offline=True),
        device=DeviceConfig(
            device="cuda",
            dtype=DType.BF16,
            backends=("flashinfer", "torch"),
            cuda_graph=True,
            graph_buckets=(1, 2, 4),
        ),
        cache=CacheConfig(block_size=16, kv_reserve_tokens=2048, prefix_cache=True),
        scheduler=SchedulerConfig(max_running=4, max_tokens_per_step=16),
    )
    plan = plan_engine(engine_cfg)
    if not isinstance(plan.group, LatentKVGroup):
        raise RuntimeError(f"expected LatentKVGroup, got {type(plan.group)}")
    latent_spec = plan.group.spec
    params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    with LLM.from_config(engine_cfg) as llm:
        cold = llm.generate(prompt, params)
        if isinstance(cold, list) or cold.error is not None:
            raise RuntimeError(f"MLA cold request failed: {cold}")
        t0 = time.perf_counter()
        warm = llm.generate((*prompt[:-4], 91, 92, 93, 94), params)
        warm_ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(warm, list) or warm.error is not None:
            raise RuntimeError(f"MLA warm request failed: {warm}")
        cold_tokens = [int(t) for t in cold.outputs[0].token_ids]
        warm_cached_tokens = warm.usage.cached_tokens

    # Canonical full-scale 60-layer DeepSeek V2 ratio: (512 + 64) / (2 * 16 * 128) = 0.140625
    full_latent = replace_dc(latent_spec, num_layers=60)
    full_gqa = KVSpec(
        num_layers=60,
        num_kv_heads=16,
        head_dim=128,
        block_size=16,
        dtype=DType.BF16,
        layout=KVLayoutKind.NHD,
    )
    checkpoint_latent_ratio = latent_spec.compression_ratio_vs_kv(
        num_kv_heads=raw_cfg["num_key_value_heads"],
        head_dim=raw_cfg["head_dim"],
    )
    return {
        "architecture": "DeepseekV2ForCausalLM",
        "kv_lora_rank": raw_cfg["kv_lora_rank"],
        "qk_rope_head_dim": raw_cfg["qk_rope_head_dim"],
        "backends": dict(plan.providers),
        "cuda_graph_buckets": [1, 2, 4],
        "prompt_len": len(prompt),
        "chunk_tokens": 16,
        "hf_tokens": hf_tokens,
        "ayaka_tokens": cold_tokens,
        "exact_greedy_match_vs_hf": cold_tokens == hf_tokens,
        "warm_prefix_cached_tokens": warm_cached_tokens,
        "warm_generate_16_tokens_ms": warm_ms,
        "plan": {
            "kv_groups": plan.kv_groups,
            "latent_kv_bytes": plan.latent_kv_bytes,
            "bytes_per_token": int(plan.group.spec.bytes_per_token()),
            "kv_blocks": plan.kv_blocks,
        },
        "checkpoint_4layer_ratio_vs_standard_kv": checkpoint_latent_ratio,
        "full_60layer_bytes_per_token_latent": int(full_latent.bytes_per_token()),
        "full_60layer_bytes_per_token_gqa16": int(full_gqa.bytes_per_token()),
        "full_60layer_compression_ratio": full_latent.compression_ratio_vs_kv(
            num_kv_heads=16, head_dim=128
        ),
        "weights_bytes": plan.weights_bytes,
        "model_config_ok": ModelConfig.from_hf_config(raw_cfg).kv_lora_rank == 512,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlashInfer MLA GPU parity and capacity eval.")
    parser.add_argument(
        "--out",
        default="benchmarks/results/2026-11-27-nvidia-a100-sxm4-80gb/mla-gpu-parity-and-capacity.json",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        result = _build_and_eval(Path(tmp))
    payload = {
        "benchmark": "mla_gpu_parity_and_capacity",
        "note": (
            "Evaluated on NVIDIA A100-SXM4-80GB using a synthetic 4-layer bf16 DeepSeek-V2 MLA "
            "checkpoint matching canonical FlashInfer geometry (kv_lora_rank=512, "
            "qk_rope_head_dim=64); no multi-hundred-GB DeepSeek checkpoint is cached locally."
        ),
        "machine": machine(),
        "result": result,
    }
    print(
        f"[mla-gpu] exact={result['exact_greedy_match_vs_hf']} "
        f"ratio={result['checkpoint_4layer_ratio_vs_standard_kv']:.4f} "
        f"warm_cached={result['warm_prefix_cached_tokens']} "
        f"warm_16tok_ms={result['warm_generate_16_tokens_ms']:.1f}",
        flush=True,
    )
    write(payload, args.out)
    if not result["exact_greedy_match_vs_hf"]:
        raise RuntimeError("MLA GPU greedy mismatch against transformers!")


if __name__ == "__main__":
    main()
