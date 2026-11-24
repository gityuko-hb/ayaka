"""Suite 5: Apple-to-apple external library comparison on L4 GPU.

Compares on identical prompts (greedy, max_new_tokens=128):
  - Ayaka plain baseline vs Ayaka speculative (lookup / mtp / dflash)
  - Hugging Face transformers plain (AutoModelForCausalLM.generate)
  - Hugging Face transformers prompt_lookup_num_tokens=4
  - Vendored z-lab/dflash reference loop (tests/_support/dflash_reference.py)
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tests" / "_support"))

from _machine import machine, write  # noqa: E402

from ayaka.configs.cache import CacheConfig  # noqa: E402
from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.engine import EngineConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.configs.speculative import SpeculativeConfig  # noqa: E402
from ayaka.configs.tokenizer import TokenizerConfig  # noqa: E402
from ayaka.core.params import SamplingParams  # noqa: E402
from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.model_loader.source import resolve_source  # noqa: E402
from ayaka.models.dflash import DFlashConfig  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402
from dflash_reference import run_hf_dflash_reference  # noqa: E402

PROMPTS = {
    "code_edit": (
        "Refactor this Python cache class with type hints while keeping exact logic:\n"
        "```python\n"
        "class RingCache:\n"
        "    def __init__(self, cap):\n"
        "        self.cap = cap\n"
        "        self.items = []\n"
        "    def push(self, x):\n"
        "        if len(self.items) >= self.cap:\n"
        "            self.items.pop(0)\n"
        "        self.items.append(x)\n"
        "```\nRewrite verbatim with type annotations:\n```python\n"
    ),
    "reasoning": "Solve cleanly step by step: What is the sum of all primes between 10 and 40?",
}


def _bench_ayaka(
    model_id: str,
    prompts: dict[str, tuple[int, ...]],
    spec_cfg: SpeculativeConfig,
    max_tokens: int,
) -> dict[str, Any]:
    snap = resolve_source(ModelSourceConfig(model_id, offline=True))
    cfg = EngineConfig(
        source=ModelSourceConfig(model_id, offline=True),
        tokenizer=TokenizerConfig(tokenizer=str(snap), offline=True, skip_tokenizer_init=True),
        device=DeviceConfig(
            device="cuda", dtype=DType.BF16, backends=("flashinfer", "torch"), cuda_graph=True
        ),
        cache=CacheConfig(block_size=16, kv_reserve_tokens=2048),
        speculative=spec_cfg,
    )
    out_map: dict[str, Any] = {}
    with LLM.from_config(cfg) as llm:
        _ = llm.generate(
            prompts["code_edit"], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        )
        torch.cuda.reset_peak_memory_stats()
        for name, pids in prompts.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = llm.generate(
                pids, SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            single = res[0] if isinstance(res, list) else res
            toks = list(single.outputs[0].token_ids)
            out_map[name] = {
                "tok_per_sec": round(len(toks) / dt, 2),
                "seconds": round(dt, 4),
                "tokens": len(toks),
                "token_ids": toks,
            }
        vram = int(torch.cuda.max_memory_allocated())
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return {"prompts": out_map, "peak_vram_bytes": vram}


def _bench_hf(
    model_id: str,
    prompts: dict[str, tuple[int, ...]],
    max_tokens: int,
    *,
    prompt_lookup: int | None = None,
    dflash_id: str | None = None,
    is_mimo: bool = False,
) -> dict[str, Any]:
    snap = resolve_source(ModelSourceConfig(model_id, offline=True))
    model: Any
    if is_mimo:
        # MiMo's main path is standard Qwen2ForCausalLM (modeling_mimo inherits Qwen2ForCausalLM).
        raw_cfg = AutoConfig.from_pretrained(
            str(snap), local_files_only=True, trust_remote_code=False
        )
        q2_cfg = Qwen2Config(
            **{k: v for k, v in raw_cfg.to_dict().items() if k in Qwen2Config().to_dict()}
        )
        q2_cfg.tie_word_embeddings = False
        q2_cls: Any = Qwen2ForCausalLM
        model = (
            q2_cls.from_pretrained(
                str(snap), config=q2_cfg, torch_dtype=torch.bfloat16, local_files_only=True
            )
            .cuda()
            .eval()
        )
    else:
        auto_cls: Any = AutoModelForCausalLM
        model = (
            auto_cls.from_pretrained(str(snap), torch_dtype=torch.bfloat16, local_files_only=True)
            .cuda()
            .eval()
        )

    draft_weights: dict[str, torch.Tensor] | None = None
    dflash_cfg: DFlashConfig | None = None
    if dflash_id is not None:
        d_snap = resolve_source(ModelSourceConfig(dflash_id, offline=True))
        dflash_cfg = DFlashConfig.from_directory(d_snap)
        draft_weights = {}
        with safe_open(str(d_snap / "model.safetensors"), framework="pt", device="cuda") as sf:
            for k in list(sf.keys()):
                draft_weights[k] = sf.get_tensor(k).to(torch.bfloat16)

    out_map: dict[str, Any] = {}
    with torch.inference_mode():
        warm = torch.tensor([list(prompts["code_edit"])], dtype=torch.long, device="cuda")
        _ = model.generate(warm, max_new_tokens=8, do_sample=False)
        torch.cuda.reset_peak_memory_stats()
        for name, pids in prompts.items():
            if dflash_cfg is not None and draft_weights is not None:
                r = run_hf_dflash_reference(
                    model, draft_weights, dflash_cfg, list(pids), max_tokens
                )
                out_map[name] = {
                    "tok_per_sec": round(r.tokens_per_sec, 2),
                    "tokens": len(r.output_ids),
                    "mean_accepted_per_step": round(r.mean_accepted_per_step, 3),
                    "token_ids": r.output_ids,
                }
            else:
                inp = torch.tensor([list(pids)], dtype=torch.long, device="cuda")
                kwargs: dict[str, Any] = {
                    "max_new_tokens": max_tokens,
                    "min_new_tokens": max_tokens,
                    "do_sample": False,
                }
                if prompt_lookup is not None:
                    kwargs["prompt_lookup_num_tokens"] = prompt_lookup
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                gen = model.generate(inp, **kwargs)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                toks = [int(t) for t in gen[0, len(pids) :].tolist()]
                out_map[name] = {
                    "tok_per_sec": round(len(toks) / dt, 2),
                    "seconds": round(dt, 4),
                    "tokens": len(toks),
                    "token_ids": toks,
                }
        vram = int(torch.cuda.max_memory_allocated())
    del model, draft_weights
    gc.collect()
    torch.cuda.empty_cache()
    return {"prompts": out_map, "peak_vram_bytes": vram}


def main() -> None:
    parser = argparse.ArgumentParser(description="External baseline comparison on L4 GPU.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--out",
        default="benchmarks/results/2026-11-24-nvidia-l4/speculative/external_compare.json",
    )
    args = parser.parse_args()

    comparisons: list[dict[str, Any]] = []

    # 1. Qwen/Qwen3-8B: Ayaka plain/lookup/dflash vs HF plain/prompt_lookup/dflash ref
    q3_snap = resolve_source(ModelSourceConfig("Qwen/Qwen3-8B", offline=True))
    q3_tok = Tokenizer.from_file(str(q3_snap / "tokenizer.json"))
    q3_prompts = {k: tuple(q3_tok.encode(v).ids) for k, v in PROMPTS.items()}
    for engine_label, runner_fn in [
        (
            "ayaka_plain",
            lambda: _bench_ayaka(
                "Qwen/Qwen3-8B", q3_prompts, SpeculativeConfig(enabled=False), args.max_tokens
            ),
        ),
        (
            "ayaka_lookup_cost",
            lambda: _bench_ayaka(
                "Qwen/Qwen3-8B",
                q3_prompts,
                SpeculativeConfig(
                    enabled=True, method="lookup", controller="cost", max_draft_tokens=8
                ),
                args.max_tokens,
            ),
        ),
        (
            "ayaka_dflash",
            lambda: _bench_ayaka(
                "Qwen/Qwen3-8B",
                q3_prompts,
                SpeculativeConfig(
                    enabled=True,
                    method="dflash",
                    draft_model="z-lab/Qwen3-8B-DFlash-b16",
                    max_draft_tokens=15,
                    controller="fixed",
                    fixed_k=15,
                ),
                args.max_tokens,
            ),
        ),
        ("transformers_plain", lambda: _bench_hf("Qwen/Qwen3-8B", q3_prompts, args.max_tokens)),
        (
            "transformers_prompt_lookup_4",
            lambda: _bench_hf("Qwen/Qwen3-8B", q3_prompts, args.max_tokens, prompt_lookup=4),
        ),
        (
            "zlab_dflash_ref_hf",
            lambda: _bench_hf(
                "Qwen/Qwen3-8B", q3_prompts, args.max_tokens, dflash_id="z-lab/Qwen3-8B-DFlash-b16"
            ),
        ),
    ]:
        try:
            res = runner_fn()
            for p in res["prompts"].values():
                p.pop("token_ids", None)
            comparisons.append({"model": "Qwen/Qwen3-8B", "runner": engine_label, **res})
        except Exception as exc:
            comparisons.append(
                {"model": "Qwen/Qwen3-8B", "runner": engine_label, "error": repr(exc)}
            )
        # Keep what is measured so far if a later runner dies.
        write({"machine": machine(), "comparisons": comparisons}, args.out)

    # 2. XiaomiMiMo/MiMo-7B-RL: Ayaka plain, Ayaka mtp vs HF Qwen2ForCausalLM plain
    mm_snap = resolve_source(ModelSourceConfig("XiaomiMiMo/MiMo-7B-RL", offline=True))
    mm_tok = Tokenizer.from_file(str(mm_snap / "tokenizer.json"))
    mm_prompts = {k: tuple(mm_tok.encode(v).ids) for k, v in PROMPTS.items()}
    for engine_label, runner_fn in [
        (
            "ayaka_plain",
            lambda: _bench_ayaka(
                "XiaomiMiMo/MiMo-7B-RL",
                mm_prompts,
                SpeculativeConfig(enabled=False),
                args.max_tokens,
            ),
        ),
        (
            "ayaka_mtp",
            lambda: _bench_ayaka(
                "XiaomiMiMo/MiMo-7B-RL",
                mm_prompts,
                SpeculativeConfig(
                    enabled=True, method="mtp", controller="fixed", fixed_k=1, max_draft_tokens=1
                ),
                args.max_tokens,
            ),
        ),
        (
            "transformers_qwen2_plain",
            lambda: _bench_hf("XiaomiMiMo/MiMo-7B-RL", mm_prompts, args.max_tokens, is_mimo=True),
        ),
    ]:
        try:
            res = runner_fn()
            for p in res["prompts"].values():
                p.pop("token_ids", None)
            comparisons.append({"model": "XiaomiMiMo/MiMo-7B-RL", "runner": engine_label, **res})
        except Exception as exc:
            comparisons.append(
                {"model": "XiaomiMiMo/MiMo-7B-RL", "runner": engine_label, "error": str(exc)}
            )

    write({"machine": machine(), "comparisons": comparisons}, args.out)


if __name__ == "__main__":
    main()
