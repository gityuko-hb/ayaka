"""The DFlash block drafters of Qwen3-8B and Llama-3.1-8B against plain decoding (#129)."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

import torch

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _machine import machine, write

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.configs.speculative import SpeculativeConfig
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.core.params import SamplingParams
from ayaka.engine.build import plan_engine
from ayaka.engine.llm import LLM
from ayaka.model_loader.source import resolve_source
from ayaka.speculative.policy import CostAwareController, FixedController
from ayaka.utils.dtypes import DType

PROMPTS = {
    "code": (
        "Write a Python function to merge two sorted linked lists in-place cleanly:\n```python\n"
    ),
    "math": "Prove cleanly why the sum of the first n odd positive integers equals n^2.",
    "chat": "Explain the trade-offs between LSM-trees and B-trees for write-heavy databases.",
}


def _run_one(
    target_id: str,
    label: str,
    spec_cfg: SpeculativeConfig,
    prompts: dict[str, tuple[int, ...]],
    max_tokens: int,
    *,
    batch_size: int = 1,
) -> dict[str, Any]:
    snap = resolve_source(ModelSourceConfig(target_id, offline=True))
    cfg = EngineConfig(
        source=ModelSourceConfig(target_id, offline=True),
        tokenizer=TokenizerConfig(tokenizer=str(snap), offline=True, skip_tokenizer_init=True),
        device=DeviceConfig(
            device="cuda", dtype=DType.BF16, backends=("flashinfer", "torch"), cuda_graph=True
        ),
        cache=CacheConfig(block_size=16, kv_reserve_tokens=2048),
        scheduler=SchedulerConfig(max_running=max(1, batch_size), max_tokens_per_step=512),
        speculative=spec_cfg,
    )
    plan = plan_engine(cfg)
    row: dict[str, Any] = {
        "target": target_id,
        "label": label,
        "batch_size": batch_size,
        "draft_bytes": plan.budget.draft_bytes,
        "displaced_kv_blocks": plan.displaced_kv_blocks,
        "displaced_kv_tokens": plan.displaced_kv_blocks * plan.block_size,
        "prompts": {},
    }
    with LLM.from_config(cfg) as llm:
        row["reconciled"] = bool(llm.engine.startup and llm.engine.startup.reconciliation.matches)
        _ = llm.generate(
            prompts["code"], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        )
        torch.cuda.reset_peak_memory_stats()
        total_toks = 0
        total_sec = 0.0
        for name, pids in prompts.items():
            ctrl = llm.engine.runner.controller
            before = (
                ctrl.snapshot()
                if isinstance(ctrl, (CostAwareController, FixedController))
                else None
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            in_arg = pids if batch_size == 1 else [pids] * batch_size
            out = llm.generate(
                in_arg, SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            out_list = out if isinstance(out, list) else [out]
            toks = sum(len(o.outputs[0].token_ids) for o in out_list)
            total_toks += toks
            total_sec += dt
            after = (
                ctrl.snapshot()
                if isinstance(ctrl, (CostAwareController, FixedController))
                else None
            )
            drafted = (after["total_drafts"] - before["total_drafts"]) if (after and before) else 0
            acc = (after["total_accepted"] - before["total_accepted"]) if (after and before) else 0
            rounds = (after["rounds"] - before["rounds"]) if (after and before) else 0
            row["prompts"][name] = {
                "tokens": toks,
                "seconds": round(dt, 4),
                "tok_per_sec": round(toks / dt, 2),
                "rounds": rounds,
                "drafted": drafted,
                "accepted": acc,
                "tau": round(1.0 + acc / rounds, 3) if rounds else 1.0,
                "token_ids": list(out_list[0].outputs[0].token_ids),
            }
        row["mean_tok_per_sec"] = round(total_toks / max(total_sec, 1e-9), 2)
        row["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash block drafter benchmark.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--out",
        default="benchmarks/results/2026-11-24-nvidia-l4/speculative/dflash_bench.json",
    )
    args = parser.parse_args()

    pairs = [
        ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", 15),
        ("unsloth/Llama-3.1-8B-Instruct", "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat", 9),
    ]
    all_rows: list[dict[str, Any]] = []
    for target_id, draft_id, max_k in pairs:
        snap = resolve_source(ModelSourceConfig(target_id, offline=True))
        tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
        prompts = {k: tuple(tok.encode(v).ids) for k, v in PROMPTS.items()}
        variants: list[tuple[str, int, SpeculativeConfig]] = [
            ("plain_b1", 1, SpeculativeConfig(enabled=False)),
            (
                "dflash_b1_t0",
                1,
                SpeculativeConfig(
                    enabled=True,
                    method="dflash",
                    draft_model=draft_id,
                    max_draft_tokens=max_k,
                    controller="fixed",
                    fixed_k=max_k,
                    draft_confidence_threshold=0.0,
                ),
            ),
            (
                "dflash_b1_conf05",
                1,
                SpeculativeConfig(
                    enabled=True,
                    method="dflash",
                    draft_model=draft_id,
                    max_draft_tokens=max_k,
                    controller="fixed",
                    fixed_k=max_k,
                    draft_confidence_threshold=0.5,
                ),
            ),
            (
                "dflash_b1_cap256",
                1,
                SpeculativeConfig(
                    enabled=True,
                    method="dflash",
                    draft_model=draft_id,
                    max_draft_tokens=max_k,
                    controller="fixed",
                    fixed_k=max_k,
                    draft_context_cap=256,
                ),
            ),
            (
                "dflash_b1_hybrid6",
                1,
                SpeculativeConfig(
                    enabled=True,
                    method="dflash",
                    draft_model=draft_id,
                    max_draft_tokens=max_k,
                    controller="fixed",
                    fixed_k=max_k,
                    lookup_override_match=6,
                ),
            ),
            (
                "dflash_b2_t0",
                2,
                SpeculativeConfig(
                    enabled=True,
                    method="dflash",
                    draft_model=draft_id,
                    max_draft_tokens=max_k,
                    controller="fixed",
                    fixed_k=max_k,
                ),
            ),
        ]
        plain_tps = 1.0
        plain_tokens: dict[str, list[int]] = {}
        for label, bsz, scfg in variants:
            try:
                r = _run_one(target_id, label, scfg, prompts, args.max_tokens, batch_size=bsz)
                if label == "plain_b1":
                    plain_tps = r["mean_tok_per_sec"]
                    plain_tokens = {k: v["token_ids"] for k, v in r["prompts"].items()}
                r["speedup_vs_plain_b1"] = round(r["mean_tok_per_sec"] / max(plain_tps, 1e-6), 3)
                r["exact_match_plain"] = all(
                    v["token_ids"] == plain_tokens[k] for k, v in r["prompts"].items()
                )
                for v in r["prompts"].values():
                    v.pop("token_ids", None)
                all_rows.append(r)
            except Exception as exc:
                all_rows.append({"target": target_id, "label": label, "error": str(exc)})

    write({"machine": machine(), "results": all_rows}, args.out)


if __name__ == "__main__":
    main()
