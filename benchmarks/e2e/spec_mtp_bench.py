"""Suite 3: MiMo-7B-RL Multi-Token Prediction (MTP) ablation benchmark (#128)."""

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
        "Write a Python function to solve the 0/1 knapsack problem using dynamic programming, "
        "including item reconstruction:\n```python\n"
    ),
    "math": (
        "Solve step by step: A train leaves station A at 9:00 AM at 72 km/h. Another train leaves "
        "station B 450 km away at 10:00 AM toward A at 88 km/h. At what time do they meet?"
    ),
    "chat": (
        "Compare TCP and UDP across ordering guarantees, congestion control, header overhead, and "
        "typical real-world use cases."
    ),
}


def _bench_variant(
    model_id: str,
    label: str,
    spec_cfg: SpeculativeConfig,
    prompts: dict[str, tuple[int, ...]],
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
    plan = plan_engine(cfg)
    row: dict[str, Any] = {
        "label": label,
        "draft_bytes": plan.budget.draft_bytes,
        "kv_blocks": plan.kv_blocks,
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
            out = llm.generate(
                pids, SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            single = out[0] if isinstance(out, list) else out
            toks = len(single.outputs[0].token_ids)
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
                "acceptance_rate": round(acc / drafted, 3) if drafted else 0.0,
                "token_ids": list(single.outputs[0].token_ids),
            }
        row["mean_tok_per_sec"] = round(total_toks / max(total_sec, 1e-9), 2)
        row["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        proposer = llm.engine.runner.proposer
        row["mtp_graph_replays"] = int(getattr(proposer, "graph_replays", 0)) if proposer else 0
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="MiMo-7B-RL MTP ablation benchmark.")
    parser.add_argument("--model", default="XiaomiMiMo/MiMo-7B-RL")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--out",
        default="benchmarks/results/2026-11-24-nvidia-l4/speculative/mtp_bench.json",
    )
    args = parser.parse_args()

    snap = resolve_source(ModelSourceConfig(args.model, offline=True))
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    prompts = {k: tuple(tok.encode(v).ids) for k, v in PROMPTS.items()}

    variants: list[tuple[str, SpeculativeConfig]] = [
        ("plain", SpeculativeConfig(enabled=False)),
        (
            "mtp_d1_fixed",
            SpeculativeConfig(
                enabled=True,
                method="mtp",
                controller="fixed",
                fixed_k=1,
                max_draft_tokens=1,
                mtp_depth=1,
            ),
        ),
        (
            "mtp_d1_cost",
            SpeculativeConfig(
                enabled=True,
                method="mtp",
                controller="cost",
                fixed_k=1,
                max_draft_tokens=1,
                mtp_depth=1,
            ),
        ),
        (
            "mtp_d2_exp",
            SpeculativeConfig(
                enabled=True,
                method="mtp",
                controller="fixed",
                fixed_k=2,
                max_draft_tokens=2,
                mtp_depth=2,
            ),
        ),
        (
            "mtp_vocab16k_exp",
            SpeculativeConfig(
                enabled=True,
                method="mtp",
                controller="fixed",
                fixed_k=1,
                max_draft_tokens=1,
                mtp_vocab_limit=16384,
            ),
        ),
        (
            "mtp_hybrid_lookup6",
            SpeculativeConfig(
                enabled=True,
                method="mtp",
                controller="cost",
                max_draft_tokens=6,
                mtp_depth=1,
                lookup_override_match=6,
            ),
        ),
    ]

    results: list[dict[str, Any]] = []
    plain_tokens: dict[str, list[int]] = {}
    plain_tps = 1.0
    for label, scfg in variants:
        r = _bench_variant(args.model, label, scfg, prompts, args.max_tokens)
        if label == "plain":
            plain_tps = r["mean_tok_per_sec"]
            plain_tokens = {k: v["token_ids"] for k, v in r["prompts"].items()}
        r["speedup_vs_plain"] = round(r["mean_tok_per_sec"] / max(plain_tps, 1e-6), 3)
        r["exact_match_plain"] = all(
            v["token_ids"] == plain_tokens[k] for k, v in r["prompts"].items()
        )
        for v in r["prompts"].values():
            v.pop("token_ids", None)
        results.append(r)

    write({"machine": machine(), "model": args.model, "variants": results}, args.out)


if __name__ == "__main__":
    main()
