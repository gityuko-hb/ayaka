"""Suite 2: N-gram lookup, cost vs fixed controller, Token Recycling, and Predicted Outputs."""

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
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.engine.llm import LLM
from ayaka.model_loader.source import resolve_source
from ayaka.speculative.policy import CostAwareController, FixedController
from ayaka.utils.dtypes import DType

WORKLOADS: dict[str, str] = {
    "code_edit": (
        "Refactor the Python class below with type annotations and docstrings, keeping exact logic:"
        "\n```python\n"
        "class TokenBucketLimiter:\n"
        "    def __init__(self, capacity, refill_rate):\n"
        "        self.capacity = capacity\n"
        "        self.tokens = capacity\n"
        "        self.refill_rate = refill_rate\n"
        "        self.last_ts = 0.0\n\n"
        "    def allow(self, now, cost=1):\n"
        "        elapsed = max(0.0, now - self.last_ts)\n"
        "        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)\n"
        "        self.last_ts = now\n"
        "        if self.tokens >= cost:\n"
        "            self.tokens -= cost\n"
        "            return True\n"
        "        return False\n"
        "```\n"
        "Rewrite the full updated Python class carefully:\n```python\n"
    ),
    "rag_extract": (
        "Read the technical note and reproduce the configuration block verbatim before summarizing:"
        "\nCONFIG START\n"
        "server.host = '127.0.0.1'\n"
        "server.port = 8443\n"
        "cache.block_size = 16\n"
        "cache.prefix_cache = true\n"
        "speculative.method = 'lookup'\n"
        "speculative.max_draft_tokens = 8\n"
        "speculative.controller = 'cost'\n"
        "CONFIG END\n"
        "First copy the CONFIG block word-for-word, then summarize the settings in 2 sentences:\n"
    ),
    "creative_chat": (
        "Write an imaginative short story about a lighthouse keeper who discovers an unusual glass "
        "compass on the rocky shore after an autumn storm."
    ),
}

PREDICTION_ORIGINAL = (
    "class TokenBucketLimiter:\n"
    "    def __init__(self, capacity: float, refill_rate: float) -> None:\n"
    "        self.capacity = capacity\n"
    "        self.tokens = capacity\n"
    "        self.refill_rate = refill_rate\n"
    "        self.last_ts = 0.0\n\n"
    "    def allow(self, now: float, cost: float = 1.0) -> bool:\n"
    "        elapsed = max(0.0, now - self.last_ts)\n"
    "        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)\n"
    "        self.last_ts = now\n"
    "        if self.tokens >= cost:\n"
    "            self.tokens -= cost\n"
    "            return True\n"
    "        return False\n"
)


def _run_config(
    model_id: str,
    spec_cfg: SpeculativeConfig,
    prompts: dict[str, tuple[int, ...]],
    max_tokens: int,
    *,
    batch_size: int = 1,
    prediction_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    snap = resolve_source(ModelSourceConfig(model_id, offline=True))
    cfg = EngineConfig(
        source=ModelSourceConfig(model_id, offline=True),
        tokenizer=TokenizerConfig(tokenizer=str(snap), offline=True, skip_tokenizer_init=True),
        device=DeviceConfig(
            device="cuda", dtype=DType.BF16, backends=("flashinfer", "torch"), cuda_graph=True
        ),
        cache=CacheConfig(block_size=16, kv_reserve_tokens=2048),
        scheduler=SchedulerConfig(max_running=max(1, batch_size), max_tokens_per_step=512),
        speculative=spec_cfg,
    )
    results: dict[str, Any] = {}
    with LLM.from_config(cfg) as llm:
        # Warmup once
        _ = llm.generate(
            prompts["code_edit"], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        )
        for name, pids in prompts.items():
            reqs: list[Any]
            if batch_size == 1 and name == "code_edit" and prediction_ids:
                reqs = [
                    GenerateRequest(
                        request_id=RequestId(f"{name}-pred"),
                        prompt=pids,
                        sampling=SamplingParams(
                            temperature=0.0, max_tokens=max_tokens, ignore_eos=True
                        ),
                        prediction=prediction_ids,
                    )
                ]
            elif batch_size == 1:
                reqs = [pids]
            else:
                reqs = [pids] * batch_size

            ctrl = llm.engine.runner.controller
            before = (
                ctrl.snapshot()
                if isinstance(ctrl, (CostAwareController, FixedController))
                else None
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = llm.generate(
                reqs if len(reqs) > 1 or isinstance(reqs[0], GenerateRequest) else reqs[0],
                None
                if isinstance(reqs[0], GenerateRequest)
                else SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            out_list = outs if isinstance(outs, list) else [outs]
            toks = sum(len(o.outputs[0].token_ids) for o in out_list)
            after = (
                ctrl.snapshot()
                if isinstance(ctrl, (CostAwareController, FixedController))
                else None
            )
            drafted = (after["total_drafts"] - before["total_drafts"]) if (after and before) else 0
            acc = (after["total_accepted"] - before["total_accepted"]) if (after and before) else 0
            rounds = (after["rounds"] - before["rounds"]) if (after and before) else 0
            results[name] = {
                "tokens": toks,
                "seconds": round(dt, 4),
                "tok_per_sec": round(toks / dt, 2),
                "rounds": rounds,
                "drafted": drafted,
                "accepted": acc,
                "acceptance_rate": round(acc / drafted, 3) if drafted else 0.0,
                "first_tokens": list(out_list[0].outputs[0].token_ids),
                "accepted_prediction_tokens": sum(
                    o.usage.accepted_prediction_tokens for o in out_list
                ),
            }
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lookup, controller, Token Recycling & Prediction benchmark."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--out",
        default="benchmarks/results/2026-11-24-nvidia-l4/speculative/lookup_bench.json",
    )
    args = parser.parse_args()

    snap = resolve_source(ModelSourceConfig(args.model, offline=True))
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    prompts = {k: tuple(tok.encode(v).ids) for k, v in WORKLOADS.items()}
    pred_ids = tuple(tok.encode(PREDICTION_ORIGINAL).ids)

    configs: list[tuple[str, int, SpeculativeConfig, tuple[int, ...]]] = [
        ("plain_b1", 1, SpeculativeConfig(enabled=False), ()),
        (
            "lookup_cost_b1",
            1,
            SpeculativeConfig(enabled=True, method="lookup", controller="cost", max_draft_tokens=8),
            (),
        ),
        (
            "lookup_fixed4_b1",
            1,
            SpeculativeConfig(
                enabled=True, method="lookup", controller="fixed", fixed_k=4, max_draft_tokens=4
            ),
            (),
        ),
        (
            "lookup_prediction_b1",
            1,
            SpeculativeConfig(enabled=True, method="lookup", controller="cost", max_draft_tokens=8),
            pred_ids,
        ),
        (
            "token_recycling_b1",
            1,
            SpeculativeConfig(
                enabled=True, method="token_recycling", controller="cost", max_draft_tokens=6
            ),
            (),
        ),
        ("plain_b4", 4, SpeculativeConfig(enabled=False), ()),
        (
            "lookup_cost_b4",
            4,
            SpeculativeConfig(enabled=True, method="lookup", controller="cost", max_draft_tokens=8),
            (),
        ),
    ]

    report: dict[str, Any] = {}
    baseline_b1: dict[str, list[int]] = {}
    for label, bsz, scfg, pids in configs:
        res = _run_config(
            args.model, scfg, prompts, args.max_tokens, batch_size=bsz, prediction_ids=pids
        )
        if label == "plain_b1":
            baseline_b1 = {k: v["first_tokens"] for k, v in res.items()}
        elif bsz == 1:
            for k, v in res.items():
                v["exact_match_plain"] = v["first_tokens"] == baseline_b1[k]
                v["speedup_vs_plain"] = round(
                    v["tok_per_sec"] / max(report["plain_b1"][k]["tok_per_sec"], 1e-6), 3
                )
        else:
            for k, v in res.items():
                if "plain_b4" in report:
                    v["speedup_vs_plain"] = round(
                        v["tok_per_sec"] / max(report["plain_b4"][k]["tok_per_sec"], 1e-6), 3
                    )
        # Drop huge token lists from output summary
        for v in res.values():
            v.pop("first_tokens", None)
        report[label] = res

    write({"machine": machine(), "model": args.model, "results": report}, args.out)


if __name__ == "__main__":
    main()
