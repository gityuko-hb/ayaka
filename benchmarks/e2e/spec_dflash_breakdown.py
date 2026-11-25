"""Where a DFlash round spends its time: Ayaka's engine against the vendored reference loop.

Same target (Qwen3-8B bf16), same drafter, same prompts (the sweep's code_edit and prose with about
1,000 tokens of context), 256 greedy tokens, one request. Every timed call is bracketed by
``torch.cuda.synchronize()``, so the totals are wall time of device work plus host work, and the
run is slower than an untimed one; the split is what matters.

``PROBE_PREFIX_CACHE=0`` turns Ayaka's prefix cache off. The warm-up request is the code_edit
prompt, so with the cache on the measured requests read their prompt (or its shared context) from
the cache, and the drafter sees no features for those positions. ``PROBE_REFERENCE=0`` skips the
reference loop.

Usage::

    uv run python benchmarks/e2e/spec_dflash_breakdown.py out.json
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "_support"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from ayaka.configs.cache import CacheConfig  # noqa: E402
from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.engine import EngineConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.configs.scheduler import SchedulerConfig  # noqa: E402
from ayaka.configs.speculative import SpeculativeConfig  # noqa: E402
from ayaka.configs.tokenizer import TokenizerConfig  # noqa: E402
from ayaka.core.params import SamplingParams  # noqa: E402
from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.model_loader.source import resolve_source  # noqa: E402
from ayaka.models.dflash import DFlashConfig  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402

TARGET = "Qwen/Qwen3-8B"
DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
TOKENS = 256
OUT = Path(sys.argv[1])

spec = importlib.util.spec_from_file_location("sweep", ROOT / "benchmarks/e2e/speculative_sweep.py")
if spec is None or spec.loader is None:
    raise ImportError("cannot load benchmarks/e2e/speculative_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

totals: dict[str, float] = defaultdict(float)
calls: dict[str, int] = defaultdict(int)


def timed(name: str, fn: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        totals[name] += time.perf_counter() - started
        calls[name] += 1
        return result

    return wrapper


def ayaka(snapshot: Path, prompts: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    config = EngineConfig(
        source=ModelSourceConfig(TARGET, offline=True),
        tokenizer=TokenizerConfig(tokenizer=str(snapshot), offline=True, skip_tokenizer_init=True),
        device=DeviceConfig(
            device="cuda", dtype=DType.BF16, backends=("flashinfer", "torch"), cuda_graph=True
        ),
        cache=CacheConfig(
            block_size=16,
            num_blocks=1024,
            kv_reserve_tokens=8192,
            prefix_cache=os.environ.get("PROBE_PREFIX_CACHE", "1") == "1",
        ),
        scheduler=SchedulerConfig(max_running=1, max_tokens_per_step=512),
        speculative=SpeculativeConfig(
            enabled=True,
            method="dflash",
            draft_model=DRAFT,
            controller="fixed",
            fixed_k=15,
            max_draft_tokens=15,
        ),
    )
    report: dict[str, Any] = {}
    with LLM.from_config(config) as llm:
        runner = llm.engine.runner
        proposer = runner.proposer
        params = SamplingParams(temperature=0.0, max_tokens=TOKENS, ignore_eos=True)
        llm.generate(prompts["code_edit"], SamplingParams(temperature=0.0, max_tokens=16))
        runner.execute = timed("step (verify, draft and feature update inside)", runner.execute)
        proposer.propose = timed("draft: propose", proposer.propose)
        proposer.observe_features = timed("draft: context update", proposer.observe_features)
        for name, ids in prompts.items():
            totals.clear()
            calls.clear()
            before = runner.controller.snapshot()
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = llm.generate(ids, params)
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            after = runner.controller.snapshot()
            rounds = after["rounds"] - before["rounds"]
            accepted = after["total_accepted"] - before["total_accepted"]
            step = totals["step (verify, draft and feature update inside)"]
            draft = totals["draft: propose"] + totals["draft: context update"]
            report[name] = {
                "tokens": len(out.outputs[0].token_ids),
                "wall_seconds": round(wall, 4),
                "tok_per_sec_timed": round(len(out.outputs[0].token_ids) / wall, 2),
                "rounds": rounds,
                "accepted_per_round": round(accepted / max(rounds, 1), 3),
                "tokens_per_round": round(len(out.outputs[0].token_ids) / max(rounds, 1), 3),
                "seconds": {k: round(v, 4) for k, v in totals.items()},
                "calls": dict(calls),
                "verify_and_rest_of_step_seconds": round(step - draft, 4),
                "outside_steps_seconds": round(wall - step, 4),
                "ids": list(out.outputs[0].token_ids),
            }
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return report


def reference(snapshot: Path, prompts: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    import dflash_reference

    model: Any = (
        AutoModelForCausalLM.from_pretrained(
            str(snapshot), dtype=torch.bfloat16, local_files_only=True
        )
        .cuda()
        .eval()
    )
    d_snap = resolve_source(ModelSourceConfig(DRAFT, offline=True))
    cfg = DFlashConfig.from_directory(d_snap)
    weights: dict[str, torch.Tensor] = {}
    with safe_open(str(d_snap / "model.safetensors"), framework="pt", device="cuda") as sf:
        for key in list(sf.keys()):
            weights[key] = sf.get_tensor(key).to(torch.bfloat16)
    draft_fn = dflash_reference.dflash_reference_draft
    forward = model.forward
    report: dict[str, Any] = {}
    with torch.inference_mode():
        dflash_reference.run_hf_dflash_reference(
            model, weights, cfg, list(prompts["code_edit"]), 16
        )
        dflash_reference.dflash_reference_draft = timed(
            "draft: forward over the whole context", draft_fn
        )
        model.forward = timed("target forward (prefill, then verify)", forward)
        for name, ids in prompts.items():
            totals.clear()
            calls.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = dflash_reference.run_hf_dflash_reference(
                model, weights, cfg, list(ids), TOKENS
            )
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            report[name] = {
                "tokens": len(result.output_ids),
                "wall_seconds": round(wall, 4),
                "tok_per_sec_timed": round(len(result.output_ids) / wall, 2),
                "rounds": result.steps,
                "tokens_per_round": round(result.mean_accepted_per_step, 3),
                "seconds": {k: round(v, 4) for k, v in totals.items()},
                "calls": dict(calls),
                "ids": result.output_ids,
            }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    snapshot = resolve_source(ModelSourceConfig(TARGET, offline=True))
    prompts = {name: sweep._prompts(snapshot, name, 1, 1000)[0] for name in ("code_edit", "prose")}
    result = {
        "target": TARGET,
        "draft": DRAFT,
        "tokens": TOKENS,
        "prefix_cache": os.environ.get("PROBE_PREFIX_CACHE", "1") == "1",
        "prompt_tokens": {k: len(v) for k, v in prompts.items()},
        "ayaka": ayaka(snapshot, prompts),
        "reference": reference(snapshot, prompts)
        if os.environ.get("PROBE_REFERENCE", "1") == "1"
        else {},
    }
    for name in prompts if result["reference"] else ():
        a, r = result["ayaka"][name], result["reference"][name]
        same = (
            a["ids"][: min(len(a["ids"]), len(r["ids"]))]
            == r["ids"][: min(len(a["ids"]), len(r["ids"]))]
        )
        result.setdefault("same_ids_as_reference", {})[name] = same
    from _machine import machine

    result["machine"] = machine()
    OUT.write_text(json.dumps(result, indent=1))
    for side in ("ayaka", "reference"):
        for name, row in result[side].items():
            print(side, name, {k: v for k, v in row.items() if k != "ids"})


if __name__ == "__main__":
    main()
