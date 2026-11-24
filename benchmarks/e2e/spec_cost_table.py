"""Suite 1: Verify step latency cost table T(B, k) across batch sizes and draft lengths."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

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
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import ForwardMode, RequestId
from ayaka.engine.build import build_engine
from ayaka.model_loader.source import resolve_source
from ayaka.utils.dtypes import DType


def _measure_model(
    model_id: str,
    dtype_label: str,
    *,
    cuda_graph: bool,
    batches: list[int],
    draft_ks: list[int],
    repeats: int,
) -> list[dict[str, Any]]:
    snap = resolve_source(ModelSourceConfig(model_id, offline=True))
    dtype = DType.from_str(dtype_label)
    max_b = max(batches)
    max_k = max(draft_ks)
    cfg = EngineConfig(
        source=ModelSourceConfig(model_id, offline=True),
        tokenizer=TokenizerConfig(tokenizer=str(snap), offline=True, skip_tokenizer_init=True),
        device=DeviceConfig(
            device="cuda",
            dtype=dtype,
            backends=("flashinfer", "torch"),
            cuda_graph=cuda_graph,
            graph_buckets=(1, 2, 4, 8),
        ),
        # A small fixed cache: the table times steps of at most 8 short requests, and the memory
        # left over is the scratch of a 128-row verify window (the default sizing OOMs, see
        # cost_table_oom_default_kv.txt next to the result).
        cache=CacheConfig(block_size=16, num_blocks=130, kv_reserve_tokens=512),
        scheduler=SchedulerConfig(max_running=max_b, max_tokens_per_step=512),
        speculative=SpeculativeConfig(
            enabled=True, method="lookup", controller="fixed", fixed_k=max_k, max_draft_tokens=max_k
        ),
    )
    core = build_engine(cfg)
    rows: list[dict[str, Any]] = []
    try:
        runner = core.runner
        sched = core.scheduler
        # Admit max_b short requests and prefill them once.
        rids: list[RequestId] = []
        for i in range(max_b):
            rid = RequestId(f"cost-{i}")
            rids.append(rid)
            sched.add_request(
                TokenizedRequest(
                    request_id=rid,
                    prompt_token_ids=tuple(100 + j for j in range(32)),
                    sampling=SamplingParams(temperature=0.0, max_tokens=200, ignore_eos=True),
                )
            )
        # Prefill every request, asking the scheduler for `max_k` look-ahead rows each so that the
        # first decode step it hands out is a VERIFY step with KV room for the widest window. That
        # step is the template of every timed step; it is never completed, so the scheduler is not
        # called again (it refuses a second `schedule()` before `update()`).
        lookahead = dict.fromkeys(rids, max_k)
        template = None
        while (step := sched.schedule(lookahead)) is not None:
            if step.mode is ForwardMode.PREFILL:
                sched.update(runner.execute(step))
            else:
                template = step
                break
        if template is None or template.mode is not ForwardMode.VERIFY:
            raise RuntimeError("the scheduler did not hand out a VERIFY template step")

        # Every timed step reruns the same positions, so the KV does not grow between repeats.
        for b in batches:
            for k in draft_ks:
                times: list[float] = []
                used_graph = False
                for rep in range(repeats + 2):
                    st = replace(
                        template,
                        mode=ForwardMode.VERIFY if k > 0 else ForwardMode.DECODE,
                        request_ids=template.request_ids[:b],
                        batch_rows=template.batch_rows[:b],
                        rows_per_request=(1 + k,) * b,
                        start_positions=template.start_positions[:b],
                        tokens=template.tokens[:b],
                        need_logits=template.need_logits[:b],
                        sampling_params=template.sampling_params[:b],
                        sequence_indices=template.sequence_indices[:b],
                        output_lengths=template.output_lengths[:b],
                    )
                    drafts = ([tuple(10 + j for j in range(k))] * b) if k > 0 else None
                    before_replays = (
                        (runner.graph.verify_replays + runner.graph.replays)
                        if runner.graph is not None
                        else 0
                    )
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = runner.execute(st, drafts=drafts)
                    torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) * 1e3
                    after_replays = (
                        (runner.graph.verify_replays + runner.graph.replays)
                        if runner.graph is not None
                        else 0
                    )
                    if rep >= 2:
                        times.append(dt)
                        used_graph = used_graph or (after_replays > before_replays)
                rows.append(
                    {
                        "model": model_id,
                        "dtype": dtype_label,
                        "cuda_graph": cuda_graph,
                        "replayed_graph": used_graph,
                        "providers": {str(k): v for k, v in runner.model.ops.providers.items()},
                        "batch_size": b,
                        "k": k,
                        "median_ms": round(statistics.median(times), 3) if times else None,
                        "min_ms": round(min(times), 3) if times else None,
                    }
                )
    finally:
        core.shutdown()
        del core
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify step latency cost table T(B, k).")
    parser.add_argument(
        "--out",
        default="benchmarks/results/2026-11-24-nvidia-l4/speculative/cost_table.json",
    )
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    batches = [1, 2, 4, 8]
    ks = [0, 1, 2, 3, 4, 8, 15]
    all_rows: list[dict[str, Any]] = []
    for model_id, dtype_label, graph in [
        ("Qwen/Qwen3-8B", "bf16", True),
        ("Qwen/Qwen3-8B", "bf16", False),
        ("Qwen/Qwen3-8B-FP8", "bf16", True),
        ("XiaomiMiMo/MiMo-7B-RL", "bf16", True),
    ]:
        try:
            all_rows.extend(
                _measure_model(
                    model_id,
                    dtype_label,
                    cuda_graph=graph,
                    batches=batches,
                    draft_ks=ks,
                    repeats=args.repeats,
                )
            )
        except Exception as exc:
            all_rows.append(
                {"model": model_id, "dtype": dtype_label, "cuda_graph": graph, "error": str(exc)}
            )

    write({"machine": machine(), "suite": "cost_table", "rows": all_rows}, args.out)


if __name__ == "__main__":
    main()
