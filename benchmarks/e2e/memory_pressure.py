"""The memory milestone's exit conditions on a real model: preemption and an injected step OOM.

One engine, a device KV pool far smaller than the work sent to it:

    pressure     ``--requests`` requests at once whose prompts fit the pool but whose answers need
                 about ``--oversubscribe`` times its blocks. The scheduler must preempt and
                 recompute; every request must finish, and the KV manager's integrity check must
                 pass after every step.
    step_oom     the same requests again, with ``torch.OutOfMemoryError`` raised once inside a
                 decode step. The engine thread must stay up and every request finish.
    resize       the pool grown and shrunk at runtime between requests, then the same requests.

Each phase also records, per request, the first token that differs from the request run alone
(``first_difference``). The control runs each request alone twice: identical runs there, and
differences only in the batched phases, mean the differences come from batch shapes in bf16
kernels, not from preemption or recovery. The script exits non-zero when a condition fails.

    uv run python benchmarks/e2e/memory_pressure.py --model Qwen/Qwen3-8B --out pressure.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import ForwardMode, RequestId
from ayaka.engine.llm import LLM
from ayaka.utils.dtypes import COMPUTE_DTYPES

_DTYPES = {d.label: d for d in COMPUTE_DTYPES}
_BLOCK = 16


def _prompt(seed: int, length: int) -> tuple[int, ...]:
    return tuple(1000 + (seed * 7919 + i * 104729) % 50000 for i in range(length))


_TOPICS = (
    "how a CPU cache hierarchy works, from registers to main memory",
    "the history of the printing press and what it changed in Europe",
    "how to design a rate limiter for a public HTTP API",
    "the water cycle and how mountains shape the weather around them",
)


def _text_prompts(model: str, count: int, length: int) -> list[tuple[int, ...]]:
    """Real instructions, about ``length`` tokens each: a model is confident on them, so greedy
    paths do not flip on rounding as they do on random ids."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    out: list[tuple[int, ...]] = []
    for index in range(count):
        topic = _TOPICS[index % len(_TOPICS)]
        notes = " ".join(f"Point {n}: keep the explanation concrete." for n in range(400))
        text = f"Write a long, detailed explanation of {topic}.\n{notes}"
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        # Keep the template's end (the closed user turn and the assistant header) whole.
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        tail = list(ids[len(ids) - 1 - ids[::-1].index(end) :]) if end in ids else list(ids[-8:])
        out.append(tuple(list(ids[: length - len(tail)]) + tail))
    return out


def _requests(tag: str, prompts: list[tuple[int, ...]], tokens: int) -> list[GenerateRequest]:
    sampling = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
    return [
        GenerateRequest(request_id=RequestId(f"{tag}-{i}"), prompt=p, sampling=sampling)
        for i, p in enumerate(prompts)
    ]


def _ids(outputs: Any) -> list[tuple[int, ...]]:
    return [tuple(o.outputs[0].token_ids) for o in outputs]


def _first_difference(got: list[tuple[int, ...]], want: list[tuple[int, ...]]) -> list[int | None]:
    """Per request, the index of the first token that differs from its run alone (None: equal)."""
    out: list[int | None] = []
    for a, b in zip(got, want, strict=True):
        index = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
        out.append(index if index is not None or len(a) == len(b) else min(len(a), len(b)))
    return out


def _run_checked(llm: LLM, requests: list[GenerateRequest]) -> tuple[list[Any], int]:
    """Run the requests stepwise, checking the KV integrity after every step."""
    engine = llm.engine
    for request in requests:
        engine.add_request(request)
    done: dict[str, Any] = {}
    steps = 0
    while engine.has_unfinished:
        for output in engine.step():
            if output.finished:
                done[str(output.request_id)] = output
        engine._kv.check_integrity()  # the condition under test
        steps += 1
    return [done[str(r.request_id)] for r in requests], steps


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    parser.add_argument("--backends", nargs="+", default=["flashinfer", "torch"])
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--oversubscribe", type=float, default=2.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    need = args.requests * ((args.prompt_tokens + args.tokens) // _BLOCK + 1)
    blocks = int(need / args.oversubscribe) + 1
    config = EngineConfig(
        source=ModelSourceConfig(args.model, offline=True),
        device=DeviceConfig(
            device="cuda", dtype=_DTYPES[args.dtype], backends=tuple(args.backends)
        ),
        cache=CacheConfig(
            block_size=_BLOCK,
            num_blocks=blocks,
            prefix_cache=False,
            # The reserve one request must hold at admission: its prompt and a little more,
            # so all requests are admitted and then outgrow the pool while they decode.
            kv_reserve_tokens=args.prompt_tokens + 64,
        ),
        scheduler=SchedulerConfig(max_running=args.requests, max_tokens_per_step=2048),
    )
    prompts = _text_prompts(args.model, args.requests, args.prompt_tokens)
    report: dict[str, Any] = {
        "model": args.model,
        "pool_blocks": blocks,
        "blocks_needed_at_once": need,
        "requests": args.requests,
        "prompt_tokens": args.prompt_tokens,
        "tokens": args.tokens,
    }
    ok = True
    with LLM.from_config(config) as llm:
        engine = llm.engine
        roster = engine._scheduler._roster
        preempt = roster.preempt
        counted = {"preemptions": 0}

        def counting_preempt(*a: Any, **k: Any) -> None:
            # Scheduler.num_preempted counts requests preempted now; this counts every preemption.
            counted["preemptions"] += 1
            preempt(*a, **k)

        roster.preempt = counting_preempt  # type: ignore[method-assign]
        alone: list[tuple[int, ...]] = []
        again: list[tuple[int, ...]] = []
        for index, prompt in enumerate(prompts):
            outputs, _ = _run_checked(llm, _requests(f"alone{index}", [prompt], args.tokens))
            alone.extend(_ids(outputs))
            outputs, _ = _run_checked(llm, _requests(f"again{index}", [prompt], args.tokens))
            again.extend(_ids(outputs))
        # The control: one request alone, twice. Any difference here is the engine's own
        # run-to-run variation, not the memory milestone's.
        report["alone_twice_equal"] = again == alone
        report["alone_twice_first_difference"] = _first_difference(again, alone)

        before = counted["preemptions"]
        start = time.perf_counter()
        outputs, steps = _run_checked(llm, _requests("pressure", prompts, args.tokens))
        report["pressure"] = {
            "seconds": round(time.perf_counter() - start, 2),
            "steps": steps,
            "preemptions": counted["preemptions"] - before,
            "finished": [str(o.outputs[0].finish_reason) for o in outputs],
            "token_ids_equal_alone": _ids(outputs) == alone,
            "first_difference": _first_difference(_ids(outputs), alone),
            "integrity_checked_every_step": True,
        }
        ok &= report["pressure"]["preemptions"] > 0 and all(
            reason == "length" for reason in report["pressure"]["finished"]
        )

        runner = engine._runner
        original = runner._forward
        tripped = {"count": 0}

        def inject(step: Any, inputs: Any) -> Any:
            if tripped["count"] == 0 and step.mode is ForwardMode.DECODE:
                import torch

                tripped["count"] += 1
                raise torch.OutOfMemoryError("CUDA out of memory (injected)")
            return original(step, inputs)

        runner._forward = inject
        try:
            outputs, _ = _run_checked(llm, _requests("oom", prompts, args.tokens))
        finally:
            runner._forward = original
        report["step_oom"] = {
            "injected": tripped["count"],
            "finished": [str(o.outputs[0].finish_reason) for o in outputs],
            "token_ids_equal_alone": _ids(outputs) == alone,
            "first_difference": _first_difference(_ids(outputs), alone),
            "engine_alive": engine._dead is None,  # what the condition asks
        }
        ok &= report["step_oom"]["engine_alive"] and all(
            reason == "length" for reason in report["step_oom"]["finished"]
        )

        grown = engine.resize_kv_pool(blocks * 2)
        shrunk = engine.resize_kv_pool(blocks)
        outputs, _ = _run_checked(llm, _requests("resize", prompts, args.tokens))
        report["resize"] = {
            "grown_to": list(grown),
            "shrunk_to": list(shrunk),
            "finished": [str(o.outputs[0].finish_reason) for o in outputs],
            "token_ids_equal_alone": _ids(outputs) == alone,
            "first_difference": _first_difference(_ids(outputs), alone),
        }
        ok &= all(reason == "length" for reason in report["resize"]["finished"])
    report["all_conditions_met"] = bool(ok)
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
