"""What a constrained decode step costs, and how much of it a CUDA graph wins back.

A request with an output constraint (a JSON schema, a strict tool, a forced tool call) used to run
its decode steps eagerly: the runner replayed a graph only when no row needed a mask. Now the graph
applies the mask, which the host computes before the replay. This script measures the step either
way, so the same run puts numbers on the gap before the change and on what the change won; the
``graph_replays`` of the ``constrained`` row say which path the engine took. It separates what it
measured from what it only estimates.

Measured, per batch size, on a real engine (greedy, ``ignore_eos``, one prompt per request):

    free_graph     no constraint, CUDA graphs on     the step a graph replays today
    eager_control  a zero logit bias, graphs on      the same eager step without a grammar: the
                                                     control for what the grammar adds, and what
                                                     the graph saves (same engine, same ops)
    constrained    JSON schema, CUDA graphs on       the step with a mask active: eager before the
                                                     graph applied masks, a replay after
    free_eager     no constraint, CUDA graphs off    only with ``--free-eager``: another engine, ops
                                                     loaded without the graph, for reference only

Each is the decode time per step: a run of ``--tokens`` tokens minus a run of one token (the
prefill), divided by ``--tokens - 1``, repeated ``--repeat`` times after a warm-up. Each row also
records the graph replays it caused: ``free_graph`` must have some and ``eager_control`` none, or
the script stops, because the arm would not measure what it is named for. ``constrained`` is
whatever the engine does with a mask.

Measured in isolation, per batch size, to split the constrained step into its parts:

    host_mask_us_per_row   the grammar matcher's ``mask()`` for one row (llguidance), per token
    host_controls_ms       ``build_controls`` for the whole batch: on the CPU, and with the copy
                           to the GPU
    gpu_mask_ms            the float32 copy of the logits and ``apply_token_bitmask`` on the GPU

Estimated, never measured (``estimates``): what a fused graph with the mask inside (D1: the mask is
computed on the host before the replay) and a graph split at the logits (D2: the host computes the
mask while the forward graph runs) would cost. They combine the numbers above and are there to
decide whether a prototype is worth writing, not to stand in for one.

    uv run python benchmarks/e2e/constrained_decode.py --model /path/to/SmolLM2-135M \
        --out benchmarks/results/<date>-<machine>/constrained-decode.json

Needs the ``structured`` extra (llguidance) and a GPU. Run it on the dev GPU; a number from another
machine says nothing about this one.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _machine import machine, write

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.constraints import ConstraintKind, StructuredParams
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.engine.llm import LLM
from ayaka.utils.dtypes import DType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.core.outputs import RequestOutput

PROMPT = "The history of the printing press begins"
_COUNTER = itertools.count()


def _schema(min_length: int) -> dict[str, Any]:
    """One long string: the grammar cannot finish before the run ends, so every step is masked."""
    return {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": min_length}},
        "required": ["text"],
        "additionalProperties": False,
    }


def _requests(
    batch: int, params: SamplingParams, structured: StructuredParams | None
) -> list[GenerateRequest]:
    return [
        GenerateRequest(
            request_id=RequestId(f"cd-{next(_COUNTER)}"),
            prompt=PROMPT,
            sampling=params,
            structured=structured,
        )
        for _ in range(batch)
    ]


def _generate(
    llm: LLM, batch: int, tokens: int, structured: StructuredParams | None, *, bias: bool = False
) -> tuple[float, list[RequestOutput]]:
    """Run ``batch`` requests of ``tokens`` tokens; check they all ran the whole length.

    ``bias`` sets a logit bias of zero on one token: it changes nothing but makes the step need a
    transform, so it runs eagerly like a masked step does, without a grammar.
    """
    params = SamplingParams(
        temperature=0.0, max_tokens=tokens, ignore_eos=True, logit_bias={0: 0.0} if bias else None
    )
    started = time.perf_counter()
    result = llm.generate(_requests(batch, params, structured))
    elapsed = time.perf_counter() - started
    outputs = result if isinstance(result, list) else [result]
    for output in outputs:
        if output.error is not None:
            raise RuntimeError(f"a request failed: {output.error.message}")
        completion = output.outputs[0]
        if output.usage.completion_tokens != tokens or completion.finish_reason is not (
            FinishReason.LENGTH
        ):
            raise RuntimeError(
                f"a request produced {output.usage.completion_tokens} of {tokens} tokens "
                f"(finish: {completion.finish_reason}); the constraint ended the answer early, so "
                "the steps are not all masked. Raise --min-length or pass a longer schema."
            )
    return elapsed, outputs


def _per_step_ms(
    llm: LLM,
    batch: int,
    tokens: int,
    repeat: int,
    structured: StructuredParams | None,
    *,
    bias: bool = False,
) -> tuple[list[float], list[RequestOutput]]:
    # Warm-up: compiles the grammar, sets kernels up.
    warm, _ = _generate(llm, batch, tokens, structured, bias=bias)
    _progress(f"  warm-up {warm:.1f} s")
    runs: list[float] = []
    last: list[RequestOutput] = []
    for index in range(repeat):
        prefill, _ = _generate(llm, batch, 1, structured, bias=bias)
        total, last = _generate(llm, batch, tokens, structured, bias=bias)
        runs.append((total - prefill) / (tokens - 1) * 1e3)
        _progress(f"  run {index + 1}/{repeat}: {runs[-1]:.2f} ms/step ({total:.1f} s)")
    return runs, last


def _progress(message: str) -> None:
    """One line on stderr per finished run, so a stalled benchmark shows where it stopped."""
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    sys.stderr.flush()


def _replays(llm: LLM) -> int:
    graph = llm.engine.runner.graph
    return graph.replays if graph is not None else 0


def _engine(args: argparse.Namespace, *, graphs: bool) -> LLM:
    top = max(args.batch)
    # A bucket for every batch measured, or the "free_graph" arm would fall back to eager.
    buckets = tuple(
        sorted({size for size in (1, 2, 4, 8, 16, 32, 64) if size <= top} | {*args.batch})
    )
    config = EngineConfig(
        source=ModelSourceConfig(args.model, offline=True),
        device=DeviceConfig(
            device="cuda",
            dtype=DType.from_str(args.dtype),
            backends=tuple(args.backends),
            cuda_graph=graphs,
            graph_buckets=buckets,
        ),
        # Room for every request to hold its whole answer: no preemption inside a measurement.
        cache=CacheConfig(block_size=16, kv_reserve_tokens=top * (args.tokens + 64)),
        scheduler=SchedulerConfig(max_running=top, max_tokens_per_step=512),
    )
    return LLM.from_config(config)


def _engine_arms(
    args: argparse.Namespace, llm: LLM, structured: StructuredParams
) -> tuple[list[dict[str, Any]], list[int]]:
    """The three step times per batch size, and the tokens of one constrained answer."""
    rows: list[dict[str, Any]] = []
    answer: list[int] = []
    if llm.engine.runner.graph is None:
        raise RuntimeError("the graph engine captured no graphs; check device.cuda_graph")
    for batch in args.batch:
        for arm, constraint, bias in (
            ("free_graph", None, False),
            ("eager_control", None, True),
            ("constrained", structured, False),
        ):
            _progress(f"{arm} batch {batch}")
            before = _replays(llm)
            runs, last = _per_step_ms(llm, batch, args.tokens, args.repeat, constraint, bias=bias)
            replays = _replays(llm) - before
            # The arm is what it says only if the graph did, or did not, run the steps.
            replays_expected = {"free_graph": True, "eager_control": False}.get(arm)
            if replays_expected is not None and replays_expected != (replays > 0):
                raise RuntimeError(
                    f"{arm} at batch {batch}: {replays} graph replays; the arm does not "
                    "measure what it is named for (is the batch above the largest bucket?)"
                )
            if constraint is not None and batch == args.batch[0]:
                answer = list(last[0].outputs[0].token_ids)
            rows.append(_row(arm, batch, runs, replays))
    return rows, answer


def _free_eager_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Reference only: a free decode on a second engine that never captured a graph."""
    rows: list[dict[str, Any]] = []
    with _engine(args, graphs=False) as llm:
        for batch in args.batch:
            _progress(f"free_eager batch {batch}")
            runs, _ = _per_step_ms(llm, batch, args.tokens, args.repeat, None)
            rows.append(_row("free_eager", batch, runs, 0))
    return rows


def _row(arm: str, batch: int, runs: Sequence[float], replays: int) -> dict[str, Any]:
    median = statistics.median(runs)
    return {
        "arm": arm,
        "batch": batch,
        "decode_ms_per_step": [round(x, 3) for x in runs],
        "median_ms": round(median, 3),
        "tokens_per_s": round(batch * 1e3 / median, 1),
        "graph_replays": replays,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _isolated(
    args: argparse.Namespace,
    structured: StructuredParams,
    answer: Sequence[int],
    compiler: Any,
) -> dict[str, Any]:
    """The parts of a masked step, each timed alone (the engine only supplies the compiler)."""
    import torch

    from ayaka.ops.backends.torch.sampling import apply_token_bitmask
    from ayaka.sampling.logits import LogitRow, build_controls
    from ayaka.sampling.structured.runtime import ConstraintRuntime

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if len(answer) < 2:
        raise RuntimeError("no constrained answer to replay through the matcher")
    out: dict[str, Any] = {"mask_tokens_timed": len(answer)}
    vocab = compiler.vocab_size
    constraint_id = compiler.compile(structured)

    # llguidance, one row, over the tokens of a real constrained answer.
    matcher = compiler.matcher(constraint_id)
    micros: list[float] = []
    accepts: list[float] = []
    for token in answer:
        started = time.perf_counter_ns()
        matcher.mask()
        micros.append((time.perf_counter_ns() - started) / 1e3)
        started = time.perf_counter_ns()
        matcher.accept([token])
        accepts.append((time.perf_counter_ns() - started) / 1e3)
    for key, values in (("host_mask_us_per_row", micros), ("host_accept_us_per_row", accepts)):
        out[key] = {
            "p50": round(statistics.median(values), 1),
            "p95": round(_percentile(values, 0.95), 1),
            "max": round(max(values), 1),
        }

    out["per_batch"] = {}
    warm = min(16, max(len(answer) - 1, 0))
    for batch in args.batch:
        runtime = ConstraintRuntime()
        rows: list[LogitRow] = []
        for index in range(batch):
            request_id = RequestId(f"iso-{index}")
            runtime.admit(request_id, compiler.matcher(constraint_id))
            runtime.accept(request_id, answer[:warm])
            rows.append(LogitRow(request_id, SamplingParams(temperature=0.0), warm))

        def controls(device: str, rows: list[LogitRow] = rows, runtime: Any = runtime) -> Any:
            return build_controls(
                rows,
                vocab_size=vocab,
                eos_token_ids=(),
                penalties=None,
                device=device,
                constraints=runtime,
            )

        def timed(device: str, controls: Any = controls) -> float:
            torch.cuda.synchronize()
            started = time.perf_counter()
            controls(device)
            torch.cuda.synchronize()
            return (time.perf_counter() - started) * 1e3

        timed("cuda")  # warm-up
        cpu_ms = statistics.median(timed("cpu") for _ in range(args.repeat * 5))
        cuda_ms = statistics.median(timed("cuda") for _ in range(args.repeat * 5))

        bitmask = controls("cuda").bitmask
        logits = torch.randn(batch, vocab, dtype=dtype, device="cuda")
        float_ms = _events(lambda logits=logits: logits.float(), args.repeat * 10)
        both_ms = _events(
            lambda logits=logits, bitmask=bitmask: apply_token_bitmask(logits.float(), bitmask),
            args.repeat * 10,
        )
        out["per_batch"][batch] = {
            "host_controls_cpu_ms": round(cpu_ms, 3),
            "host_controls_cuda_ms": round(cuda_ms, 3),
            "gpu_float_copy_ms": round(float_ms, 4),
            "gpu_mask_ms": round(both_ms, 4),
        }
    return out


def _events(call: Any, repeat: int) -> float:
    """Median GPU milliseconds of ``call()`` between two CUDA events, after warm-up."""
    import torch

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeat):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def _estimates(
    rows: Sequence[dict[str, Any]], isolated: dict[str, Any], gate_pct: float
) -> dict[str, Any]:
    """Combine the measured parts into the two designs' step times, and the verdict on the gate."""
    median = {(row["arm"], row["batch"]): row["median_ms"] for row in rows}
    out: dict[str, Any] = {
        "note": (
            "estimates, not measurements: each adds measured parts. D1 = free_graph + gpu_mask + "
            "the whole host mask cost, serial before the replay. D2 = free_graph + gpu_mask + the "
            "part of the host mask cost that the forward (taken as free_graph) cannot hide."
        ),
        "gate_pct": gate_pct,
        "per_batch": {},
    }
    accept_ms = isolated["host_accept_us_per_row"]["p50"] / 1e3
    for batch, parts in isolated["per_batch"].items():
        free = median[("free_graph", batch)]
        control = median[("eager_control", batch)]
        constrained = median[("constrained", batch)]
        host = parts["host_controls_cuda_ms"]
        gpu = parts["gpu_mask_ms"]
        gap_pct = (constrained / free - 1) * 100
        # What the parts measured alone explain of the grammar's extra cost over the control, and
        # what they leave unexplained: a large remainder is a cost this script does not see.
        explained = host + accept_ms * batch + max(0.0, gpu - parts["gpu_float_copy_ms"])
        reference = median.get(("free_eager", batch))
        out["per_batch"][batch] = {
            "gap_constrained_vs_free_graph_pct": round(gap_pct, 1),
            "graph_saves_ms": round(control - free, 3),
            "free_eager_other_engine_ms": None if reference is None else round(reference, 3),
            "grammar_extra_over_control_ms": round(constrained - control, 3),
            "explained_by_parts_ms": round(explained, 3),
            "unexplained_ms": round(constrained - control - explained, 3),
            "d1_fused_ms": round(free + gpu + host + accept_ms * batch, 3),
            "d2_split_ms": round(free + gpu + max(0.0, host + accept_ms * batch - free), 3),
            "gate_met": gap_pct >= gate_pct,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost of a constrained decode step.")
    parser.add_argument("--model", required=True, help="a checkpoint directory")
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--backends", nargs="+", default=["torch"], help="op backends, e.g. flashinfer torch"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2048,
        help="minLength of the schema's string, so the answer cannot end before --tokens",
    )
    parser.add_argument(
        "--gate-pct",
        type=float,
        default=15.0,
        help="the gap (constrained over free_graph) below which a graph is not worth building; "
        "write the value into the issue before the run",
    )
    parser.add_argument(
        "--free-eager",
        action="store_true",
        help="also time a free decode on a second engine without graphs (reference only; the "
        "second engine does not fit next to the first with flashinfer on a 4 GB GPU)",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    structured = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=_schema(args.min_length))
    with _engine(args, graphs=True) as llm:
        rows, answer = _engine_arms(args, llm, structured)
        isolated = _isolated(args, structured, answer, llm.engine._input.compiler)
    if args.free_eager:
        gc.collect()
        rows.extend(_free_eager_rows(args))
    write(
        {
            "benchmark": "e2e/constrained_decode",
            "model": args.model,
            "tokens": args.tokens,
            "dtype": args.dtype,
            "backends": args.backends,
            "min_length": args.min_length,
            "machine": machine(),
            "measured": {"steps": rows, "isolated": isolated},
            "estimates": _estimates(rows, isolated, args.gate_pct),
        },
        args.out,
    )


if __name__ == "__main__":
    main()
