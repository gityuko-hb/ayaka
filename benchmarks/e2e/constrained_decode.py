"""What a constrained decode step costs next to a free one, with the mask inside the CUDA graph.

A request with an output constraint (a JSON schema, a strict tool, a forced tool call) replays a
graph like any other: the host computes the mask of the token about to be drawn before the replay,
and the captured sampling path applies it. This script measures how much that step costs over a
free one, and what the graphs cost in memory.

Measured, per batch size, on a real engine (greedy, ``ignore_eos``, one prompt per request):

    free_graph     no constraint, CUDA graphs on     the step a graph replays without a mask
    eager_control  a zero logit bias, graphs on      the same step run eagerly, without a grammar:
                                                     what a constrained step cost before its mask
                                                     went into the graph (same engine, same ops)
    constrained    JSON schema, CUDA graphs on       the step with a mask active
    free_eager     no constraint, CUDA graphs off    only with ``--free-eager``: another engine, ops
                                                     loaded without the graph, for reference only

Each is the decode time per step: a run of ``--tokens`` tokens minus a run of one token (the
prefill), divided by ``--tokens - 1``, repeated ``--repeat`` times after a warm-up. Each row also
records the graph replays it caused: ``free_graph`` and ``constrained`` must have some and
``eager_control`` none, or the script stops, because the arm would not measure what it is named for.

Measured in isolation, per batch size, to split the constrained step into its parts:

    host_mask_us_per_row   the grammar matcher's ``mask()`` for one row (llguidance), per token
    host_controls_ms       ``build_controls`` for the whole batch: on the CPU, and with the copy
                           to the GPU
    gpu_mask_ms            the float32 copy of the logits and ``apply_token_bitmask`` on the GPU,
                           launched eagerly: it includes the gaps between the launches, so at a
                           small batch it is launch-bound and overstates what a graph pays
    gpu_mask_graph_ms      the same two calls captured in a CUDA graph and replayed: what the
                           masked graph adds to a free one

``graphs`` records what the captured graphs took (the memory plan's ``graph_bytes``). ``summary``
only subtracts the measured numbers; it estimates nothing.

    uv run python benchmarks/e2e/constrained_decode.py --model /path/to/SmolLM2-135M \
        --backends flashinfer torch

Needs the ``structured`` extra (llguidance) and a GPU. Run it on the dev GPU; a number from another
machine says nothing about this one. ``--out`` writes the JSON; put the numbers that matter in the
PR and in ``docs/EXPERIMENTS.md`` instead of committing the file.
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
        cache=CacheConfig(
            block_size=16,
            kv_reserve_tokens=top * (args.tokens + 64),
            **({} if args.memory_ratio is None else {"memory_ratio": args.memory_ratio}),
        ),
        scheduler=SchedulerConfig(max_running=top, max_tokens_per_step=512),
        structured_backend=args.structured_backend,
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
        for arm, constraint, bias, replays_expected in (
            ("free_graph", None, False, True),
            ("eager_control", None, True, False),
            ("constrained", structured, False, True),
        ):
            _progress(f"{arm} batch {batch}")
            before = _replays(llm)
            runs, last = _per_step_ms(llm, batch, args.tokens, args.repeat, constraint, bias=bias)
            replays = _replays(llm) - before
            # The arm is what it says only if the graph did, or did not, run the steps.
            if replays_expected != (replays > 0):
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
    matcher = compiler.matcher(constraint_id, structured)
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
            runtime.admit(request_id, compiler.matcher(constraint_id, structured))
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

        def mask(logits: Any = logits, bitmask: Any = bitmask) -> None:
            apply_token_bitmask(logits.float(), bitmask)

        both_ms = _events(mask, args.repeat * 10)
        in_graph_ms = _events(_captured(mask), args.repeat * 10)
        out["per_batch"][batch] = {
            "host_controls_cpu_ms": round(cpu_ms, 3),
            "host_controls_cuda_ms": round(cuda_ms, 3),
            "gpu_float_copy_ms": round(float_ms, 4),
            "gpu_mask_ms": round(both_ms, 4),
            "gpu_mask_graph_ms": round(in_graph_ms, 4),
        }
    return out


def _captured(call: Any) -> Any:
    """``call`` captured in a CUDA graph, as the function that replays it (no launch gaps)."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        call()
    return graph.replay


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


def _summary(rows: Sequence[dict[str, Any]], isolated: dict[str, Any]) -> dict[str, Any]:
    """Subtract the measured step times from each other, per batch size; nothing is estimated."""
    median = {(row["arm"], row["batch"]): row["median_ms"] for row in rows}
    out: dict[str, Any] = {}
    accept_ms = isolated["host_accept_us_per_row"]["p50"] / 1e3
    for batch, parts in isolated["per_batch"].items():
        free = median[("free_graph", batch)]
        control = median[("eager_control", batch)]
        constrained = median[("constrained", batch)]
        # What the parts measured alone say of the constrained step's cost over a free one, and
        # what they leave unexplained: a large remainder is a cost this script does not see.
        # The masked graph also copies the logits to float32, which the free graph does not.
        explained = parts["host_controls_cuda_ms"] + accept_ms * batch + parts["gpu_mask_graph_ms"]
        out[batch] = {
            "constrained_over_free_graph_pct": round((constrained / free - 1) * 100, 1),
            "constrained_over_free_graph_ms": round(constrained - free, 3),
            "eager_control_over_free_graph_ms": round(control - free, 3),
            "gpu_mask_over_free_graph_pct": round(parts["gpu_mask_ms"] / free * 100, 1),
            "gpu_mask_in_graph_over_free_graph_pct": round(
                parts["gpu_mask_graph_ms"] / free * 100, 1
            ),
            "explained_by_parts_ms": round(explained, 3),
            "unexplained_ms": round(constrained - free - explained, 3),
        }
    return out


def _check_memory(llm: LLM) -> dict[str, Any]:
    """The memory ledger against the allocator after start-up; an over-committed GPU is not timed.

    Under WSL2 the driver serves memory beyond the device's from the host, so every step time of
    an over-committed engine would measure paging, not the step (exit 2). Lower ``--memory-ratio``
    to leave room for the graphs.
    """
    startup = llm.engine.startup
    if startup is None:
        raise RuntimeError("the engine has no start-up report: it did not start on a GPU")
    found = startup.reconciliation
    out = {
        "matches": found.matches,
        "ledger_bytes": found.ledger_bytes,
        "measured_bytes": found.measured_bytes,
        "difference_bytes": found.difference_bytes,
        "tolerance_bytes": found.tolerance_bytes,
    }
    if not found.matches:
        _progress(
            f"invalid run: the allocator holds {found.difference_bytes / 2**20:,.0f} MiB more than "
            f"the plan (tolerance {found.tolerance_bytes / 2**20:,.0f} MiB); the GPU is "
            "over-committed. Lower --memory-ratio."
        )
        raise SystemExit(2)
    return out


def _graph_memory(llm: LLM) -> dict[str, Any]:
    """What the engine's start-up captured: bytes the allocator kept, per bucket and in all."""
    startup = llm.engine.startup
    captured = startup.graphs if startup is not None else ()
    return {
        "buckets": [
            {"bucket": item.bucket, "bytes": item.bytes, "seconds": round(item.seconds, 2)}
            for item in captured
        ],
        "total_bytes": sum(item.bytes for item in captured),
    }


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
        "--memory-ratio",
        type=float,
        default=None,
        help="cache.memory_ratio (the engine's default when omitted); the graphs of many buckets "
        "need room beside the weights and the KV cache, or the run is refused as over-committed",
    )
    parser.add_argument(
        "--free-eager",
        action="store_true",
        help="also time a free decode on a second engine without graphs (reference only; the "
        "second engine does not fit next to the first with flashinfer on a 4 GB GPU)",
    )
    parser.add_argument(
        "--structured-backend",
        choices=("llguidance", "xgrammar"),
        default="llguidance",
        help="the grammar backend the engine compiles the constraint with",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    structured = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=_schema(args.min_length))
    with _engine(args, graphs=True) as llm:
        memory = _check_memory(llm)
        graphs = _graph_memory(llm)
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
            "memory_ratio": args.memory_ratio,
            "memory": memory,
            "graphs": graphs,
            "measured": {"steps": rows, "isolated": isolated},
            "summary": _summary(rows, isolated),
        },
        args.out,
    )


if __name__ == "__main__":
    main()
