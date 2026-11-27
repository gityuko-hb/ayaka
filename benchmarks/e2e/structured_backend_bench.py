"""What each structured output backend costs on the host: compiling, and every constrained step.

``llguidance`` and ``xgrammar`` compile the same constraints through
:class:`~ayaka.sampling.structured.compiler.GrammarCompiler` for a real tokenizer. The GPU side of
a constrained step (the bitmask applied inside the CUDA graph) is the same code for both, so what
differs between them is measured here, on the host, where the engine thread pays it:

    setup_ms          the first compile of a new compiler: the backend's view of the vocabulary
                      plus a small grammar (once per server)
    compile_ms        a grammar the compiler has not seen, after setup (once per new schema)
    cached_ms         the same grammar again: the cache lookup a repeated request pays
    step_us           one row's ``mask()`` and ``accept()`` for one token, along a real answer
    batch_ms[N]       ``build_controls`` for N constrained rows at the same point of their answers:
                      every mask of a decode step packed into the table the graph reads

Each constraint is measured in ``--rounds`` rounds that alternate the backends (A B A B ...), and
every number is the median over the rounds. Answers are fixed texts tokenized with the real
tokenizer, so both backends see the same tokens.

    uv run python benchmarks/e2e/structured_backend_bench.py --tokenizer Qwen/Qwen3-8B \\
        --out structured_backends.json

Needs the ``structured``, ``xgrammar`` and ``hf`` extras, and torch for ``batch_ms``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from ayaka.core.constraints import ConstraintKind, StructuredParams, ToolCallSyntax, ToolSchema
from ayaka.core.params import SamplingParams
from ayaka.core.types import RequestId
from ayaka.sampling.structured.compiler import GrammarCompiler

if TYPE_CHECKING:
    from ayaka.sampling.structured.backend import StructuredOutputBackendName

_BACKENDS: Final[tuple[StructuredOutputBackendName, ...]] = ("llguidance", "xgrammar")
_BATCHES: Final = (1, 8, 32)
_ORDER: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "customer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
                    "required": ["street", "city"],
                },
            },
            "required": ["name", "address"],
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
                "required": ["sku", "qty"],
            },
        },
    },
    "required": ["order_id", "customer", "items"],
}
_ORDER_ANSWER: Final = (
    '{"order_id": "ORD-2026-0412", "customer": {"name": "Nguyễn Văn An", "address": '
    '{"street": "12 Tràng Tiền", "city": "Hà Nội"}}, "items": [{"sku": "KB-104", "qty": 2}, '
    '{"sku": "MS-220", "qty": 1}]}'
)
_WEATHER: Final = ToolSchema(
    "get_weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city", "days"],
    },
)


def _cases(tokenizer: Any) -> dict[str, tuple[StructuredParams, list[int]]]:
    """Each constraint with the token ids of an answer it accepts."""

    def ids(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    open_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    close_id = tokenizer.convert_tokens_to_ids("</tool_call>")
    hermes = ToolCallSyntax(
        open="<tool_call>", close="</tool_call>", open_token=open_id, close_token=close_id
    )
    call = '\n{"name": "get_weather", "arguments": {"city": "Hanoi", "days": 3}}\n'
    return {
        "json_schema (nested order)": (
            StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=_ORDER),
            ids(_ORDER_ANSWER),
        ),
        "json_object": (
            StructuredParams(kind=ConstraintKind.JSON_OBJECT),
            ids('{"a": [1, 2, 3], "b": {"c": "text"}}'),
        ),
        "regex (date)": (
            StructuredParams(kind=ConstraintKind.REGEX, pattern=r"\d{4}-\d{2}-\d{2}"),
            ids("2026-11-27"),
        ),
        "tool call (hermes, forced, strict)": (
            StructuredParams(
                kind=ConstraintKind.TOOL_CALL,
                tools=(_WEATHER,),
                syntax=hermes,
                tool_choice="required",
                strict=True,
            ),
            [open_id, *ids(call), close_id],
        ),
    }


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _round(
    backend: StructuredOutputBackendName,
    tokenizer: Any,
    adapter: Any,
    params: StructuredParams,
    answer: list[int],
) -> dict[str, Any]:
    vocab = len(tokenizer)
    eos = (tokenizer.convert_tokens_to_ids("<|im_end|>"),)
    compiler = GrammarCompiler(adapter, vocab_size=vocab, eos_token_ids=eos, backend=backend)
    try:
        # Setup: the first compile pays for the vocabulary, with a throwaway grammar.
        warmup = StructuredParams(kind=ConstraintKind.REGEX, pattern="[0-9]{3}")
        t = time.perf_counter()
        compiler.compile(warmup)
        setup = _ms(t)
        t = time.perf_counter()
        constraint_id = compiler.compile(params)
        compile_ms = _ms(t)
        t = time.perf_counter()
        compiler.compile(params)
        cached = _ms(t)

        matcher = compiler.matcher(constraint_id, params)
        steps: list[float] = []
        for token in answer:
            t = time.perf_counter()
            matcher.mask()
            matcher.accept([token])
            steps.append((time.perf_counter() - t) * 1e6)
        batch = _batch_ms(compiler, constraint_id, params, answer, vocab, eos)
    finally:
        compiler.close()
    return {
        "setup_ms": setup,
        "compile_ms": compile_ms,
        "cached_ms": cached,
        "step_us": statistics.median(steps),
        "batch_ms": batch,
    }


def _batch_ms(
    compiler: GrammarCompiler,
    constraint_id: str,
    params: StructuredParams,
    answer: list[int],
    vocab: int,
    eos: tuple[int, ...],
) -> dict[str, float]:
    """``build_controls`` for N rows that are halfway through ``answer``."""
    from ayaka.sampling.logits import LogitRow, build_controls
    from ayaka.sampling.structured.runtime import ConstraintRuntime

    half = len(answer) // 2
    out: dict[str, float] = {}
    for size in _BATCHES:
        runtime = ConstraintRuntime()
        rows: list[LogitRow] = []
        for index in range(size):
            request_id = RequestId(f"row-{index}")
            runtime.admit(request_id, compiler.matcher(constraint_id, params))
            runtime.accept(request_id, answer[:half])
            rows.append(LogitRow(request_id, SamplingParams(temperature=0.0), half))
        times: list[float] = []
        for _ in range(5):
            t = time.perf_counter()
            build_controls(
                rows,
                vocab_size=vocab,
                eos_token_ids=eos,
                penalties=None,
                device="cpu",
                constraints=runtime,
            )
            times.append(_ms(t))
        out[str(size)] = statistics.median(times)
    return out


def _median(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        key: round(statistics.median(row[key] for row in rows), 4)
        for key in ("setup_ms", "compile_ms", "cached_ms", "step_us")
    }
    result["batch_ms"] = {
        size: round(statistics.median(row["batch_ms"][size] for row in rows), 4)
        for size in rows[0]["batch_ms"]
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from ayaka.tokenizers.adapters.hf import HfTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    adapter = HfTokenizer(tokenizer)
    result: dict[str, Any] = {"tokenizer": args.tokenizer, "rounds": args.rounds, "cases": {}}
    for name, (params, answer) in _cases(tokenizer).items():
        per_backend: dict[str, list[dict[str, Any]]] = {backend: [] for backend in _BACKENDS}
        for _ in range(args.rounds):
            for backend in _BACKENDS:
                per_backend[backend].append(_round(backend, tokenizer, adapter, params, answer))
        result["cases"][name] = {
            "answer_tokens": len(answer),
            **{backend: _median(rows) for backend, rows in per_backend.items()},
        }
        for backend in _BACKENDS:
            row = result["cases"][name][backend]
            print(
                f"{name:36s} {backend:10s} setup {row['setup_ms']:8.1f} ms  compile "
                f"{row['compile_ms']:7.2f} ms  cached {row['cached_ms']:.3f} ms  step "
                f"{row['step_us']:6.1f} us  batch {row['batch_ms']}",
                flush=True,
            )
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
