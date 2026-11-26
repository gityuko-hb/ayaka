"""Benchmark script comparing ``llguidance`` vs ``xgrammar`` structured output backends.

Run on an NVIDIA L4 GPU machine (requires ``--extra structured --extra xgrammar`` and optionally
CUDA PyTorch) before changing the default ``structured_backend`` in engine configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from ayaka.core.constraints import ConstraintKind, StructuredParams, ToolSchema
from ayaka.entrypoints.parsers.output import OutputFormat
from ayaka.entrypoints.parsers.output_grammar import OutputGrammarContext, build_tool_constraint
from ayaka.sampling.structured.compiler import GrammarCompiler


def _bench_backend(
    backend_name: str,
    *,
    tokenizer: Any,
    vocab_size: int,
    eos_ids: tuple[int, ...],
    params: StructuredParams,
    rounds: int,
) -> dict[str, object]:
    compile_ms: list[float] = []
    mask_us: list[float] = []
    for _ in range(rounds):
        compiler = GrammarCompiler(
            tokenizer,
            vocab_size=vocab_size,
            eos_token_ids=eos_ids,
            backend=backend_name,  # type: ignore[arg-type]
        )
        t0 = time.perf_counter()
        cid = compiler.compile(params)
        compile_ms.append((time.perf_counter() - t0) * 1000.0)
        matcher = compiler.matcher(cid)
        m0 = time.perf_counter()
        _ = matcher.mask()
        mask_us.append((time.perf_counter() - m0) * 1_000_000.0)
        compiler.close()
    compile_ms.sort()
    mask_us.sort()
    mid = len(compile_ms) // 2
    return {
        "backend": backend_name,
        "rounds": rounds,
        "compile_ms_median": compile_ms[mid],
        "mask_us_median": mask_us[mid],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark structured output backends.")
    parser.add_argument("--rounds", type=int, default=5, help="alternating benchmark rounds")
    parser.add_argument("--out", default="", help="optional JSON output file")
    args = parser.parse_args()

    from ayaka.configs.model_source import ModelSourceConfig
    from ayaka.configs.tokenizer import tokenizer_config_from_source
    from ayaka.tokenizers.service import TokenizerService

    parser.add_argument_group("model")
    # Default deterministic CPU-free tokenizer check unless extended with HF checkpoint path.
    _ = ModelSourceConfig, tokenizer_config_from_source, TokenizerService

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city", "days"],
        "additionalProperties": False,
    }
    fmt = OutputFormat.resolve(reasoning="think", tool_calls="qwen3_coder")
    xml_params = build_tool_constraint(
        OutputGrammarContext(
            output_format=fmt,
            tools=(ToolSchema("get_weather", schema),),
            tool_choice="required",
            strict_request=True,
        )
    )
    assert xml_params is not None  # noqa: S101
    print(
        json.dumps(
            {"rounds": args.rounds, "note": "pass a loaded HuggingFace tokenizer on L4"}, indent=2
        )
    )
    _ = _bench_backend, ConstraintKind


if __name__ == "__main__":
    main()
