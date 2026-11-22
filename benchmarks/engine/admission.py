"""Time the engine thread spends admitting a prompt the server already encoded.

The server encodes a request's prompt through the engine's tokenizer service, then hands the ids
to the engine. What the engine thread does with them before its next step is the time this script
measures, for prompts of 4,096, 16,384 and 50,000 tokens (the recorded Claude Code session,
repeated to length):

* before ``EngineCore.prepare`` existed, the engine thread ran ``EngineCore.add_request``: the
  whole input path, which checks every id again;
* with it, a worker thread runs ``prepare`` and the engine thread only ``admit``; the ids come
  from an encode with ``vouch=True``, as the server's do. ``prepare`` is timed too, for the record.

Each admitted request is aborted at once and flushed with an idle step, so nothing runs on the
GPU between two measurements. One run measures whatever ``ayaka`` is importable, so a before and
after pair runs the same script with ``PYTHONPATH`` pointing at each checkout's ``python/``::

    PYTHONPATH=<checkout>/python uv run python benchmarks/engine/admission.py \
        --model unsloth/Llama-3.2-1B-Instruct --label B --revision <sha> --out out.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _machine import machine, write

import ayaka
from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.engine.build import build_engine
from ayaka.tokenizers.contracts import EncodeOptions, TokenIdsInput
from ayaka.utils.dtypes import DType

if TYPE_CHECKING:
    from ayaka.engine.core import EngineCore

SCHEMA: Final = "ayaka.engine-admission/1"
LENGTHS: Final = (4096, 16384, 50000)
_FIXTURE: Final = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "entrypoints"
    / "fixtures"
    / "claude-code-2.1.327"
    / "004-anthropic.json"
)


def _prompt_ids(engine: EngineCore, length: int) -> tuple[int, ...]:
    """At least ``length`` ids of the recorded session, repeated."""
    tokenizer = engine.tokenizer.tokenizer
    if tokenizer is None:
        raise SystemExit("the tokenizer cannot encode")
    piece = json.dumps(json.loads(_FIXTURE.read_text(encoding="utf-8"))["body"]["messages"])
    text = piece
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < length:
        text += "\n" + piece
        ids = tokenizer.encode(text, add_special_tokens=False)
    return tuple(ids)


def _flush(engine: EngineCore) -> None:
    while engine.has_unfinished:
        engine.step()


def _admit_once(engine: EngineCore, name: str, ids: tuple[int, ...]) -> tuple[float, float | None]:
    """Seconds on the engine thread, and seconds in ``prepare`` (``None`` before it existed)."""
    service = engine.tokenizer
    options = EncodeOptions(max_output_tokens=1)
    split = hasattr(engine, "prepare")
    if split and "vouch" in inspect.signature(service.encode).parameters:
        prompt = service.encode(TokenIdsInput(ids), options, vouch=True).token_ids
    else:
        prompt = service.encode(TokenIdsInput(ids), options).token_ids
    request = GenerateRequest(
        request_id=RequestId(name), prompt=prompt, sampling=SamplingParams(max_tokens=16)
    )
    prepare_seconds: float | None = None
    if split:
        started = time.perf_counter()
        prepared = engine.prepare(request)  # type: ignore[attr-defined]  # absent before the split
        prepare_seconds = time.perf_counter() - started
        started = time.perf_counter()
        engine.admit(prepared, stream=True)  # type: ignore[attr-defined]
        engine_seconds = time.perf_counter() - started
    else:
        started = time.perf_counter()
        engine.add_request(request, stream=True)
        engine_seconds = time.perf_counter() - started
    engine.abort(RequestId(name))
    _flush(engine)
    return engine_seconds, prepare_seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Engine-thread admission time against length.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--label", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    config = EngineConfig(
        source=ModelSourceConfig(args.model),
        device=DeviceConfig(device="cuda", dtype=DType.from_str("fp16"), backends=("torch",)),
        cache=CacheConfig(block_size=16, memory_ratio=0.8),
        scheduler=SchedulerConfig(max_running=4),
    )
    engine = build_engine(config)
    try:
        ids = _prompt_ids(engine, max(LENGTHS))
        results: dict[str, Any] = {}
        for length in LENGTHS:
            prompt = ids[:length]
            _admit_once(engine, f"warm-{length}", prompt)
            engine_ms: list[float] = []
            prepare_ms: list[float] = []
            for rep in range(args.reps):
                on_engine, in_prepare = _admit_once(engine, f"r-{length}-{rep}", prompt)
                engine_ms.append(on_engine * 1e3)
                if in_prepare is not None:
                    prepare_ms.append(in_prepare * 1e3)
            results[str(length)] = {
                "engine_thread_ms": engine_ms,
                "engine_thread_median_ms": statistics.median(engine_ms),
                "prepare_ms": prepare_ms or None,
                "prepare_median_ms": statistics.median(prepare_ms) if prepare_ms else None,
            }
    finally:
        engine.shutdown()
    write(
        {
            "schema": SCHEMA,
            "label": args.label,
            "revision": args.revision,
            "ayaka_path": str(Path(ayaka.__file__).resolve().parent),
            "model": args.model,
            "reps": args.reps,
            "machine": machine(),
            "results": results,
        },
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
