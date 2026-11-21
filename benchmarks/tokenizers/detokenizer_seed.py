"""Benchmark detokenizer creation against prompt length, and token delivery during a long encode.

Two measurements on a real tokenizer, for the byte path and the ``DecodeStream`` path:

* **creation:** the median time of ``TokenizerService.new_detokenizer`` for prompts of 4,096,
  16,384 and 50,000 tokens (the recorded Claude Code session, repeated to length);
* **contention:** while another thread encodes the 50,000-token prompt in a loop, one-token
  ``update()`` calls are made every millisecond for a few seconds and their latencies recorded.

One run measures whatever ``ayaka`` is importable, so a before/after pair runs the same script
with ``PYTHONPATH`` pointing at each checkout's ``python/``::

    PYTHONPATH=<checkout>/python uv run python benchmarks/tokenizers/detokenizer_seed.py \
        --tokenizer unsloth/Llama-3.2-1B-Instruct --label B --revision <sha> --out out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _machine import machine, write

import ayaka
from ayaka.configs.tokenizer import DetokenizeParams, TokenizerConfig
from ayaka.tokenizers.contracts import EncodeOptions, TextInput
from ayaka.tokenizers.factory import DefaultTokenizerFactory
from ayaka.tokenizers.service import TokenizerService

SCHEMA: Final = "ayaka.detokenizer-seed/1"
LENGTHS: Final = (4096, 16384, 50000)
_FIXTURE: Final = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "entrypoints"
    / "fixtures"
    / "claude-code-2.1.327"
    / "004-anthropic.json"
)


def _prompt_ids(service: TokenizerService, length: int) -> tuple[tuple[int, ...], str]:
    """At least ``length`` ids of the recorded session repeated, and the text they came from."""
    tokenizer = service.tokenizer
    if tokenizer is None:
        raise SystemExit("the tokenizer cannot encode")
    piece = json.dumps(json.loads(_FIXTURE.read_text(encoding="utf-8"))["body"]["messages"])
    text = piece
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < length:
        text += "\n" + piece
        ids = tokenizer.encode(text, add_special_tokens=False)
    return tuple(ids), text


def _service(name: str, loaded: Any, backend: str) -> TokenizerService:
    tokenizer = loaded.tokenizer
    return TokenizerService(
        TokenizerConfig(name, mode="hf", detokenize_backend=backend),
        max_model_len=200_000,
        model_vocab_size=max(tokenizer.valid_token_ids) + 1,
        loaded=loaded,
    )


def _creation(service: TokenizerService, ids: tuple[int, ...], reps: int) -> dict[str, Any]:
    params = DetokenizeParams(skip_special_tokens=True)
    result: dict[str, Any] = {}
    for length in LENGTHS:
        prompt = ids[:length]
        service.new_detokenizer(params, prompt_token_ids=prompt)  # warm-up
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            service.new_detokenizer(params, prompt_token_ids=prompt)
            times.append((time.perf_counter() - start) * 1000)
        result[str(length)] = {"median_ms": statistics.median(times), "runs_ms": times}
    return result


def _contention(
    service: TokenizerService, ids: tuple[int, ...], text: str, seconds: float
) -> dict[str, Any]:
    """Latencies of one-token updates while another thread encodes the long prompt."""
    params = DetokenizeParams(skip_special_tokens=True)
    detok = service.new_detokenizer(params, prompt_token_ids=ids[:64])
    stop = threading.Event()
    encodes = [0]
    long_text = TextInput(text, add_special_tokens=False)

    def encoder() -> None:
        while not stop.is_set():
            service.encode(long_text, EncodeOptions(max_output_tokens=1))
            encodes[0] += 1

    thread = threading.Thread(target=encoder, daemon=True)
    thread.start()
    time.sleep(0.2)  # let the first encode start
    latencies: list[float] = []
    tokens = ids[64:]
    deadline = time.perf_counter() + seconds
    index = 0
    while time.perf_counter() < deadline:
        start = time.perf_counter()
        detok.update([tokens[index % len(tokens)]])
        latencies.append((time.perf_counter() - start) * 1000)
        index += 1
        if index % 2000 == 0:  # keep the detokenizer's own text short
            detok = service.new_detokenizer(params, prompt_token_ids=ids[:64])
        time.sleep(0.001)
    stop.set()
    thread.join()
    ordered = sorted(latencies)
    return {
        "updates": len(latencies),
        "encodes": encodes[0],
        "p50_ms": ordered[len(ordered) // 2],
        "p99_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "max_ms": ordered[-1],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--label", default="", help="which side of a pair this run is (A or B)")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--contention-seconds", type=float, default=3.0)
    parser.add_argument("--revision", default="", help="the commit of the measured python/")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    loaded = DefaultTokenizerFactory.from_config(TokenizerConfig(args.tokenizer, mode="hf"))
    services = {
        backend: _service(args.tokenizer, loaded, backend) for backend in ("bytes", "stream")
    }
    ids, text = _prompt_ids(services["bytes"], max(LENGTHS))
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "label": args.label,
        "revision": args.revision,
        "ayaka_path": str(Path(ayaka.__file__).resolve().parent),
        "machine": machine(),
        "tokenizer": args.tokenizer,
        "prompt_tokens": len(ids),
        "creation": {},
        "contention": {},
    }
    for backend, service in services.items():
        result["creation"][backend] = _creation(service, ids, args.reps)
        result["contention"][backend] = _contention(service, ids, text, args.contention_seconds)
        service.close()
    write(result, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
