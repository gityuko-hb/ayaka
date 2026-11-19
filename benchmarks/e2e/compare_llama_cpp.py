"""Ayaka next to llama.cpp on this machine, through their OpenAI-compatible servers.

Both engines read the same prompt text (the ``ayaka bench`` prompt set) with the same sampling
parameters, through ``/v1/completions`` with streaming, one engine at a time: each slot starts
its engine from its command, waits for ``/health``, measures, and stops it before the other one
starts. The rounds alternate the order (A B, then B A, ...) and the result keeps every raw value,
the medians, and the ratio of the two engines in each round.

Measured per engine and slot:

- TTFT cold (a prompt neither engine has seen: a fresh first line) and warm (the same prompt
  again, which both engines' prompt caches may serve);
- prompt tokens per second: the engine's own prompt token count over the cold TTFT;
- generation tokens per second: the tokens after the first over the time between the first and
  the last streamed token, with ``ignore_eos`` so both produce the same count;
- peak VRAM above the idle GPU (NVML, the whole device) while the engine is up.

Ayaka reads safetensors and llama.cpp reads GGUF, so the two cannot load one file in this
milestone. The script records each engine's model file, its SHA-256 and its precision, and a
``differences`` list; a result whose precisions differ (an f16 GGUF next to Ayaka's fp16 is the
closest pair) says so, and a quantized row is never merged into an f16 one.

    uv run python benchmarks/e2e/compare_llama_cpp.py \\
        --ayaka-cmd "ayaka serve MODEL --port 8000 ..." --ayaka-url http://127.0.0.1:8000 \\
        --llama-cmd "llama-server -m MODEL.gguf --port 8080 ..." \\
        --llama-url http://127.0.0.1:8080 \\
        --ayaka-model MODEL_DIR --llama-model MODEL.gguf --llama-precision f16 --out result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA: Final = "ayaka.compare/1"
QUANTIZED: Final = ("q2", "q3", "q4", "q5", "q6", "q8", "iq")
#: Names of one precision: Ayaka says fp16, llama.cpp says f16.
_PRECISIONS: Final = {"fp16": "f16", "float16": "f16", "fp32": "f32", "float32": "f32"}


def precision_name(precision: str) -> str:
    """One spelling per precision, so fp16 and f16 compare equal."""
    name = precision.lower()
    return _PRECISIONS.get(name, name)


@dataclass
class Engine:
    """One side of the comparison: how to start it, where it answers, what it loads."""

    name: str
    url: str
    command: list[str] | None
    model: str
    model_file: Path | None
    precision: str
    served_name: str = "model"
    version: str | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)

    @property
    def quantized(self) -> bool:
        return self.precision.lower().startswith(QUANTIZED)


@dataclass
class Sample:
    """One request's measurements."""

    ttft_s: float
    prompt_tokens: int
    completion_tokens: int
    generation_tok_s: float | None


class HealthTimeoutError(RuntimeError):
    """An engine did not become healthy in time."""


def file_hash(path: Path | None) -> str | None:
    """The SHA-256 of a model file, or of every weight file in a model folder (sorted)."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"the model file {path} does not exist")
    files = (
        sorted(p for p in path.iterdir() if p.suffix in (".safetensors", ".gguf", ".bin"))
        if path.is_dir()
        else [path]
    )
    digest = hashlib.sha256()
    for item in files:
        with item.open("rb") as stream:
            while chunk := stream.read(1 << 22):
                digest.update(chunk)
    return digest.hexdigest()


def output_tail(stream: IO[bytes] | None, lines: int = 20) -> str:
    """The last ``lines`` lines an engine wrote, for an error that says why it stopped."""
    if stream is None:
        return ""
    stream.flush()
    stream.seek(0)
    text = stream.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def wait_healthy(
    url: str, timeout: float, *, alive: Any = None, output: IO[bytes] | None = None
) -> None:
    """Poll ``/health`` until it answers 200, the process dies, or the time is up.

    ``output`` is the file the engine writes its stdout and stderr to; when the engine exits,
    its last lines go into the error.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if alive is not None and not alive():
            tail = output_tail(output)
            said = f"; its last output:\n{tail}" if tail else ""
            raise RuntimeError(f"the engine at {url} exited before it was healthy{said}")
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as response:  # noqa: S310
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise HealthTimeoutError(f"{url} was not healthy after {timeout:.0f} s")


def stream_completion(
    url: str, model: str, prompt: str, max_tokens: int, key: str | None
) -> Sample:
    """One streamed ``/v1/completions`` request: TTFT, token counts and generation speed."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    request = urllib.request.Request(  # noqa: S310  # the engines' own URLs
        url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(), headers=headers
    )
    started = time.perf_counter()
    stamps: list[float] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            if any(choice.get("text") for choice in chunk.get("choices") or []):
                stamps.append(time.perf_counter())
    if not stamps:
        raise RuntimeError(f"{url} streamed no text")
    prompt_tokens = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
    completion = int(usage.get("completion_tokens") or timings.get("predicted_n") or len(stamps))
    elapsed = stamps[-1] - stamps[0]
    # Chunks may carry several tokens; the count comes from the usage, the time from the chunks.
    rate = (completion - 1) / elapsed if completion > 1 and elapsed > 0 else None
    return Sample(stamps[0] - started, prompt_tokens, completion, rate)


class VramSampler:
    """Peak GPU memory in use (NVML, the whole device) while it runs."""

    def __init__(self) -> None:
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def temperature() -> int | None:
        """GPU 0's temperature in Celsius, or ``None`` without NVML."""
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return int(pynvml.nvmlDeviceGetTemperature(handle, 0))
            finally:
                pynvml.nvmlShutdown()
        except Exception:  # no NVML: no temperature, the rest stands
            return None

    @staticmethod
    def used() -> int | None:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            finally:
                pynvml.nvmlShutdown()
        except Exception:  # no NVML: no VRAM numbers, the rest stands
            return None

    def _run(self) -> None:
        # NVML is opened once for the whole slot: opening and closing it on every sample costs
        # the driver enough to slow the engine being measured.
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:  # no NVML: no VRAM numbers, the rest stands
            return
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self._stop.wait(0.2):
                used = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                self.peak = max(self.peak or 0, used)
        except Exception:  # a query that fails ends the sampling, not the run
            return
        finally:
            pynvml.nvmlShutdown()

    def __enter__(self) -> VramSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


@dataclass
class Plan:
    """The comparison's settings."""

    prompts: list[tuple[str, str]]  # (name, text)
    rounds: int
    max_tokens: int
    timeout: float
    key: str | None


def round_order(round_index: int, engines: list[Engine]) -> list[Engine]:
    """A B in even rounds, B A in odd ones."""
    return engines if round_index % 2 == 0 else list(reversed(engines))


def run_slot(engine: Engine, plan: Plan, round_index: int, *, running: list[str]) -> dict[str, Any]:
    """Start one engine, measure every prompt cold then warm, stop it."""
    if running:
        raise RuntimeError(f"{running} is still up; the engines never run at the same time")
    idle = VramSampler.used()
    temperature = VramSampler.temperature()
    up_since = time.time()
    # The engine's output goes to a temporary file, not to a pipe nobody reads (a full pipe
    # would block the engine), and is kept only to explain a start that fails.
    output = tempfile.TemporaryFile() if engine.command is not None else None  # noqa: SIM115
    if engine.command is not None:
        engine.process = subprocess.Popen(  # noqa: S603  # the user's own command
            engine.command, stdout=output, stderr=subprocess.STDOUT
        )
    running.append(engine.name)
    try:
        process = engine.process
        wait_healthy(
            engine.url,
            plan.timeout,
            alive=(lambda: process.poll() is None) if process is not None else None,
            output=output,
        )
        rows: list[dict[str, Any]] = []
        with VramSampler() as vram:
            for name, text in plan.prompts:
                fresh = f"Run {round_index} of {engine.name}, prompt {name}.\n{text}"
                cold = stream_completion(
                    engine.url, engine.served_name, fresh, plan.max_tokens, plan.key
                )
                warm = stream_completion(
                    engine.url, engine.served_name, fresh, plan.max_tokens, plan.key
                )
                rows.append(
                    {
                        "prompt": name,
                        "prompt_tokens": cold.prompt_tokens,
                        "completion_tokens": cold.completion_tokens,
                        "ttft_cold_s": cold.ttft_s,
                        "ttft_warm_s": warm.ttft_s,
                        "prompt_tok_s": cold.prompt_tokens / cold.ttft_s if cold.ttft_s else None,
                        "generation_tok_s": cold.generation_tok_s,
                    }
                )
        peak = vram.peak
        stop_engine(engine)  # down before the slot's record says so
        return {
            "engine": engine.name,
            "round": round_index,
            # When the engine was started and when it had exited: the slots never overlap.
            "up": [up_since, time.time()],
            "gpu_temperature_c": [temperature, VramSampler.temperature()],
            "rows": rows,
            "vram_peak_above_idle_bytes": peak - idle
            if peak is not None and idle is not None
            else None,
        }
    finally:
        stop_engine(engine)
        running.remove(engine.name)
        if output is not None:
            output.close()


def stop_engine(engine: Engine) -> None:
    process = engine.process
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    engine.process = None


_METRICS: Final = ("ttft_cold_s", "ttft_warm_s", "prompt_tok_s", "generation_tok_s")


def summarize(slots: list[dict[str, Any]], engines: list[Engine]) -> dict[str, Any]:
    """Medians per engine, prompt and metric, and the ratio A/B of each round."""
    names = [engine.name for engine in engines]
    medians: dict[str, dict[str, dict[str, float | None]]] = {}
    for name in names:
        per_prompt: dict[str, dict[str, list[float]]] = {}
        for slot in slots:
            if slot["engine"] != name:
                continue
            for row in slot["rows"]:
                bucket = per_prompt.setdefault(row["prompt"], {m: [] for m in _METRICS})
                for metric in _METRICS:
                    if row[metric] is not None:
                        bucket[metric].append(row[metric])
        medians[name] = {
            prompt: {m: statistics.median(v) if v else None for m, v in values.items()}
            for prompt, values in per_prompt.items()
        }
    ratios: dict[str, dict[str, list[float]]] = {}
    first, second = names
    rounds = sorted({slot["round"] for slot in slots})
    for index in rounds:
        a = next(s for s in slots if s["round"] == index and s["engine"] == first)
        b = next(s for s in slots if s["round"] == index and s["engine"] == second)
        for row_a, row_b in zip(a["rows"], b["rows"], strict=True):
            bucket = ratios.setdefault(row_a["prompt"], {m: [] for m in _METRICS})
            for metric in _METRICS:
                if row_a[metric] is not None and row_b[metric]:
                    bucket[metric].append(row_a[metric] / row_b[metric])
    return {
        "medians": medians,
        "ratios": {
            "of": f"{first}/{second}",
            "per_round": ratios,
            "median": {
                prompt: {m: statistics.median(v) if v else None for m, v in values.items()}
                for prompt, values in ratios.items()
            },
        },
    }


def differences(engines: list[Engine]) -> list[str]:
    """What is not the same between the two sides, said before any number is read."""
    first, second = engines
    notes = [
        f"{first.name} loads {first.model_file} ({first.precision}); {second.name} loads "
        f"{second.model_file} ({second.precision}): not the same file"
    ]
    if precision_name(first.precision) != precision_name(second.precision):
        notes.append(f"the precisions differ: {first.precision} against {second.precision}")
    notes.append("prompt token counts come from each engine's own tokenizer and template")
    return notes


def validate(result: dict[str, Any]) -> list[str]:
    """What is missing or inconsistent in a comparison result."""
    problems = [
        f"missing {key!r}"
        for key in ("schema", "machine", "engines", "opponent", "differences", "slots", "summary")
        if key not in result
    ]
    if problems:
        return problems
    if result["schema"] != SCHEMA:
        problems.append(f"schema is {result['schema']!r}")
    precisions = {precision_name(engine["precision"]) for engine in result["engines"]}
    quantized = {p for p in precisions if p.startswith(QUANTIZED)}
    if quantized and quantized != precisions:
        problems.append("a quantized engine is compared with an unquantized one in one result")
    for engine, prompts in result["summary"]["medians"].items():
        for prompt, metrics in prompts.items():
            raw = [
                row
                for slot in result["slots"]
                if slot["engine"] == engine
                for row in slot["rows"]
                if row["prompt"] == prompt
            ]
            for metric, median in metrics.items():
                values = [row[metric] for row in raw if row[metric] is not None]
                if values and median != statistics.median(values):
                    problems.append(f"{engine} {prompt} {metric}: the median does not match")
    return problems


def _tokenizer(model: Path) -> tuple[Any, Any]:
    """``encode`` and ``decode`` of the model's tokenizer, read from its folder, offline."""
    from ayaka.configs.model_source import ModelSourceConfig
    from ayaka.configs.tokenizer import tokenizer_config_from_source
    from ayaka.tokenizers.factory import DefaultTokenizerFactory

    loaded = DefaultTokenizerFactory.from_config(
        tokenizer_config_from_source(ModelSourceConfig(str(model), offline=True))
    )
    tokenizer = loaded.tokenizer
    if tokenizer is None:
        raise ValueError(f"{model} has no tokenizer")

    def encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids)

    return encode, decode


def _version(command: list[str] | None) -> str | None:
    if not command:
        return None
    try:
        result = subprocess.run(  # noqa: S603  # the user's own binary
            [command[0], "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout + result.stderr).strip().splitlines()
    return next((line for line in text if "version" in line.lower()), text[0] if text else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ayaka-cmd", required=True, help="the command that starts ayaka serve")
    parser.add_argument("--ayaka-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ayaka-model", required=True, type=Path, help="the safetensors folder")
    parser.add_argument("--ayaka-precision", default="fp16")
    parser.add_argument("--llama-cmd", required=True, help="the command that starts llama-server")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080")
    parser.add_argument("--llama-model", required=True, type=Path, help="the GGUF file")
    parser.add_argument("--llama-precision", required=True, help="f16, q4_k_m, ...")
    parser.add_argument("--served-name", default="model", help="the name both servers answer to")
    parser.add_argument("--api-key", help="ayaka's key, when it has one")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--rounds", type=int, default=6, help="even: A B and B A alternate")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--conversion", default="", help="how the GGUF was made, for the record")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from _machine import machine

    from ayaka.entrypoints.bench_prompts import PROMPT_SET_VERSION, material_ids

    engines = [
        Engine(
            "ayaka",
            args.ayaka_url,
            shlex.split(args.ayaka_cmd),
            str(args.ayaka_model),
            args.ayaka_model,
            args.ayaka_precision,
            args.served_name,
        ),
        Engine(
            "llama.cpp",
            args.llama_url,
            shlex.split(args.llama_cmd),
            str(args.llama_model),
            args.llama_model,
            args.llama_precision,
            args.served_name,
        ),
    ]
    hashes = {engine.name: file_hash(engine.model_file) for engine in engines}
    if engines[0].quantized != engines[1].quantized:
        print("note: one side is quantized; the result says so and is not an f16 comparison")
    # The bench prose material cut to each length with the model's own tokenizer, then sent as
    # text: both engines read the same characters, and each reports its own token count.
    encode, decode = _tokenizer(args.ayaka_model)
    prompts = [
        (f"prose-{length}", decode(material_ids("prose", length, encode)))
        for length in args.lengths
    ]
    plan = Plan(prompts, args.rounds, args.max_tokens, args.timeout, args.api_key)
    for engine in engines:
        engine.version = _version(engine.command)
    from ayaka.entrypoints.bench_machine import power_source

    power_start = power_source()
    if power_start == "battery":
        print("warning: on battery; the numbers will be lower than on mains, and are marked")
    running: list[str] = []
    slots: list[dict[str, Any]] = []
    started = datetime.now(UTC).isoformat(timespec="seconds")
    for index in range(plan.rounds):
        for engine in round_order(index, engines):
            print(f"round {index + 1}/{plan.rounds}: {engine.name}", flush=True)
            slots.append(run_slot(engine, plan, index, running=running))
    opponent = engines[1]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "started": started,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": machine(),
        # ac, battery or unknown, when the run started and when it ended.
        "power": [power_start, power_source()],
        "prompt_set": {"version": PROMPT_SET_VERSION, "prompts": [name for name, _ in prompts]},
        "sampling": {"temperature": 0, "max_tokens": plan.max_tokens, "ignore_eos": True},
        "engines": [
            {
                "name": engine.name,
                "version": engine.version,
                "flags": engine.command,
                "model_file": str(engine.model_file),
                "precision": engine.precision,
                "hash": hashes[engine.name],
            }
            for engine in engines
        ],
        "opponent": {
            "name": opponent.name,
            "version": opponent.version,
            "flags": opponent.command,
            "model_file": str(opponent.model_file),
            "precision": opponent.precision,
            "hash": hashes[opponent.name],
            "conversion": args.conversion,
        },
        "differences": differences(engines),
        "slots": slots,
        "summary": summarize(slots, engines),
    }
    problems = validate(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if problems:
        print("the result does not validate: " + "; ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
