"""``ayaka bench``: one run that measures a model on this machine and says whether to trust it.

The run is in-process (no HTTP), on the engine ``ayaka serve`` would build from the same flags:

* **prompt read**: tokens per second of each prompt group at each length, cold (the prefix cache
  is not used), three repetitions, every raw value kept;
* **generation**: tokens per second while producing 256 tokens at several context lengths, the
  first token (which belongs to the prompt read) left out;
* **first token**: TTFT of a long system prompt with a short user turn, cold and then warm (the
  same system prompt with another user turn);
* **agent**: a short multi-turn session with a tool call and an edited history tail, as the
  share of each turn's prompt the cache gave and the total time;
* **sanity**: a forced tool call and a JSON answer, each checked against its schema. A failed item
  marks the whole result invalid (``valid: false``, exit 2); the speed numbers stay, marked;
* **memory**: peak VRAM and peak RAM, sampled during the run.

``results.json`` (:data:`SCHEMA`) keeps the raw runs, the medians, the engine configuration, the
backends, the memory plan, the machine, the versions, the power source and the GPU temperature at
the start and the end. No aggregate "speed" is computed: the parts are reported separately.

A laptop on battery is warned about and marked, not refused. Comparing two configurations (A B B
A pairs) is the comparison script's job; one ``bench`` run measures one configuration.
"""

from __future__ import annotations

import gc
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from ayaka.core.errors import AyakaError
from ayaka.entrypoints.bench_machine import MemorySampler, gpu_state, machine_info, power_source
from ayaka.entrypoints.bench_prompts import (
    GROUPS,
    PROMPT_SET_VERSION,
    build_prompt,
    material_ids,
    prompt_digest,
)
from ayaka.entrypoints.console import EXIT_BUG, EXIT_OK, EXIT_USAGE, say

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

    from ayaka.configs.engine import EngineConfig
    from ayaka.core.outputs import RequestOutput
    from ayaka.engine.llm import LLM
    from ayaka.engine.recorder import SessionRecorder

__all__ = [
    "GENERATION_TOKENS",
    "LENGTHS",
    "QUICK_LENGTHS",
    "SCHEMA",
    "BenchOptions",
    "SanityItem",
    "bench_command",
    "median_of",
    "model_label",
    "run_bench",
]

_LOG = logging.getLogger(__name__)

#: The format of ``results.json``; a reader refuses another.
SCHEMA: Final = "ayaka.bench/1"

#: The prompt lengths of a full run; the longest one the machine holds is added to them.
LENGTHS: Final = (1024, 4096, 16384)
#: The prompt lengths of ``--quick``.
QUICK_LENGTHS: Final = (1024, 4096)
#: Tokens produced in a generation run.
GENERATION_TOKENS: Final = 256
#: Repetitions of each measurement (a warm-up run comes first and is discarded).
REPEATS: Final = 3
#: The system prompt of the first-token runs, in tokens (about; it is text).
SYSTEM_TOKENS: Final = 2048


@dataclass(frozen=True, slots=True)
class BenchOptions:
    """What one run does.

    Attributes:
        model: The model as the user named it.
        quick: Two lengths and one repetition fewer, for a check that the command works.
        seed: The sampling seed (the runs are greedy; the seed only names the run).
        record: A directory for the session recorder, or ``None``.
    """

    model: str
    quick: bool = False
    seed: int = 0
    record: Path | None = None

    @property
    def repeats(self) -> int:
        """Measured runs per point."""
        return 2 if self.quick else REPEATS


@dataclass(frozen=True, slots=True)
class SanityItem:
    """One sanity check: passed or not, and what was seen."""

    name: str
    passed: bool
    detail: str


def model_label(model: str) -> str:
    """The model as a result names it: a Hub cache path becomes its repo id, so no user's home
    folder ends up in a shared result; any other name is kept."""
    for part in reversed(Path(model).parts):
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return model


def median_of(values: Sequence[float]) -> float | None:
    """The median, ``None`` for no value."""
    return statistics.median(values) if values else None


# -- the measurements --------------------------------------------------------------------------


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        # A plain value as itself; a structured one (DType's tuple) by its label or name.
        plain = value.value
        if isinstance(plain, (str, int, float)):
            return plain
        return str(getattr(value, "label", value.name))
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        items = cast("dict[object, object]", value).items()
        return {str(key): _jsonable(item) for key, item in items}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in cast("Sequence[object]", list(value))]  # pyright: ignore[reportUnknownArgumentType]  # any iterable of JSON values
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _Runner:
    """The measurements, on one loaded engine."""

    def __init__(self, llm: LLM, options: BenchOptions) -> None:
        self.llm = llm
        self.options = options
        self._counter = 0
        tokenizer = llm.tokenizer.tokenizer
        if tokenizer is None:
            raise ValueError("ayaka bench needs a model with a tokenizer")
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _request(self, prompt: tuple[int, ...], max_tokens: int, *, cache: bool) -> Any:
        from ayaka.core.params import SamplingParams
        from ayaka.core.request import GenerateRequest
        from ayaka.core.types import RequestId

        self._counter += 1
        return GenerateRequest(
            request_id=RequestId(f"bench-{self._counter}"),
            prompt=prompt,
            sampling=SamplingParams(
                temperature=0.0, max_tokens=max_tokens, ignore_eos=True, seed=self.options.seed
            ),
            prefix_cache=cache,
        )

    def _run(self, request: Any) -> RequestOutput:
        output = cast("RequestOutput", self.llm.generate(request))
        if output.error is not None:
            raise RuntimeError(f"a benchmark request failed: {output.error.message}")
        return output

    def prompt_read(self, prompt: tuple[int, ...]) -> list[float]:
        """Seconds to read the prompt and produce one token, cold, per repetition."""
        self._run(self._request(prompt[:64], 1, cache=False))  # warm-up, discarded
        seconds: list[float] = []
        for _ in range(self.options.repeats):
            started = time.perf_counter()
            self._run(self._request(prompt, 1, cache=False))
            seconds.append(time.perf_counter() - started)
        return seconds

    def generation(self, prompt: tuple[int, ...]) -> list[float]:
        """Tokens per second after the first token, per repetition."""
        self._run(self._request(prompt[:64], 8, cache=True))  # warm-up, discarded
        rates: list[float] = []
        for _ in range(self.options.repeats):
            stamps: list[tuple[float, int]] = []
            for output in self.llm.stream(self._request(prompt, GENERATION_TOKENS, cache=True)):
                if output.error is not None:
                    raise RuntimeError(f"a benchmark request failed: {output.error.message}")
                count = sum(len(sequence.token_ids) for sequence in output.outputs)
                if count:
                    stamps.append((time.perf_counter(), count))
            first_time, first_count = stamps[0]
            later = sum(count for _, count in stamps[1:])
            elapsed = stamps[-1][0] - first_time
            if later and elapsed > 0:
                rates.append(later / elapsed)
            _LOG.debug("generation: %d tokens after the first %d", later, first_count)
        return rates

    def first_token(self, system: str, users: tuple[str, str]) -> tuple[float, float]:
        """TTFT of a fresh system prompt (cold) and of the same one with another turn (warm)."""
        from ayaka.core.params import SamplingParams

        sampling = SamplingParams(temperature=0.0, max_tokens=4, seed=self.options.seed)
        times: list[float] = []
        for user in users:
            started = time.perf_counter()
            first: float | None = None
            for output in self.llm.stream_chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                sampling,
            ):
                if output.error is not None:
                    raise RuntimeError(f"a benchmark request failed: {output.error.message}")
                if first is None and any(sequence.token_ids for sequence in output.outputs):
                    first = time.perf_counter() - started
            times.append(first if first is not None else time.perf_counter() - started)
        return times[0], times[1]

    def agent(self, system: str, file_text: str) -> dict[str, object]:
        """A session of three turns: ask, a tool call with its result, an edited result."""
        from ayaka.core.params import SamplingParams

        sampling = SamplingParams(temperature=0.0, max_tokens=32, seed=self.options.seed)
        call = '<tool_call>{"name": "read_file", "arguments": {"path": "report.py"}}</tool_call>'
        edited = file_text[: len(file_text) * 3 // 4] + "\n# (the rest was cut by the tool)\n"
        turns = [
            [{"role": "user", "content": "Find why the totals in report.py are wrong."}],
            [
                {"role": "assistant", "content": call},
                {"role": "user", "content": f"Tool result for read_file:\n{file_text}"},
            ],
            [
                {"role": "assistant", "content": "The loop stops early. Let me re-read it."},
                {"role": "user", "content": f"Tool result for read_file:\n{edited}"},
            ],
        ]
        history: list[dict[str, str]] = [{"role": "system", "content": system}]
        rows: list[dict[str, object]] = []
        total = 0.0
        for turn in turns:
            history.extend(turn)
            started = time.perf_counter()
            output = self.llm.chat(history, sampling)
            seconds = time.perf_counter() - started
            total += seconds
            if output.error is not None:
                raise RuntimeError(f"a benchmark request failed: {output.error.message}")
            usage = output.usage
            rows.append(
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "cached_tokens": usage.cached_tokens,
                    "cached_share": usage.cached_tokens / usage.prompt_tokens
                    if usage.prompt_tokens
                    else 0.0,
                    "seconds": seconds,
                }
            )
            # The next turn brings the assistant message the session recorded, not this
            # model's answer, so every run sends the same history.
        later = [cast("float", row["cached_share"]) for row in rows[1:]]
        return {"turns": rows, "cached_share": median_of(later), "total_s": total}

    def sanity(self) -> list[SanityItem]:
        """The forced tool call and the JSON answer, each checked against its schema."""
        from ayaka.utils.import_utils import has_module

        if not has_module("llguidance"):
            reason = (
                "the structured extra (llguidance) is not installed; uv sync --extra structured"
            )
            return [
                SanityItem("tool call", passed=False, detail=reason),
                SanityItem("structured output", passed=False, detail=reason),
            ]
        return [self._tool_call(), self._structured()]

    def _constrained(self, prompt: str, structured: Any) -> str:
        from ayaka.core.params import SamplingParams
        from ayaka.core.request import GenerateRequest
        from ayaka.core.types import RequestId

        self._counter += 1
        output = self.llm.generate(
            GenerateRequest(
                request_id=RequestId(f"bench-{self._counter}"),
                prompt=prompt,
                sampling=SamplingParams(temperature=0.0, max_tokens=96, seed=self.options.seed),
                structured=structured,
            )
        )
        output = cast("RequestOutput", output)
        if output.error is not None:
            raise RuntimeError(output.error.message)
        return output.outputs[0].text if output.outputs else ""

    def _tool_call(self) -> SanityItem:
        from ayaka.core.constraints import (
            ConstraintKind,
            StructuredParams,
            ToolCallSyntax,
            ToolSchema,
        )

        schema = {
            "type": "object",
            "properties": {"city": {"type": "string", "maxLength": 24}},
            "required": ["city"],
            "additionalProperties": False,
        }
        from ayaka.entrypoints.parsers.markers import resolve_marker

        # The hermes markers: one token when the vocabulary has them, else spelled out.
        opening, closing = "<tool_call>", "</tool_call>"
        try:
            open_id = resolve_marker(opening, self.tokenizer).token_id
            close_id = resolve_marker(closing, self.tokenizer).token_id
        except ValueError as exc:
            return SanityItem("tool call", passed=False, detail=f"no usable markers: {exc}")
        syntax = ToolCallSyntax(
            opening,
            closing,
            open_id,
            close_id,
            () if open_id is not None else tuple(self.encode(opening)),
        )
        params = StructuredParams(
            kind=ConstraintKind.TOOL_CALL,
            tools=(ToolSchema("get_weather", schema),),
            syntax=syntax,
            tool_choice="get_weather",
            strict=True,
            parallel=False,
        )
        try:
            text = self._constrained("What is the weather in Hanoi? Call a tool.", params)
        except Exception as exc:  # the item fails with the reason; the run goes on
            return SanityItem("tool call", passed=False, detail=f"the request failed: {exc}")
        body = text.split(opening, 1)[-1].split(closing, 1)[0].strip()
        call = _json_object(body)
        if call is None:
            return SanityItem(
                "tool call", passed=False, detail=f"not a JSON object: {body[:120]!r}"
            )
        arguments = call.get("arguments")
        city = (
            cast("dict[str, object]", arguments).get("city")
            if isinstance(arguments, dict)
            else None
        )
        valid = (
            call.get("name") == "get_weather"
            and isinstance(arguments, dict)
            and len(cast("dict[str, object]", arguments)) == 1
            and isinstance(city, str)
            and len(city) <= 24
        )
        return SanityItem("tool call", passed=valid, detail=f"call: {body[:160]}")

    def _structured(self) -> SanityItem:
        from ayaka.core.constraints import ConstraintKind, StructuredParams

        schema = {
            "type": "object",
            "properties": {
                "length": {"type": "integer", "minimum": 0, "maximum": 1000},
                "unit": {"type": "string", "enum": ["mm", "cm", "m"]},
            },
            "required": ["length", "unit"],
            "additionalProperties": False,
        }
        params = StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=schema)
        try:
            text = self._constrained("How long is a pencil? Answer in JSON.", params)
        except Exception as exc:  # the item fails with the reason; the run goes on
            return SanityItem(
                "structured output", passed=False, detail=f"the request failed: {exc}"
            )
        value = _json_object(text)
        if value is None:
            return SanityItem(
                "structured output", passed=False, detail=f"not a JSON object: {text[:120]!r}"
            )
        length = value.get("length")
        valid = (
            set(value) == {"length", "unit"}
            and isinstance(length, int)
            and not isinstance(length, bool)
            and 0 <= length <= 1000
            and value.get("unit") in ("mm", "cm", "m")
        )
        return SanityItem("structured output", passed=valid, detail=f"answer: {text[:160]}")


def _json_object(text: str) -> dict[str, object] | None:
    try:
        value: object = json.loads(text)
    except ValueError:
        return None
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _lengths(options: BenchOptions, longest: int) -> list[int]:
    """The length points: the fixed ones that fit, and the longest the machine holds."""
    fixed = QUICK_LENGTHS if options.quick else LENGTHS
    points = [length for length in fixed if length <= longest]
    if not options.quick and longest not in points:
        points.append(longest)
    return points


def run_bench(
    config: EngineConfig,
    options: BenchOptions,
    *,
    say: Callable[[str], None] = print,
    load: Callable[[EngineConfig, SessionRecorder | None], LLM] | None = None,
) -> dict[str, object]:
    """Measure the model and return ``results.json`` as data.

    Raises:
        Whatever loading the engine raises; a failed measurement raises ``RuntimeError``.
    """
    from ayaka.engine.build import plan_engine
    from ayaka.engine.llm import LLM
    from ayaka.engine.recorder import SessionRecorder

    started = datetime.now(UTC)
    gpu_start = gpu_state()
    power = power_source()
    notes: list[str] = []
    if power == "battery":
        notes.append("on battery: the speed numbers may be lower than on mains")
        say("warning: running on battery; the result is marked")
    plan = plan_engine(replace(config, source=replace(config.source, offline=True)))
    say(f"plan: {plan.max_context_tokens} tokens per request, {plan.max_running} at once")

    def from_config(config: EngineConfig, recorder: SessionRecorder | None) -> LLM:
        return LLM.from_config(config, recorder=recorder)

    loader = load or from_config
    recorder = SessionRecorder(options.record) if options.record is not None else None
    with MemorySampler() as memory:
        llm = loader(config, recorder)
        try:
            if recorder is not None:
                say(
                    f"recording every request to {recorder.path} (token ids of the "
                    "benchmark prompts)"
                )
            runner = _Runner(llm, options)
            # The loaded engine's cache, not the plan: the start-up run may have measured more
            # runtime memory than the plan assumed and given the cache fewer blocks.
            longest = min(plan.max_context_tokens, llm.info.max_context_tokens)
            try:
                results = _measure(runner, longest, options, say)
            except AyakaError as exc:  # a measurement the engine refused is the run failing
                raise RuntimeError(f"a benchmark request failed: {exc}") from exc
        finally:
            llm.close()
            del llm
            gc.collect()
    sanity = cast("list[SanityItem]", results.pop("sanity"))
    label = model_label(options.model)
    engine = cast("dict[str, object]", _jsonable(config))
    source = engine.get("source")
    if isinstance(source, dict):
        cast("dict[str, object]", source)["model"] = label
    valid = all(item.passed for item in sanity)
    if not valid:
        notes.append("a sanity check failed: the speed numbers are kept but not valid")
    return {
        "schema": SCHEMA,
        "valid": valid,
        "model": {
            "name": label,
            "architecture": plan.model_config.architecture,
            "dtype": plan.group.spec.dtype.label,
            "weights_bytes": plan.weights_bytes,
            "quantization": plan.quantization,
            "max_position_embeddings": plan.model_config.max_position_embeddings,
        },
        "engine": engine,
        "backends": dict(plan.providers),
        "plan": {
            "max_context_tokens": plan.max_context_tokens,
            "engine_context_tokens": longest,
            "max_running": plan.max_running,
            "kv_blocks": plan.kv_blocks,
            "block_size": plan.block_size,
            "chunk_tokens": plan.chunk_tokens,
            "usable_bytes": plan.budget.usable_bytes,
        },
        "machine": machine_info(),
        "power": power,
        "thermal": {
            "gpu_start_c": gpu_start.get("temperature_c"),
            "gpu_end_c": gpu_state().get("temperature_c"),
        },
        "options": {"quick": options.quick, "seed": options.seed, "repeats": options.repeats},
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        **results,
        "sanity": [asdict(item) for item in sanity],
        "memory": {
            "peak_vram_bytes": memory.peak_vram,
            "peak_rss_bytes": memory.peak_rss,
            "samples": memory.samples,
            "vram_scope": "the whole GPU (NVML)",
        },
        "notes": notes,
    }


def _measure(
    runner: _Runner, longest: int, options: BenchOptions, say: Callable[[str], None]
) -> dict[str, object]:
    # Room for the answer: the generation runs produce GENERATION_TOKENS after the prompt.
    usable = longest - GENERATION_TOKENS - 16
    lengths = _lengths(options, usable)
    prompts = {
        (group, length): build_prompt(group, length, runner.encode)
        for group in GROUPS
        for length in lengths
    }
    prompt_read: list[dict[str, object]] = []
    for (group, length), prompt in prompts.items():
        say(f"prompt read: {group} {length} tokens")
        seconds = runner.prompt_read(prompt.token_ids)
        rates = [length / value for value in seconds]
        prompt_read.append(
            {
                "group": group,
                "length": length,
                "seconds": seconds,
                "tokens_per_s": rates,
                "median_tokens_per_s": median_of(rates),
            }
        )
    generation: list[dict[str, object]] = []
    for length in lengths:
        say(f"generation: {GENERATION_TOKENS} tokens after {length}")
        rates = runner.generation(prompts["prose", length].token_ids)
        generation.append(
            {
                "context": length,
                "output_tokens": GENERATION_TOKENS,
                "tokens_per_s": rates,
                "median_tokens_per_s": median_of(rates),
            }
        )
    say("first token: cold and warm")
    # The agent session holds the system prompt, a file and three short turns.
    system_tokens = min(SYSTEM_TOKENS, usable // 4)
    file_tokens = min(1024, usable // 8)  # the last turn holds the file twice
    system_base = runner.tokenizer.decode(material_ids("prose", system_tokens, runner.encode))
    cold: list[float] = []
    warm: list[float] = []
    for repetition in range(options.repeats + 1):
        system = f"Session {options.seed}-{repetition}.\n{system_base}"
        first, second = runner.first_token(system, ("Say hello.", "Say goodbye."))
        if repetition:  # the first pair warms the engine up
            cold.append(first)
            warm.append(second)
    say("agent session")
    file_text = runner.tokenizer.decode(material_ids("edit", file_tokens, runner.encode))
    agent = runner.agent(f"Agent run {options.seed}.\n{system_base}", file_text)
    say("sanity")
    sanity = runner.sanity()
    return {
        "prompt_set": {
            "version": PROMPT_SET_VERSION,
            "groups": list(GROUPS),
            "lengths": lengths,
            "digests": {f"{g}/{n}": prompt_digest(p) for (g, n), p in prompts.items()},
        },
        "prompt_read": prompt_read,
        "generation": generation,
        "ttft": {
            "system_tokens": len(runner.encode(system_base)),
            "cold_s": cold,
            "warm_s": warm,
            "cold_median_s": median_of(cold),
            "warm_median_s": median_of(warm),
        },
        "agent": agent,
        "sanity": sanity,
    }


def bench_command(args: argparse.Namespace, config: EngineConfig) -> int:
    """``ayaka bench``: run, validate, write and summarise; exit 2 when a sanity check failed."""
    from ayaka.entrypoints.bench_results import (
        default_out,
        summary_lines,
        validate_results,
        write_results,
    )

    options = BenchOptions(model=args.model, quick=args.quick, seed=args.seed, record=args.record)
    if args.record is not None:
        say(f"warning: {args.record} will hold the benchmark's prompts and answers as token ids")
    try:
        results = run_bench(config, options, say=say)
    except AyakaError:
        raise  # a model that does not load or fit exits with its own code, as serve does
    except (RuntimeError, OSError) as exc:
        say(f"the benchmark failed: {exc}")
        return EXIT_BUG
    problems = validate_results(results)
    if problems:
        say("results.json does not validate: " + "; ".join(problems))
        return EXIT_BUG
    history = args.history
    if history is None and (Path.cwd() / "benchmarks" / "history").is_dir():
        history = Path.cwd() / "benchmarks" / "history"
    out = args.out or default_out(results, Path.cwd())
    path, line = write_results(results, out, history)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for text in summary_lines(results):
            print(text)
    say(f"wrote {path}" + (f"; summary line in {line}" if line else ""))
    return EXIT_OK if results["valid"] else EXIT_USAGE
