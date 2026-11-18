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

import contextlib
import gc
import json
import logging
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from ayaka.__version__ import __version__
from ayaka.entrypoints.bench_prompts import (
    GROUPS,
    PROMPT_SET_VERSION,
    build_prompt,
    material_ids,
    prompt_digest,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ayaka.configs.engine import EngineConfig
    from ayaka.core.outputs import RequestOutput
    from ayaka.engine.llm import LLM

__all__ = [
    "GENERATION_TOKENS",
    "LENGTHS",
    "QUICK_LENGTHS",
    "SCHEMA",
    "BenchOptions",
    "SanityItem",
    "default_out",
    "machine_slug",
    "median_of",
    "model_label",
    "power_source",
    "run_bench",
    "summary_line",
    "summary_lines",
    "validate_results",
    "write_results",
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


# -- the machine -------------------------------------------------------------------------------


def _version_of(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(  # noqa: S603  # a fixed argv
            ["git", "-C", str(root), *args],  # noqa: S607  # git from PATH
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _running_on_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def power_source(
    *,
    platform_name: str | None = None,
    sysfs: Path = Path("/sys/class/power_supply"),
    wsl: bool | None = None,
    windows_status: Callable[[], bool | None] | None = None,
) -> str:
    """``"ac"``, ``"battery"`` or ``"unknown"``.

    Linux reads ``/sys/class/power_supply``; WSL2 has none there and asks Windows through
    ``powershell.exe``; Windows asks ``GetSystemPowerStatus``.
    """
    name = platform_name or sys.platform
    if name == "win32" or (wsl if wsl is not None else _running_on_wsl()):
        online = (windows_status or _windows_on_ac)()
        return "unknown" if online is None else ("ac" if online else "battery")
    if not sysfs.is_dir():
        return "unknown"
    mains = [entry for entry in sysfs.iterdir() if _read(entry / "type") == "Mains"]
    batteries = [entry for entry in sysfs.iterdir() if _read(entry / "type") == "Battery"]
    if any(_read(entry / "online") == "1" for entry in mains):
        return "ac"
    if batteries:
        return "battery"
    return "ac" if not mains else "unknown"  # a machine with neither is a desktop on mains


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _windows_on_ac() -> bool | None:
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = (  # the ctypes layout
                ("ac_line", ctypes.c_byte),
                ("battery_flag", ctypes.c_byte),
                ("battery_percent", ctypes.c_byte),
                ("system_status", ctypes.c_byte),
                ("battery_life", ctypes.c_ulong),
                ("battery_full_life", ctypes.c_ulong),
            )

        status = _Status()
        kernel = getattr(ctypes, "windll").kernel32  # noqa: B009  # windll exists on Windows only
        if not kernel.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        return {0: False, 1: True}.get(int(status.ac_line))
    try:  # WSL2: Windows answers through interop
        result = subprocess.run(
            [  # noqa: S607  # Windows' own PowerShell through WSL interop
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Add-Type -AssemblyName System.Windows.Forms;"
                "[System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    answer = result.stdout.strip().lower()
    return {"online": True, "offline": False}.get(answer)


def _gpu_state() -> dict[str, object]:
    """Temperature, memory and driver of GPU 0 from NVML; empty without one."""
    from ayaka.utils.nvml_utils import get_driver_version, nvml_session

    with nvml_session() as nvml:
        if nvml is None:
            return {}
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "temperature_c": int(nvml.nvmlDeviceGetTemperature(handle, 0)),
                "memory_used_bytes": int(memory.used),
                "memory_total_bytes": int(memory.total),
                "driver": get_driver_version(nvml),
            }
        except Exception:  # NVML differs between drivers; a missing value is not an error
            _LOG.debug("bench: NVML did not answer", exc_info=True)
            return {}


def machine_slug(gpu: str | None) -> str:
    """A name for the results folder from the GPU, never the host name: ``rtx-3050-laptop``."""
    if not gpu:
        return "cpu"
    words = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    return re.sub(r"^(nvidia-)?(geforce-)?", "", words) or "gpu"


def _machine() -> dict[str, object]:
    import torch

    cuda = torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info() if cuda else (None, None)
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "vram_free_bytes": free,
        "vram_total_bytes": total,
        "os": platform.platform(),
        "wsl2": _running_on_wsl(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": _version_of("flashinfer-python"),
        "llguidance": _version_of("llguidance"),
        "transformers": _version_of("transformers"),
        "ayaka": {
            "version": __version__,
            "commit": commit,
            "dirty": bool(status) if status is not None else None,
        },
    }


# -- memory sampling ---------------------------------------------------------------------------


def _rss_bytes() -> int | None:
    """The process's resident memory now."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = (  # the ctypes layout
                ("cb", wintypes.DWORD),
                ("page_faults", wintypes.DWORD),
                ("peak_working_set", ctypes.c_size_t),
                ("working_set", ctypes.c_size_t),
                ("quota_peak_paged", ctypes.c_size_t),
                ("quota_paged", ctypes.c_size_t),
                ("quota_peak_nonpaged", ctypes.c_size_t),
                ("quota_nonpaged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("peak_pagefile", ctypes.c_size_t),
            )

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        windll = getattr(ctypes, "windll")  # noqa: B009  # windll exists on Windows only
        process = windll.kernel32.GetCurrentProcess()
        if not windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.working_set)
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return pages * os.sysconf("SC_PAGE_SIZE")


class _MemorySampler:
    """Samples VRAM (NVML, the whole device) and RAM (this process) on a thread."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.peak_vram: int | None = None
        self.peak_rss: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        from ayaka.utils.nvml_utils import nvml_session

        with nvml_session() as nvml:
            handle = None
            if nvml is not None:
                with contextlib.suppress(Exception):
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
            while True:
                if handle is not None and nvml is not None:
                    with contextlib.suppress(Exception):
                        used = int(nvml.nvmlDeviceGetMemoryInfo(handle).used)
                        self.peak_vram = max(self.peak_vram or 0, used)
                rss = _rss_bytes()
                if rss is not None:
                    self.peak_rss = max(self.peak_rss or 0, rss)
                self.samples += 1
                if self._stop.wait(self.interval):
                    return


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
        tokenizer = llm.engine.tokenizer.tokenizer
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
        rendered = self.llm.engine.tokenizer.tokenizer
        assert rendered is not None  # noqa: S101  # checked in __init__
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
    load: Callable[[EngineConfig], LLM] | None = None,
) -> dict[str, object]:
    """Measure the model and return ``results.json`` as data.

    Raises:
        Whatever loading the engine raises; a failed measurement raises ``RuntimeError``.
    """
    from ayaka.engine.build import plan_engine
    from ayaka.engine.llm import LLM

    started = datetime.now(UTC)
    gpu_start = _gpu_state()
    power = power_source()
    notes: list[str] = []
    if power == "battery":
        notes.append("on battery: the speed numbers may be lower than on mains")
        say("warning: running on battery; the result is marked")
    plan = plan_engine(replace(config, source=replace(config.source, offline=True)))
    say(f"plan: {plan.max_context_tokens} tokens per request, {plan.max_running} at once")
    loader = load or LLM.from_config
    with _MemorySampler() as memory:
        llm = loader(config)
        try:
            if options.record is not None:
                from ayaka.engine.recorder import SessionRecorder

                llm.engine.recorder = SessionRecorder(options.record)
                say(
                    f"recording every request to {llm.engine.recorder.path} (token ids of the "
                    "benchmark prompts)"
                )
            runner = _Runner(llm, options)
            results = _measure(runner, plan.max_context_tokens, options, say)
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
            "max_position_embeddings": plan.model_config.max_position_embeddings,
        },
        "engine": engine,
        "backends": dict(plan.providers),
        "plan": {
            "max_context_tokens": plan.max_context_tokens,
            "max_running": plan.max_running,
            "kv_blocks": plan.kv_blocks,
            "block_size": plan.block_size,
            "chunk_tokens": plan.chunk_tokens,
            "usable_bytes": plan.budget.usable_bytes,
        },
        "machine": _machine(),
        "power": power,
        "thermal": {
            "gpu_start_c": gpu_start.get("temperature_c"),
            "gpu_end_c": _gpu_state().get("temperature_c"),
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


# -- results.json ------------------------------------------------------------------------------

_REQUIRED: Final = (
    "schema",
    "valid",
    "model",
    "engine",
    "backends",
    "plan",
    "machine",
    "power",
    "thermal",
    "prompt_set",
    "prompt_read",
    "generation",
    "ttft",
    "agent",
    "sanity",
    "memory",
    "notes",
)


def validate_results(data: Mapping[str, object]) -> list[str]:
    """What is wrong with a ``results.json``: missing parts, medians that do not match their raw
    values, a ``valid`` flag that disagrees with the sanity items. Empty when it is sound."""
    problems = [f"missing {key!r}" for key in _REQUIRED if key not in data]
    if data.get("schema") != SCHEMA:
        problems.append(f"schema is {data.get('schema')!r}, not {SCHEMA}")
    if problems:
        return problems
    for kind in ("prompt_read", "generation"):
        for row in cast("list[dict[str, object]]", data[kind]):
            raw = cast("list[float]", row["tokens_per_s"])
            if not raw:
                problems.append(f"{kind} row without raw values: {row}")
            elif not _close(row["median_tokens_per_s"], statistics.median(raw)):
                problems.append(f"{kind} median does not match its raw values: {row}")
    ttft = cast("dict[str, object]", data["ttft"])
    for name in ("cold", "warm"):
        raw = cast("list[float]", ttft[f"{name}_s"])
        if raw and not _close(ttft[f"{name}_median_s"], statistics.median(raw)):
            problems.append(f"ttft {name} median does not match its raw values")
    sanity = cast("list[dict[str, object]]", data["sanity"])
    if not sanity:
        problems.append("no sanity item")
    if data["valid"] is not all(bool(item.get("passed")) for item in sanity):
        problems.append("valid does not equal all(sanity passed)")
    return problems


def _close(value: object, expected: float) -> bool:
    return isinstance(value, (int, float)) and abs(float(value) - expected) <= 1e-9 * max(
        1.0, abs(expected)
    )


def summary_lines(results: Mapping[str, object]) -> list[str]:
    """The printed summary: one line per part, no single aggregate."""
    lines = [
        f"model      {cast('dict[str, object]', results['model'])['name']}",
        f"valid      {'yes' if results['valid'] else 'NO (a sanity check failed)'}",
        f"power      {results['power']}",
    ]
    for row in cast("list[dict[str, object]]", results["prompt_read"]):
        lines.append(
            f"read       {row['group']:<6} {row['length']:>6} tokens: "
            f"{cast('float', row['median_tokens_per_s']):,.0f} tok/s (median)"
        )
    for row in cast("list[dict[str, object]]", results["generation"]):
        lines.append(
            f"generate   after {row['context']:>6} tokens: "
            f"{cast('float | None', row['median_tokens_per_s']) or 0:,.1f} tok/s (median)"
        )
    ttft = cast("dict[str, object]", results["ttft"])
    lines.append(
        f"first tok  cold {cast('float', ttft['cold_median_s']) * 1e3:,.0f} ms, warm "
        f"{cast('float', ttft['warm_median_s']) * 1e3:,.0f} ms "
        f"({ttft['system_tokens']}-token system prompt)"
    )
    agent = cast("dict[str, object]", results["agent"])
    share = cast("float | None", agent["cached_share"])
    lines.append(
        f"agent      cached share {share or 0:.0%} of later turns, "
        f"{cast('float', agent['total_s']):.2f} s in all"
    )
    for item in cast("list[dict[str, object]]", results["sanity"]):
        lines.append(f"sanity     {'pass' if item['passed'] else 'FAIL'}  {item['name']}")
    memory = cast("dict[str, object]", results["memory"])
    vram = cast("int | None", memory["peak_vram_bytes"])
    rss = cast("int | None", memory["peak_rss_bytes"])
    lines.append(
        f"memory     peak VRAM {vram / 2**20 if vram else 0:,.0f} MiB (whole GPU), peak RAM "
        f"{rss / 2**20 if rss else 0:,.0f} MiB"
    )
    return lines


def summary_line(results: Mapping[str, object]) -> dict[str, object]:
    """The one line ``benchmarks/history`` keeps per run: the medians and where they came from."""
    machine = cast("dict[str, object]", results["machine"])
    ayaka = cast("dict[str, object]", machine["ayaka"])
    return {
        "date": results.get("finished"),
        "machine": machine_slug(cast("str | None", machine.get("gpu"))),
        "model": cast("dict[str, object]", results["model"])["name"],
        "commit": ayaka.get("commit"),
        "dirty": ayaka.get("dirty"),
        "valid": results["valid"],
        "power": results["power"],
        "read_tok_s": {
            f"{row['group']}/{row['length']}": row["median_tokens_per_s"]
            for row in cast("list[dict[str, object]]", results["prompt_read"])
        },
        "generate_tok_s": {
            str(row["context"]): row["median_tokens_per_s"]
            for row in cast("list[dict[str, object]]", results["generation"])
        },
        "ttft_cold_s": cast("dict[str, object]", results["ttft"])["cold_median_s"],
        "ttft_warm_s": cast("dict[str, object]", results["ttft"])["warm_median_s"],
        "agent_cached_share": cast("dict[str, object]", results["agent"])["cached_share"],
    }


def write_results(
    results: Mapping[str, object], out: Path, history: Path | None
) -> tuple[Path, Path | None]:
    """Write ``results.json`` and append the summary line to ``history/bench.jsonl``."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    line_path: Path | None = None
    if history is not None:
        history.mkdir(parents=True, exist_ok=True)
        line_path = history / "bench.jsonl"
        with line_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary_line(results)) + "\n")
    return path, line_path


def default_out(results: Mapping[str, object], root: Path) -> Path:
    """``benchmarks/results/<date>-<machine>/`` under ``root``."""
    machine = cast("dict[str, object]", results["machine"])
    date = str(results.get("started", ""))[:10] or datetime.now(UTC).strftime("%Y-%m-%d")
    return (
        root
        / "benchmarks"
        / "results"
        / f"{date}-{machine_slug(cast('str | None', machine.get('gpu')))}"
    )
