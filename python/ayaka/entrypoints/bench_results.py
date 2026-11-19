"""``results.json`` of ``ayaka bench``: checking it, summarising it and writing it.

:func:`validate_results` refuses a file whose medians do not match its raw runs or that misses a
field the schema requires; :func:`summary_lines` is what the command prints and
:func:`summary_line` the one line appended to the history file.
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

from ayaka.entrypoints.bench import SCHEMA
from ayaka.entrypoints.bench_machine import machine_slug

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = ["default_out", "summary_line", "summary_lines", "validate_results", "write_results"]


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
