"""Write docs/CAPABILITIES.md from docs/capabilities.toml and the result files it names.

    uv run python tools/generate_capabilities.py           # write the file
    uv run python tools/generate_capabilities.py --check   # exit 1 when the file differs

Every number in the table is read from a result file (``ayaka bench``'s ``results.json``) and
carries that file's path and the Ayaka version and commit that produced it. A ``[[quality]]``
entry names the result of ``benchmarks/e2e/quant_kl.py`` for a quantized checkpoint; the page
shows its first-token KL against the 16-bit original in a second table. A row without a result
says ``untested``; a row that does not run says why. Nothing is rounded up: "untested" is never
shown as "runs".
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE: Final = ROOT / "docs" / "capabilities.toml"
TARGET: Final = ROOT / "docs" / "CAPABILITIES.md"
STATUSES: Final = ("runs", "untested", "does-not-run")
UNTESTED: Final = "untested"
_KEYS: Final = {"model", "precision", "machine", "status", "result", "reason"}
_INTRO: Final = (
    "What is known about each model, precision and machine: **runs** (a result file shows it), "
    "**untested** (nobody ran it: not a promise either way) or **does not run** (a result file "
    "shows it, and why). Every number comes from the result file in its row, with the version "
    "that produced it. Prompt read is the median of the cold 4K-token prose prompt; generation is "
    "the median after a 4K-token context; the context is what one request could hold on that run; "
    "the last two columns are `ayaka bench`'s sanity checks (a forced tool call and a JSON "
    "answer, each checked against its schema)."
)
_QUALITY_KEYS: Final = {"model", "precision", "machine", "reference", "result"}
_QUALITY_INTRO: Final = (
    "First-token quality of quantized checkpoints against their 16-bit original, from "
    "`benchmarks/e2e/quant_kl.py`: on fixed prompts rendered with the chat template, the "
    "distribution of the first generated token, read from Ayaka's own models in turn on the same "
    "card. KL is in nats, KL(16 bit || quantized), averaged over the prompts (the largest in "
    "brackets); top-1 is how often the two most likely tokens agree; the threshold is the one "
    "fixed before the run."
)
_QUALITY_HEADER: Final = (
    "Model",
    "Precision",
    "Machine",
    "Against",
    "Prompts",
    "First-token KL",
    "Top-1 agreement",
    "Threshold",
    "Result",
)
_HEADER: Final = (
    "Model",
    "Precision",
    "Machine",
    "Status",
    "Prompt read (4K)",
    "Generation (after 4K)",
    "Context per request",
    "Tool call",
    "JSON answer",
    "Result",
)


def load(path: Path = SOURCE) -> list[dict[str, Any]]:
    """The rows of the source file, checked.

    Raises:
        ValueError: If the file breaks its own rules.
    """
    return _load(path)[0]


def load_quality(path: Path = SOURCE) -> list[dict[str, Any]]:
    """The ``[[quality]]`` entries of the source file, checked; empty when it has none.

    Raises:
        ValueError: If an entry lacks a key, has an unknown one, or names a missing file or one
            that is not a ``quant_kl`` result.
    """
    return _load(path)[1]


def _load(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError(f"{path}: version must be 1")
    rows = data.get("row")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: no [[row]]")
    for index, row in enumerate(rows):
        where = f"{path}: row {index + 1}"
        for key in ("model", "precision", "machine", "status"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{where}: {key} is required")
        status = row["status"]
        if status not in STATUSES:
            raise ValueError(f"{where}: status must be one of {STATUSES}, not {status!r}")
        if status != UNTESTED and not row.get("result"):
            raise ValueError(f"{where}: a {status} row needs the result file that shows it")
        if status == UNTESTED and row.get("result"):
            raise ValueError(f"{where}: an untested row has no result; set its status")
        if status == "does-not-run" and not row.get("reason"):
            raise ValueError(f"{where}: a does-not-run row needs its reason")
        unknown = set(row) - _KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        if row.get("result") and not (ROOT / row["result"]).is_file():
            raise ValueError(f"{where}: {row['result']} does not exist")
    quality = data.get("quality", [])
    if not isinstance(quality, list):
        raise ValueError(f"{path}: quality must be a list of [[quality]] tables")
    for index, entry in enumerate(quality):
        where = f"{path}: quality {index + 1}"
        for key in sorted(_QUALITY_KEYS):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f"{where}: {key} is required")
        unknown = set(entry) - _QUALITY_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
        file = ROOT / entry["result"]
        if not file.is_file():
            raise ValueError(f"{where}: {entry['result']} does not exist")
        if json.loads(file.read_text(encoding="utf-8")).get("benchmark") != "quant_kl":
            raise ValueError(f"{where}: {entry['result']} is not a quant_kl result")
    return rows, quality


def _result(row: dict[str, Any]) -> dict[str, Any] | None:
    if not row.get("result"):
        return None
    return json.loads((ROOT / row["result"]).read_text(encoding="utf-8"))


def _median(rows: list[dict[str, Any]], **match: object) -> float | None:
    for entry in rows:
        if all(entry.get(key) == value for key, value in match.items()):
            return entry.get("median_tokens_per_s")
    return None


def _cells(row: dict[str, Any]) -> list[str]:
    """Prompt read at 4K, generation after 4K, context per request, the two sanity checks."""
    if row["status"] == "does-not-run":
        return ["does not run"] * 5
    result = _result(row)
    if row["status"] != "runs" or result is None:
        return [UNTESTED] * 5
    read = _median(result["prompt_read"], group="prose", length=4096)
    generate = _median(result["generation"], context=4096)
    plan = result["plan"]
    context = plan.get("engine_context_tokens") or plan.get("max_context_tokens")
    sanity = {item["name"]: item["passed"] for item in result["sanity"]}

    def check(name: str) -> str:
        if name not in sanity:
            return UNTESTED
        return "pass" if sanity[name] else "fail"

    return [
        f"{read:,.0f} tok/s" if read is not None else UNTESTED,
        f"{generate:,.1f} tok/s" if generate is not None else UNTESTED,
        f"{context:,} tokens" if context else UNTESTED,
        check("tool call"),
        check("structured output"),
    ]


def _source(row: dict[str, Any]) -> str:
    if not row.get("result"):
        return ""
    result = _result(row) or {}
    ayaka = (result.get("machine") or {}).get("ayaka") or {}
    version = ayaka.get("version")
    commit = (ayaka.get("commit") or "")[:7]
    stamp = (f"ayaka {version}" + (f" @ {commit}" if commit else "")) if version else ""
    link = f"[{Path(row['result']).name}](../{row['result']})"
    return link + (f", {stamp}" if stamp else "")


def _quality_cells(entry: dict[str, Any]) -> list[str]:
    result = json.loads((ROOT / entry["result"]).read_text(encoding="utf-8"))
    commit = ((result.get("machine") or {}).get("commit") or "")[:7]
    link = f"[{Path(entry['result']).name}](../{entry['result']})"
    verdict = "passes" if result["passes"] else "fails"
    return [
        entry["model"],
        entry["precision"],
        entry["machine"],
        entry["reference"],
        str(result["prompts"]),
        f"{result['mean_kl_nats']:.4f} ({result['max_kl_nats']:.4f})",
        f"{result['top1_agreement']:.0%}",
        f"{result['threshold_nats']} nats: {verdict}",
        link + (f", @ {commit}" if commit else ""),
    ]


def render(rows: list[dict[str, Any]], quality: list[dict[str, Any]] | None = None) -> str:
    """The Markdown page."""
    lines = [
        "# Capabilities",
        "",
        "<!-- Generated by tools/generate_capabilities.py from docs/capabilities.toml. -->",
        "<!-- Do not edit: the drift test fails. -->",
        "",
        _INTRO,
        "",
        "| " + " | ".join(_HEADER) + " |",
        "|" + " --- |" * len(_HEADER),
    ]
    for row in rows:
        status = row["status"].replace("-", " ")
        if row["status"] == "does-not-run":
            status += f": {row['reason']}"
        cells = [row["model"], row["precision"], row["machine"], status, *_cells(row), _source(row)]
        lines.append("| " + " | ".join(cells) + " |")
    if quality:
        lines += [
            "",
            "## Quantized against 16 bit",
            "",
            _QUALITY_INTRO,
            "",
            "| " + " | ".join(_QUALITY_HEADER) + " |",
            "|" + " --- |" * len(_QUALITY_HEADER),
        ]
        lines += ["| " + " | ".join(_quality_cells(entry)) + " |" for entry in quality]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write docs/CAPABILITIES.md.")
    parser.add_argument("--check", action="store_true", help="exit 1 when the file differs")
    args = parser.parse_args(argv)
    page = render(load(), load_quality())
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != page:
            print(f"{TARGET} is not what the generator writes; run it", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
