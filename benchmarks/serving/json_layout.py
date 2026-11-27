"""What the JSON layout of a schema constraint does to the answers of a real model.

A running OpenAI-compatible server answers JSON-schema requests (``response_format`` of type
``json_schema``) for two schemas, a flat person record and a nested order, over several prompts
and seeds. Each layout under test is sent as the schema's own ``x-guidance`` options, which Ayaka
keeps as they are; the arm named ``server default`` sends none and measures what the server does
by itself. Per layout the result counts:

    valid            answers ``json.loads`` accepts
    cut_at_length    answers stopped by ``max_tokens`` (a string or a reasoning that never ended)
    colon_string     answers with a string value that starts with ``:`` (``{"name": ":name:"}``),
                     the placeholder a model writes when the layout leaves no space after a colon
    tokens           mean completion tokens per schema, reasoning included

Nothing is estimated: every number is a count over the answers the server returned. Point
``--base-url`` at the server, for example::

    uv run ayaka serve Qwen/Qwen3-8B --device cuda --dtype bf16 --backends flashinfer,torch \
        --cuda-graph --reasoning-parser think --port 8100
    uv run python benchmarks/serving/json_layout.py --base-url http://127.0.0.1:8100 \
        --model Qwen/Qwen3-8B --out json_layout.json

``--no-thinking`` adds ``reasoning_effort: "none"`` to every request, for a server that reads it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import urllib.request
from pathlib import Path
from typing import Any, Final

_FLAT: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}
_GEO: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
    "required": ["lat", "lon"],
}
_ADDRESS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {"street": {"type": "string"}, "city": {"type": "string"}, "geo": _GEO},
    "required": ["street", "city", "geo"],
}
_NESTED: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "customer": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "address": _ADDRESS},
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
_PROMPTS: Final[tuple[tuple[str, dict[str, Any], str], ...]] = (
    ("flat", _FLAT, "Invent a person. Reply in JSON."),
    ("flat", _FLAT, "Give me a fictional customer record as JSON."),
    ("flat", _FLAT, "Describe a character from a novel as JSON."),
    ("flat", _FLAT, "Make up a user profile in JSON."),
    ("nested", _NESTED, "Create a realistic example e-commerce order with two items as JSON."),
    ("nested", _NESTED, "Make up an order shipped to Hanoi as JSON."),
)
_LAYOUTS: Final[dict[str, dict[str, object] | None]] = {
    "server default": None,
    "compact": {"whitespace_flexible": False},
    "single line, spaced": {
        "whitespace_flexible": False,
        "item_separator": ", ",
        "key_separator": ": ",
    },
}
_COLON_STRING: Final = re.compile(r'"\s*:\s*":')


def _ask(
    args: argparse.Namespace, schema: dict[str, Any], prompt: str, seed: int
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": seed,
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
    }
    if args.no_thinking:
        body["reasoning_effort"] = "none"
    request = urllib.request.Request(  # noqa: S310  # the URL is the server under test
        f"{args.base_url}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310  # the URL is the server under test
        return json.load(response)


def _measure(args: argparse.Namespace, options: dict[str, object] | None) -> dict[str, Any]:
    counts = {"runs": 0, "valid": 0, "cut_at_length": 0, "colon_string": 0}
    tokens: dict[str, list[int]] = {"flat": [], "nested": []}
    samples: list[str] = []
    for seed in range(args.seeds):
        for kind, base, prompt in _PROMPTS:
            schema = base if options is None else {**base, "x-guidance": options}
            answer = _ask(args, schema, prompt, seed)
            choice = answer["choices"][0]
            content = choice["message"].get("content") or ""
            try:
                json.loads(content)
            except ValueError:
                valid = False
            else:
                valid = True
            counts["runs"] += 1
            counts["valid"] += valid
            counts["cut_at_length"] += choice["finish_reason"] == "length"
            counts["colon_string"] += bool(_COLON_STRING.search(content))
            tokens[kind].append(answer["usage"]["completion_tokens"])
            if seed == 0:
                samples.append(content[:200])
    return {
        **counts,
        "mean_tokens": {kind: round(statistics.fmean(v), 1) for kind, v in tokens.items()},
        "samples_seed_0": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--layouts", nargs="*", default=list(_LAYOUTS), choices=list(_LAYOUTS))
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "label": args.label,
        "model": args.model,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "no_thinking": args.no_thinking,
        "layouts": {},
    }
    for name in args.layouts:
        row = _measure(args, _LAYOUTS[name])
        result["layouts"][name] = row
        print(
            f"{name:22s} valid {row['valid']}/{row['runs']}  cut {row['cut_at_length']}  "
            f"colon-string {row['colon_string']}  tokens {row['mean_tokens']}",
            flush=True,
        )
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
