"""Speculative against plain decoding, in alternating pairs, per prompt class and concurrency.

Every pair builds a fresh plain engine and a fresh engine of the arm, one after the other, and
runs every cell (prompt class and concurrency) on each before closing it. Pair ``p`` builds plain
first when ``p`` is even and the arm first when it is odd (A B, B A, A B, ...), so a slow drift of
the GPU clock lands on both arms. Every run records its throughput, every token id, and the drafts
the controller made and accepted.

``--context-tokens N`` puts about ``N`` tokens of repository code in a system message before every
prompt, the shape of a coding agent's request; the default 0 sends the prompts alone. That context
is the same for every request, so the prefix cache serves it after the first; ``--no-prefix-cache``
turns the cache off in both engines of every pair, so every prompt is computed.

The prompt classes are rendered through the model's own chat template (``--classes`` picks
them; the default is the four of the roadmap's speculative decoding milestone):

* ``code_edit``: rewrite a function with a small change, so most of the answer copies the prompt;
* ``rag_extract``: quote and extract facts from a retrieved document given in the prompt;
* ``creative_chat`` (also ``prose``): a short story, with nothing to copy;
* ``json_tools``: a JSON tool call whose keys and schema are in the prompt;
* ``random``: a list of random words to answer about, which no drafter can predict.

A request at concurrency ``c`` gets its own first line (``Ticket 3 of 8``), so the ``c`` answers are
not one answer computed ``c`` times. Greedy decoding throughout; whether the ids of an arm equal
plain decoding's is recorded per request, not assumed.

Arms are ``SpeculativeConfig`` fields: ``--arm lookup=method:lookup,controller:cost``.

Usage::

    uv run python benchmarks/e2e/speculative_sweep.py --model Qwen/Qwen3-8B \\
        --arm lookup=method:lookup,controller:cost,max_draft_tokens:8 \\
        --out benchmarks/results/<run>/speculative_sweep.json
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _machine import machine, write

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.configs.speculative import SpeculativeConfig
from ayaka.configs.tokenizer import TokenizerConfig
from ayaka.core.params import SamplingParams
from ayaka.engine.llm import LLM
from ayaka.model_loader.source import resolve_source
from ayaka.speculative.policy import CostAwareController, FixedController
from ayaka.utils.dtypes import DType

_FUNCTION = """def summarize_orders(orders, currency="EUR"):
    totals = {}
    for order in orders:
        if order["status"] == "cancelled":
            continue
        customer = order["customer"]
        amount = order["amount"]
        if order["currency"] != currency:
            amount = amount * order["rate"]
        totals[customer] = totals.get(customer, 0) + amount
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    lines = []
    for customer, total in ranked:
        lines.append(f"{customer}: {total:.2f} {currency}")
    return "\\n".join(lines)
"""

_TOOLS = """Tools:
read_file(path: string, start_line: integer, end_line: integer) -> text
search_code(query: string, path_glob: string, max_results: integer) -> list of matches
apply_patch(path: string, patch: string) -> result"""


_DOCUMENT = """Incident report INC-2291, payments service, 14 March.
At 09:12 UTC the refund worker began rejecting refunds for orders paid in a currency other than EUR.
The rejection message was "rate missing for currency", logged 4,318 times between 09:12 and 10:47.
The cause was a configuration change, deployed at 09:05, that removed the exchange-rate table from
the worker's cache. Refunds in EUR were not affected. The on-call engineer restored the table at
10:41 and the worker drained its queue by 10:47. Customers whose refund was rejected were refunded
again automatically by 11:30; no customer was refunded twice. Follow-up actions: add a start-up
check that the exchange-rate table is present, and alert when more than 50 refunds fail in five
minutes. The report was reviewed by the payments team on 16 March and closed on 18 March."""


def _random_words(seed: int) -> str:
    rng = random.Random(seed)  # noqa: S311  # reproducible prompt text, not a secret
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = ["".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(60)]
    return " ".join(words)


def _messages(prompt_class: str, ticket: int, of: int) -> list[dict[str, str]]:
    head = f"Ticket {ticket + 1} of {of}.\n"
    if prompt_class == "code_edit":
        text = (
            f"{head}Rewrite this function in full, with the same logic, but skip orders whose "
            f'status is "refunded" as well as "cancelled". Answer with the code only.\n'
            f"```python\n{_FUNCTION}```"
        )
    elif prompt_class == "rag_extract":
        text = (
            f"{head}Answer from the retrieved document only. First quote, word for word, every "
            "sentence that gives a time of day. Then list the cause, the fix and the follow-up "
            f"actions as bullet points.\n\nRetrieved document:\n{_DOCUMENT}"
        )
    elif prompt_class in ("creative_chat", "prose"):
        text = f"{head}Write a short story about a lighthouse keeper who finds a glass compass."
    elif prompt_class == "json_tools":
        text = (
            f"{head}{_TOOLS}\nAnswer with one JSON object per line, each a tool call "
            '{"name": ..., "arguments": {...}}: read lines 1 to 80 of src/orders.py, search for '
            '"summarize_orders" in "**/*.py" with at most 20 results, then read lines 80 to 160 '
            "of src/orders.py."
        )
    elif prompt_class == "random":
        text = f"{head}Sort these words alphabetically and explain any pattern:\n" + _random_words(
            ticket
        )
    else:
        raise ValueError(f"unknown prompt class {prompt_class!r}")
    return [{"role": "user", "content": text}]


def _context(tokenizer: Any, tokens: int) -> str:
    """About ``tokens`` tokens of numbered copies of the function, as files of a repository."""
    files: list[str] = []
    while len(tokenizer.encode("\n".join(files))) < tokens:
        index = len(files)
        files.append(f"# src/orders_{index}.py\n" + _FUNCTION.replace("orders", f"orders_{index}"))
    return "Repository files:\n" + "\n".join(files)


def _prompts(
    snapshot: Path, prompt_class: str, count: int, context_tokens: int
) -> list[tuple[int, ...]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    system = (
        [{"role": "system", "content": _context(tokenizer, context_tokens)}]
        if context_tokens
        else []
    )
    prompts: list[tuple[int, ...]] = []
    for ticket in range(count):
        ids = tokenizer.apply_chat_template(
            system + _messages(prompt_class, ticket, count),
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3's template; other templates ignore it
            tokenize=True,
        )
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):  # transformers 5 returns an encoding
            ids = ids["input_ids"]
        prompts.append(tuple(int(token) for token in ids))
    return prompts


# Field annotations are strings under ``from __future__ import annotations``; the rest are text.
_CONVERT: dict[str, Any] = {"int": int, "float": float, "bool": lambda raw: raw == "true"}


def _parse_arm(text: str) -> tuple[str, SpeculativeConfig]:
    """``name=field:value,field:value`` into a speculative config with ``enabled=True``."""
    name, _, spec = text.partition("=")
    types = {f.name: f.type for f in dataclasses.fields(SpeculativeConfig)}
    values: dict[str, Any] = {"enabled": True}
    for item in filter(None, spec.split(",")):
        key, _, raw = item.partition(":")
        convert = _CONVERT.get(str(types[key]), str)
        values[key] = convert(raw)
    return name, SpeculativeConfig(**values)


def _engine(args: argparse.Namespace, snapshot: Path, speculative: SpeculativeConfig) -> LLM:
    return LLM.from_config(
        EngineConfig(
            source=ModelSourceConfig(args.model, offline=True),
            tokenizer=TokenizerConfig(
                tokenizer=str(snapshot), offline=True, skip_tokenizer_init=True
            ),
            device=DeviceConfig(
                device="cuda",
                dtype=DType.BF16,
                backends=("flashinfer", "torch"),
                cuda_graph=True,
            ),
            cache=CacheConfig(
                block_size=16,
                num_blocks=args.num_blocks,
                kv_reserve_tokens=args.kv_reserve,
                prefix_cache=args.prefix_cache,
            ),
            scheduler=SchedulerConfig(max_running=max(args.concurrency), max_tokens_per_step=512),
            speculative=speculative,
        )
    )


def _run(llm: LLM, prompts: list[tuple[int, ...]], max_tokens: int) -> dict[str, Any]:
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    controller = llm.engine.runner.controller
    if not isinstance(controller, (CostAwareController, FixedController)):
        controller = None
    before = controller.snapshot() if controller is not None else None
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, params)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if not isinstance(outputs, list):
        raise TypeError("generate with a list of prompts returns a list")
    tokens = [tuple(output.outputs[0].token_ids) for output in outputs]
    after = controller.snapshot() if controller is not None else None
    run: dict[str, Any] = {
        "seconds": round(seconds, 4),
        "tokens": sum(len(ids) for ids in tokens),
        "tok_per_sec": round(sum(len(ids) for ids in tokens) / seconds, 2),
        "ids": tokens,
    }
    if before is not None and after is not None:
        for key in ("rounds", "total_drafts", "total_accepted"):
            run[key] = int(after[key]) - int(before[key])
    return run


def _first_difference(got: tuple[int, ...], want: tuple[int, ...]) -> int | None:
    return next((i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Speculative against plain decoding, in pairs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", action="append", required=True, help="name=field:value,...")
    parser.add_argument(
        "--classes",
        default="code_edit,rag_extract,creative_chat,json_tools",
        help="comma-separated: code_edit, rag_extract, creative_chat (prose), json_tools, random",
    )
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--kv-reserve", type=int, default=8192)
    parser.add_argument("--context-tokens", type=int, default=0)
    parser.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.concurrency = [int(c) for c in args.concurrency.split(",")]
    classes = args.classes.split(",")
    arms = dict(_parse_arm(text) for text in args.arm)

    snapshot = resolve_source(ModelSourceConfig(args.model, offline=True))
    prompts = {
        name: _prompts(snapshot, name, max(args.concurrency), args.context_tokens)
        for name in classes
    }
    report: dict[str, Any] = {
        "machine": machine(),
        "model": args.model,
        "protocol": {
            "pairs": args.pairs,
            "order": "pair p: fresh engines, plain first when p is even, the arm first when odd",
            "context_tokens": args.context_tokens,
            "prefix_cache": args.prefix_cache,
            "max_tokens": args.max_tokens,
            "greedy": True,
            "dtype": "bf16",
            "backends": ["flashinfer", "torch"],
            "cuda_graph": True,
            "num_blocks": args.num_blocks,
            "prompt_tokens": {name: [len(p) for p in prompts[name]] for name in classes},
        },
        "arms": {name: dataclasses.asdict(config) for name, config in arms.items()},
        "results": {},
    }
    cells = [(c, n) for c in classes for n in args.concurrency]
    for arm_name, config in arms.items():
        # runs[which][cell] is one list of runs, one per pair
        runs: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = {"plain": {}, "arm": {}}
        orders: list[list[str]] = []
        for pair in range(args.pairs):
            order = ["plain", "arm"] if pair % 2 == 0 else ["arm", "plain"]
            orders.append(order)
            for which in order:
                with _engine(
                    args, snapshot, config if which == "arm" else SpeculativeConfig()
                ) as llm:
                    _run(llm, prompts[classes[0]][:1], 16)  # warm-up, not kept
                    for prompt_class, concurrency in cells:
                        batch = prompts[prompt_class][:concurrency]
                        runs[which].setdefault((prompt_class, concurrency), []).append(
                            _run(llm, batch, args.max_tokens)
                        )
                torch.cuda.empty_cache()
        report["results"][arm_name] = {
            f"{c}/c{n}": _cell(runs["plain"][(c, n)], runs["arm"][(c, n)], orders, arm_name, c, n)
            for c, n in cells
        }
        write(report, args.out)  # after every arm, so a crash keeps what finished


def _cell(
    plain: list[dict[str, Any]],
    arm: list[dict[str, Any]],
    orders: list[list[str]],
    arm_name: str,
    prompt_class: str,
    concurrency: int,
) -> dict[str, Any]:
    """One prompt class at one concurrency: the pairs, their ratios and the noise of plain."""
    pairs: list[dict[str, Any]] = []
    for order, base, run in zip(orders, plain, arm, strict=True):
        differing = [
            {"request": index, "first_difference": _first_difference(got, want)}
            for index, (got, want) in enumerate(zip(run["ids"], base["ids"], strict=True))
            if got != want
        ]
        pairs.append(
            {
                "order": order,
                "plain_tok_per_sec": base["tok_per_sec"],
                "arm_tok_per_sec": run["tok_per_sec"],
                "ratio": round(run["tok_per_sec"] / base["tok_per_sec"], 4),
                "drafted": run.get("total_drafts", 0),
                "accepted": run.get("total_accepted", 0),
                "rounds": run.get("rounds", 0),
                "requests_differing_from_plain": differing,
            }
        )
    ratios = [p["ratio"] for p in pairs]
    plain_speeds = [p["plain_tok_per_sec"] for p in pairs]
    arm_speeds = [p["arm_tok_per_sec"] for p in pairs]
    median_plain = statistics.median(plain_speeds)
    drafted = sum(p["drafted"] for p in pairs)
    accepted = sum(p["accepted"] for p in pairs)
    cell = {
        "pairs": pairs,
        # The ratio of the medians, as #127 defines it, and the spread of the pairs' ratios.
        "ratio": round(statistics.median(arm_speeds) / median_plain, 4),
        "worst_pair": min(ratios),
        "best_pair": max(ratios),
        # noise = max(0.02, 2 * (max - min) / median) over the plain runs (#127)
        "noise": round(max(0.02, 2 * (max(plain_speeds) - min(plain_speeds)) / median_plain), 4),
        "median_plain_tok_per_sec": median_plain,
        "median_arm_tok_per_sec": statistics.median(arm_speeds),
        "acceptance_rate": round(accepted / drafted, 4) if drafted else 0.0,
        "accepted_per_round": round(accepted / max(sum(p["rounds"] for p in pairs), 1), 3),
        "requests_differing_from_plain": sum(
            len(p["requests_differing_from_plain"]) for p in pairs
        ),
        "requests_compared": concurrency * len(pairs),
    }
    print(
        f"{arm_name} {prompt_class} c={concurrency}: {cell['ratio']}x (worst pair "
        f"{cell['worst_pair']}, noise {cell['noise']}), acceptance {cell['acceptance_rate']}, "
        f"differing {cell['requests_differing_from_plain']}/{cell['requests_compared']}",
        flush=True,
    )
    return cell


if __name__ == "__main__":
    main()
