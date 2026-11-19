"""The prompt set of ``ayaka bench``: three groups of fixed text, cut to fixed token lengths.

A benchmark result is only comparable with another one when both read the same prompts, so the
text here is generated, not collected: a seeded generator writes Python modules (the ``edit`` and
``write`` groups) and prose (the ``prose`` group) from fixed word lists. The same
:data:`PROMPT_SET_VERSION`, group and tokenizer always give the same token ids; changing a word
list, a template or the generator is a new version, and results of two versions are not compared.

Each prompt is a head (what the text is), the material cut to the length, and the instruction at
the end, as a coding tool sends it: the context first, the request last. The ids are built by
concatenating the three parts' ids, so a prompt has exactly the length asked for.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "GROUPS",
    "PROMPT_SET_VERSION",
    "BenchPrompt",
    "build_prompt",
    "material_ids",
    "material_text",
    "prompt_digest",
]

#: Bumped whenever a word list, a template or the generator changes.
PROMPT_SET_VERSION: Final = 1

#: The groups: edit existing code, write new code, prose.
GROUPS: Final = ("edit", "write", "prose")

_SEEDS: Final = {"edit": 11, "write": 23, "prose": 37}

_HEADS: Final = {
    "edit": "Here is a Python module from the project, as it is now:\n\n",
    "write": "Here are the modules the new code has to fit with:\n\n",
    "prose": "Here is the report the team wrote last quarter:\n\n",
}

_INSTRUCTIONS: Final = {
    "edit": "\n\nRename `compute_total` to `sum_amounts` everywhere in this module, keep the "
    "behaviour, and show the changed functions only.",
    "write": "\n\nWrite a new function `merge_reports(left, right)` in the same style that "
    "combines two report dictionaries, with a docstring and two tests.",
    "prose": "\n\nSummarise the report in five sentences for a reader who has not seen it.",
}

_NOUNS: Final = (
    "amount",
    "record",
    "buffer",
    "account",
    "invoice",
    "batch",
    "entry",
    "report",
    "ledger",
    "item",
    "order",
    "window",
    "segment",
    "table",
    "column",
    "index",
    "budget",
    "shard",
)
_VERBS: Final = (
    "compute",
    "merge",
    "split",
    "load",
    "store",
    "check",
    "update",
    "parse",
    "render",
    "count",
    "filter",
    "sort",
    "flush",
    "scale",
    "round",
    "encode",
    "decode",
    "collect",
)
_WORDS: Final = (
    "the",
    "team",
    "measured",
    "a",
    "steady",
    "growth",
    "in",
    "requests",
    "while",
    "costs",
    "stayed",
    "flat",
    "and",
    "every",
    "region",
    "reported",
    "fewer",
    "errors",
    "than",
    "before",
    "after",
    "the",
    "migration",
    "most",
    "customers",
    "noticed",
    "faster",
    "pages",
    "but",
    "some",
    "asked",
    "for",
    "clearer",
    "billing",
    "notes",
    "which",
    "the",
    "support",
    "group",
    "wrote",
)


def _code_unit(rng: random.Random, number: int) -> str:
    verb, noun = rng.choice(_VERBS), rng.choice(_NOUNS)
    other = rng.choice(_NOUNS)
    name = "compute_total" if number % 7 == 0 else f"{verb}_{noun}_{number}"
    lines = [
        f"def {name}({noun}s, {other}_limit={rng.randint(2, 99)}):",
        f'    """{verb.capitalize()} the {noun}s, at most {other}_limit of them."""',
        "    result = []",
        f"    for position, {noun} in enumerate({noun}s):",
        f"        if position >= {other}_limit:",
        "            break",
        f"        value = {noun} * {rng.randint(2, 9)} + {rng.randint(0, 99)}",
        f"        if value % {rng.randint(2, 7)} == 0:",
        "            result.append(value)",
        "    return result",
        "",
        "",
    ]
    return "\n".join(lines)


def _prose_unit(rng: random.Random, number: int) -> str:
    sentences: list[str] = []
    for _ in range(rng.randint(3, 6)):
        words = [rng.choice(_WORDS) for _ in range(rng.randint(8, 18))]
        sentences.append(" ".join(words).capitalize() + ".")
    return f"Section {number}. " + " ".join(sentences) + "\n\n"


def material_text(group: str, units: int) -> str:
    """The first ``units`` units of a group's material (functions or report sections)."""
    rng = random.Random(_SEEDS[group])  # noqa: S311  # fixed text, not a secret
    unit = _prose_unit if group == "prose" else _code_unit
    return "".join(unit(rng, number) for number in range(units))


@dataclass(frozen=True, slots=True)
class BenchPrompt:
    """One prompt of the set: its group, its length and its token ids."""

    group: str
    length: int
    token_ids: tuple[int, ...]


def build_prompt(group: str, length: int, encode: Callable[[str], list[int]]) -> BenchPrompt:
    """The group's prompt of exactly ``length`` tokens with this tokenizer.

    Args:
        group: One of :data:`GROUPS`.
        length: Tokens in the prompt.
        encode: The model's tokenizer, without special tokens.

    Raises:
        ValueError: If the group is unknown or the length cannot hold the head and instruction.
    """
    if group not in GROUPS:
        raise ValueError(f"unknown prompt group {group!r}; the groups are {GROUPS}")
    head = encode(_HEADS[group])
    tail = encode(_INSTRUCTIONS[group])
    room = length - len(head) - len(tail)
    if room < 1:
        raise ValueError(f"{length} tokens cannot hold the {group} prompt's head and instruction")
    return BenchPrompt(group, length, tuple(head + material_ids(group, room, encode) + tail))


def material_ids(group: str, tokens: int, encode: Callable[[str], list[int]]) -> list[int]:
    """The first ``tokens`` token ids of a group's material."""
    body: list[int] = []
    units = 64
    while len(body) < tokens:
        body = encode(material_text(group, units))
        units *= 2
    return body[:tokens]


def prompt_digest(prompt: BenchPrompt) -> str:
    """A short digest of the prompt's ids, so two results can show they read the same prompt."""
    data = ",".join(map(str, prompt.token_ids)).encode()
    return hashlib.sha256(data).hexdigest()[:16]
