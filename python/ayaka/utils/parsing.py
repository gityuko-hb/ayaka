"""Parsing and formatting helpers for operator-facing strings.

Sizes accept SI suffixes (``k``, ``M``, ``G``, ``T``; powers of 1000) and IEC
suffixes (``Ki``, ``Mi``, ``Gi``, ``Ti``; powers of 1024). Suffixes are
case-sensitive, and decimal points are only accepted with SI suffixes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal
from typing import Any, Final

__all__ = [
    "format_human_int",
    "parse_bool",
    "parse_human_int",
    "parse_json_strict",
]

_TRUE: Final = frozenset({"1", "true", "yes", "y", "on"})
_FALSE: Final = frozenset({"0", "false", "no", "n", "off"})

_SI: Final = {"k": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}
_IEC: Final = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}

# Suffix alternatives are ordered longest-first so "Ki" wins over "K".
_SIZE_RE: Final = re.compile(r"(\d+(?:\.\d+)?)\s*(Ki|Mi|Gi|Ti|k|M|G|T)?")


def parse_bool(value: str, *, what: str = "value") -> bool:
    """Parse a string into a boolean.

    Raises `ValueError` with an actionable message on anything else.

    Examples:
        >>> parse_bool("true")
        True
        >>> parse_bool("no")
        False
    """
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(
        f"{what}: cannot parse {value!r} as a boolean. "
        f"Use one of {sorted(_TRUE)} or {sorted(_FALSE)}."
    )


def parse_human_int(value: str, *, what: str = "value") -> int:
    """Parse ``'80Gi'`` / ``'25.6k'`` / ``'4096'`` into an int.

    See the module docstring for the SI/IEC contract. Raises `ValueError`
    with an actionable message on anything else.

    Examples:
        >>> parse_human_int("1k")
        1000
        >>> parse_human_int("1Ki")
        1024
        >>> parse_human_int("25.6k")
        25600
        >>> parse_human_int("4096")
        4096
    """
    text = value.strip()
    if not text:
        raise ValueError(f"{what}: empty string is not an integer")

    match = _SIZE_RE.fullmatch(text)
    if match is None:
        raise ValueError(
            f"{what}: cannot parse {value!r} as an integer. Use a plain "
            f"integer, an SI suffix (1k, 1M, 1G, 1T = powers of 1000), or "
            f"an IEC suffix (1Ki, 1Mi, 1Gi, 1Ti = powers of 1024). "
            f"Suffixes are case-sensitive."
        )

    number, suffix = match.groups()

    if suffix is None:
        if "." in number:
            raise ValueError(
                f"{what}: {value!r} is not an integer. Drop the decimal "
                f"point, or use an SI suffix such as {number}k."
            )
        return int(number)

    if suffix in _IEC:
        if "." in number:
            raise ValueError(
                f"{what}: decimals are not allowed with the IEC suffix "
                f"{suffix!r} because the result would be truncated "
                f"silently. Use an integer such as "
                f"{int(Decimal(number))}{suffix}, or the SI form "
                f"{number}{suffix[0]}."
            )
        return int(number) * _IEC[suffix]

    # Decimal arithmetic, not float: int(0.1 * 10**9) is 99999999 under
    # binary floating point.
    return int(Decimal(number) * _SI[suffix])


def format_human_int(value: int, *, binary: bool = True) -> str:
    """Render an int with an exact SI/IEC suffix when one applies.

    A suffix is emitted only when the value is an exact multiple of the unit,
    so the result always round-trips through `parse_human_int`. Used in log
    lines and error messages, never on a hot path.

    Examples:
        >>> format_human_int(1000)
        '1000'
        >>> format_human_int(1000, binary=False)
        '1k'
        >>> format_human_int(1024)
        '1Ki'
        >>> format_human_int(2_000_000, binary=False)
        '2M'
    """
    table = _IEC if binary else _SI
    for suffix, mult in sorted(table.items(), key=lambda kv: -kv[1]):
        if mult <= abs(value) and value % mult == 0:
            return f"{value // mult}{suffix}"
    return str(value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    counts = Counter(key for key, _ in pairs)
    duplicated = sorted(key for key, count in counts.items() if count > 1)
    if duplicated:
        raise ValueError(f"duplicate object keys: {duplicated}")
    return dict(pairs)


def parse_json_strict(text: str) -> object:
    """Parse JSON from an untrusted source, refusing what ``json.loads`` forgives.

    ``json.loads`` keeps the last value of a repeated key, so a document that
    says ``{"w": 1, "w": 2}`` parses cleanly and one of its entries silently
    disappears. This parser refuses a repeated key in any object, at any depth.

    Raises:
        ValueError: If the text is not valid JSON, repeats an object key, holds
            an integer too long to convert, or nests deeper than the parser can
            follow. The message says which.

    Examples:
        >>> parse_json_strict('{"a": [1, {"b": 2}]}')
        {'a': [1, {'b': 2}]}
        >>> parse_json_strict('{"a": 1, "a": 2}')
        Traceback (most recent call last):
        ...
        ValueError: duplicate object keys: ['a']
    """
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except RecursionError:
        raise ValueError("JSON is nested too deeply") from None
