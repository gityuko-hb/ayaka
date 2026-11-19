"""The server reaches the engine only through ``EngineClient``.

Nothing under ``python/ayaka/entrypoints`` may read an ``engine`` attribute (``llm.engine.kv``,
``llm.engine.tokenizer``) or a ``recorder`` attribute (``core.recorder = ...``): what the server
needs to know about the engine is ``EngineInfo``, the tokenizer is passed to it, and the recorder
is a parameter of the build. A name that is only a variable or a parameter
(``engine: EngineConfig``) is not an attribute and is fine.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = ROOT / "python" / "ayaka" / "entrypoints"

#: Attributes that are the engine's insides.
_INSIDES = frozenset({"engine", "recorder"})


def _reaches(source: str) -> list[tuple[int, str]]:
    """The ``(line, attribute)`` of every access to an attribute that is the engine's insides."""
    return sorted(
        (node.lineno, node.attr)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute) and node.attr in _INSIDES
    )


def test_nothing_in_entrypoints_reaches_into_the_engine() -> None:
    sources = sorted(ENTRYPOINTS.rglob("*.py"))
    assert sources, f"no module found under {ENTRYPOINTS}"
    violations = [
        f"{path.relative_to(ROOT)}:{line} reads .{attribute}"
        for path in sources
        for line, attribute in _reaches(path.read_text(encoding="utf-8"))
    ]

    assert violations == [], "\n".join(
        [*violations, "Read the engine through EngineClient and EngineInfo (ayaka.engine.seam)."]
    )


def test_the_scan_flags_a_read_of_the_engine_and_of_the_recorder() -> None:
    source = (
        "tokenizer = llm.engine.tokenizer\n"
        "limit = serving.llm.engine.kv.max_context_tokens\n"
        "core.recorder = None\n"
    )

    assert _reaches(source) == [(1, "engine"), (2, "engine"), (3, "recorder")]


def test_the_scan_leaves_names_alone() -> None:
    source = (
        "def serve(engine: EngineConfig, recorder: SessionRecorder | None) -> None:\n"
        "    build = engine if callable(engine) else None\n"
        "    llm.health()\n"
    )

    assert _reaches(source) == []
