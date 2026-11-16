"""A package imports only packages from a lower layer.

``LAYERS`` is the layer table of ``docs/TREE.md`` section 3. A package may import a
package in a strictly lower layer and its own submodules, nothing else. ``core`` and
``utils`` share the foundation layer and may import each other. An import counts
wherever it appears: at module level, inside a function or under ``TYPE_CHECKING``,
because a type-only dependency is still a dependency of the layer.

Packages the table lists but that are not on disk yet are tolerated; a package that is
on disk but missing from the table fails.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "python" / "ayaka"

_LAYER_PACKAGES: tuple[tuple[str, ...], ...] = (
    ("core", "utils"),
    ("configs", "metrics"),
    ("ops", "device", "memory", "tokenizers", "multimodal", "telemetry"),
    ("weights", "sampling", "adapters"),
    ("layers", "kvcache", "moe"),
    ("models", "scheduler", "speculative"),
    ("execution", "model_loader", "session", "distributed"),
    ("engine",),
    ("entrypoints",),
)
LAYERS: dict[str, int] = {
    package: layer for layer, packages in enumerate(_LAYER_PACKAGES) for package in packages
}
_FOUNDATION = frozenset(_LAYER_PACKAGES[0])


@dataclass(frozen=True, slots=True)
class LayerViolation:
    line: int
    importer: str
    target: str
    reason: str

    def describe(self, path: str) -> str:
        return f"{path}:{self.line} {self.importer} imports {self.target}: {self.reason}"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT / "python").with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(module: str) -> str | None:
    """The package under ``ayaka`` that owns ``module``; ``None`` for ``ayaka`` itself."""

    parts = module.split(".")
    if parts[0] != "ayaka" or len(parts) < 2:
        return None
    return parts[1]


def _label(package: str) -> str:
    layer = LAYERS.get(package)
    return f"{package}, L{layer}" if layer is not None else f"{package}, not in the layer table"


def _absolute(node: ast.ImportFrom, module: str, *, is_package: bool) -> str | None:
    if not node.level:
        return node.module
    parts = module.split(".")
    anchor = parts if is_package else parts[:-1]
    keep = len(anchor) - (node.level - 1)
    if keep < 1:
        return None
    base = ".".join(anchor[:keep])
    return f"{base}.{node.module}" if node.module else base


def _imported_modules(
    tree: ast.Module, module: str, *, is_package: bool
) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node, module, is_package=is_package)
            if base is None:
                continue
            yield node.lineno, base
            if base == "ayaka":
                # ``from ayaka import ops`` names a package, not an attribute of ``ayaka``.
                for alias in node.names:
                    yield node.lineno, f"ayaka.{alias.name}"


def find_violations(source: str, module: str, *, is_package: bool = False) -> list[LayerViolation]:
    """Imports in ``source`` (the code of ``module``) that reach the same or a higher layer."""

    owner = _package_of(module)
    if owner is None:
        return []
    owner_layer = LAYERS.get(owner)
    violations: list[LayerViolation] = []
    for line, imported in _imported_modules(ast.parse(source), module, is_package=is_package):
        target = _package_of(imported)
        if target is None or target == owner or target not in LAYERS:
            continue
        if owner_layer is None:
            continue
        target_layer = LAYERS[target]
        if target_layer < owner_layer or (target_layer == owner_layer and target in _FOUNDATION):
            continue
        relation = "a higher layer" if target_layer > owner_layer else "the same layer"
        violations.append(
            LayerViolation(
                line=line,
                importer=f"{module} ({_label(owner)})",
                target=f"{imported} ({_label(target)})",
                reason=(
                    f"{owner} may only import packages from a lower layer, "
                    f"and {target} is in {relation}"
                ),
            )
        )
    return violations


_SOURCES = sorted(PACKAGE_ROOT.rglob("*.py"))
_ON_DISK = sorted(
    {path.relative_to(PACKAGE_ROOT).parts[0] for path in _SOURCES if path.parent != PACKAGE_ROOT}
)


def test_layer_table_names_each_package_once() -> None:
    flat = [package for packages in _LAYER_PACKAGES for package in packages]

    assert len(flat) == len(set(flat))
    assert set(flat) == set(LAYERS)


def test_scan_sees_the_package() -> None:
    assert _SOURCES, f"no module found under {PACKAGE_ROOT}"
    assert {"core", "utils"} <= set(_ON_DISK)


def test_every_package_on_disk_is_in_the_layer_table() -> None:
    unknown = sorted(set(_ON_DISK) - set(LAYERS))

    assert unknown == [], (
        f"ayaka packages missing from the layer table in tests/test_layers.py: {unknown}. "
        "Add them to the table in docs/TREE.md section 3 and to this test in the same change."
    )


@pytest.mark.parametrize("package", _ON_DISK, ids=lambda package: package)
def test_package_imports_only_lower_layers(package: str) -> None:
    violations: list[str] = []
    for path in _SOURCES:
        module = _module_name(path)
        if _package_of(module) != package:
            continue
        found = find_violations(
            path.read_text(encoding="utf-8"), module, is_package=path.name == "__init__.py"
        )
        violations.extend(item.describe(str(path.relative_to(ROOT))) for item in found)

    assert violations == [], "\n".join(violations)


def test_scanner_flags_a_module_level_import_from_a_higher_layer() -> None:
    source = "from ayaka.engine.core import Engine\n"

    [violation] = find_violations(source, "ayaka.configs.base")

    assert violation.line == 1
    assert "ayaka.configs.base (configs, L1)" in violation.importer
    assert "ayaka.engine.core (engine, L7)" in violation.target
    assert "higher layer" in violation.reason


def test_scanner_flags_a_plain_import_statement() -> None:
    [violation] = find_violations("import ayaka.models.gpt2\n", "ayaka.ops.backend")

    assert "ayaka.models.gpt2 (models, L5)" in violation.target


def test_scanner_flags_an_import_inside_a_function() -> None:
    source = "def load() -> None:\n    from ayaka.scheduler.step import StepInput\n"

    [violation] = find_violations(source, "ayaka.layers.attention")

    assert violation.line == 2
    assert "higher layer" in violation.reason


def test_scanner_flags_an_import_under_type_checking() -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from ayaka.scheduler.step import StepInput\n"
    )

    [violation] = find_violations(source, "ayaka.weights.mapping")

    assert violation.line == 3


def test_scanner_flags_a_package_named_by_from_ayaka_import() -> None:
    [violation] = find_violations("from ayaka import engine\n", "ayaka.utils.cancellation")

    assert "ayaka.engine (engine, L7)" in violation.target


def test_scanner_resolves_a_relative_import_that_leaves_the_package() -> None:
    source = "from ..engine import Engine\n"

    [violation] = find_violations(source, "ayaka.core.outputs")

    assert "ayaka.engine (engine, L7)" in violation.target


def test_scanner_resolves_a_relative_import_from_a_package_init() -> None:
    source = "from ...scheduler import Scheduler\n"

    [violation] = find_violations(source, "ayaka.kvcache.storage", is_package=True)

    assert "ayaka.scheduler (scheduler, L5)" in violation.target


def test_scanner_flags_a_peer_package_in_the_same_layer() -> None:
    [violation] = find_violations(
        "from ayaka.metrics.stats import IterationStats\n", "ayaka.configs.base"
    )

    assert "same layer" in violation.reason


def test_scanner_allows_lower_layers_and_the_foundation_pair() -> None:
    assert (
        find_violations("from ayaka.core.errors import ConfigError\n", "ayaka.utils.validation")
        == []
    )
    assert (
        find_violations("from ayaka.utils.validation import require_int\n", "ayaka.core.params")
        == []
    )
    assert (
        find_violations(
            "from ayaka.configs.tokenizer import TokenizerConfig\n", "ayaka.tokenizers.ports"
        )
        == []
    )


def test_scanner_allows_own_package_relative_imports_and_the_root_module() -> None:
    assert find_violations("from .errors import ConfigError\n", "ayaka.core.outputs") == []
    assert find_violations("from ayaka.core import errors\n", "ayaka.core.outputs") == []
    assert find_violations("from .__version__ import __version__\n", "ayaka", is_package=True) == []


def test_scanner_ignores_third_party_and_unlisted_packages() -> None:
    source = (
        "import torch\n"
        "from ayaka.future_package import thing\n"
        "from ayaka.__version__ import __version__\n"
    )

    assert find_violations(source, "ayaka.core.errors") == []


def test_violation_message_names_file_line_and_both_layers() -> None:
    [violation] = find_violations("\n\nfrom ayaka.engine import Engine\n", "ayaka.ops.backend")

    message = violation.describe("python/ayaka/ops/backend.py")

    assert message.startswith("python/ayaka/ops/backend.py:3 ")
    assert "ops, L2" in message
    assert "engine, L7" in message
