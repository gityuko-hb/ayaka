"""Every public name lives where it is defined, in the package and in the tests.

``__all__`` lists what a module defines, never what it imports. A package facade
may export a name only through the submodule that defines it, and every import,
including a test's, names that module. This keeps one home per name, so a failure
points at the real owner instead of at a re-export.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "python" / "ayaka"
TESTS_ROOT = ROOT / "tests"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT / "python").with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _top_level_names(tree: ast.Module) -> set[str]:
    """Names a module defines itself, including PEP 695 type aliases."""

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.TypeAlias):
            names.add(node.name.id)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _literal_assignment(tree: ast.Module, target: str) -> object | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(name, ast.Name) and name.id == target for name in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


def _export_table(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    raw = _literal_assignment(tree, "_EXPORTS")
    if not isinstance(raw, dict):
        return {}
    return {str(module): tuple(str(name) for name in names) for module, names in raw.items()}


def _is_submodule(module: str, name: str) -> bool:
    base = ROOT / "python" / Path(*module.split("."))
    return (base / f"{name}.py").is_file() or (base / name / "__init__.py").is_file()


_SOURCES = sorted(PACKAGE_ROOT.rglob("*.py"))
_MODULES = {_module_name(path): path for path in _SOURCES}
_TREES = {module: ast.parse(path.read_text(encoding="utf-8")) for module, path in _MODULES.items()}
_DEFINED = {module: _top_level_names(tree) for module, tree in _TREES.items()}
_PACKAGES = {module for module, path in _MODULES.items() if path.name == "__init__.py"}
_FACADE_NAMES = {
    module: {name for names in _export_table(_TREES[module]).values() for name in names}
    for module in _PACKAGES
}
_HOMES: dict[str, list[str]] = {}
for _module, _names in _DEFINED.items():
    for _name in _names:
        _HOMES.setdefault(_name, []).append(_module)


@pytest.mark.parametrize(
    "path",
    [path for path in _SOURCES if path.name != "__init__.py"],
    ids=_module_name,
)
def test_module_all_defines_every_name(path: Path) -> None:
    module = _module_name(path)
    declared = _literal_assignment(_TREES[module], "__all__")
    if declared is None:
        return
    assert isinstance(declared, list)

    imported = sorted(set(declared) - _DEFINED[module])

    assert imported == [], (
        f"{module}: __all__ lists names it does not define: {imported}. "
        "A name lives in the module that defines it; do not re-export it."
    )


@pytest.mark.parametrize(
    "package",
    sorted(_PACKAGES),
    ids=lambda package: package,
)
def test_facade_exports_only_names_its_submodule_defines(package: str) -> None:
    for submodule, names in _export_table(_TREES[package]).items():
        owner = f"{package}.{submodule}"

        assert owner in _DEFINED, f"{package}: _EXPORTS points at unknown module {owner!r}"
        missing = sorted(set(names) - _DEFINED[owner])

        assert missing == [], (
            f"{package}: _EXPORTS[{submodule!r}] lists names that {owner} does not define: "
            f"{missing}. Export a name through the submodule that defines it."
        )


def _import_sites() -> Iterator[tuple[Path, int, str, str]]:
    """Absolute ``ayaka`` imports; relative imports cannot cross packages."""

    sources = [*_SOURCES, *sorted(TESTS_ROOT.rglob("*.py"))]
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if not node.module.startswith("ayaka"):
                continue
            for alias in node.names:
                if alias.name != "*":
                    yield path, node.lineno, node.module, alias.name


def test_imports_name_the_defining_module() -> None:
    violations: list[str] = []
    for path, line, module, name in _import_sites():
        if name in _DEFINED.get(module, set()):
            continue
        if _is_submodule(module, name):
            continue
        if name in _FACADE_NAMES.get(module, set()):
            continue
        homes = ", ".join(sorted(_HOMES.get(name, ()))) or "no module in python/ayaka"
        violations.append(
            f"{path.relative_to(ROOT)}:{line} imports {name} from {module}; "
            f"it is defined in {homes}"
        )

    assert violations == [], "\n".join(violations)
