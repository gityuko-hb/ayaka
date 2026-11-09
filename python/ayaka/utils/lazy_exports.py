"""Lazy package-level exports (PEP 562).

A package ``__init__`` declares a literal ``__all__`` and a
``{submodule: (names, ...)}`` table, then calls :func:`build_lazy_exports`::

    from ayaka.utils.lazy_exports import build_lazy_exports

    __all__ = ["Histogram", "IterationStats"]
    _EXPORTS = {
        "histogram": ("Histogram",),
        "stats": ("IterationStats",),
    }
    __getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())

``from package import Name`` imports only the submodule that defines ``Name``,
so declaring a public API never forces an eager import of the whole package
(and never creates an import cycle). The literal ``__all__`` keeps linters and
type checkers able to see the public names; the helper fails at import time if
it and the table disagree.

Add a ``TYPE_CHECKING`` block in the package to give type checkers the real
types of the exported names; without it they are seen as ``Any``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["build_lazy_exports"]


def build_lazy_exports(
    package: str,
    exports: Mapping[str, tuple[str, ...]],
    namespace: dict[str, Any],
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build ``__getattr__`` and ``__dir__`` for a package ``__init__``.

    Args:
        package: Dotted package name (``__name__`` of the caller).
        exports: Relative submodule name to the public names it defines.
        namespace: The package ``globals()``; resolved names are cached there and
            its literal ``__all__`` is checked against ``exports``.

    Returns:
        A ``(__getattr__, __dir__)`` pair to assign at module scope in the
        package ``__init__``.

    Raises:
        ValueError: If one public name is declared by two submodules, or the
            package ``__all__`` does not list exactly the exported names.
    """

    owner: dict[str, str] = {}
    for module, names in exports.items():
        for name in names:
            if name in owner:
                raise ValueError(f"{package}: {name!r} exported by {owner[name]} and {module}")
            owner[name] = module

    declared = set(namespace.get("__all__", ()))
    if declared != set(owner):
        missing = sorted(set(owner) - declared)
        extra = sorted(declared - set(owner))
        raise ValueError(
            f"{package}: __all__ and the export table differ "
            f"(missing from __all__: {missing}; not in the export table: {extra})"
        )

    def resolve(name: str) -> Any:
        """Import the owning submodule on first access and cache the value."""
        module = owner.get(name)
        if module is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(import_module(f"{package}.{module}"), name)
        namespace[name] = value
        return value

    def list_names() -> list[str]:
        """List public names without importing their submodules."""
        return sorted(set(namespace) | set(owner))

    return resolve, list_names
