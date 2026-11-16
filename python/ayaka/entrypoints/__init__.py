"""The ways to reach the engine from outside the process: the command line today.

Public names are resolved from the module that defines them on first access. Nothing here imports
torch: the command line builds an engine, and the engine loads torch, only when a command runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.cli import main

__all__ = ["main"]

_EXPORTS = {
    "cli": ("main",),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
