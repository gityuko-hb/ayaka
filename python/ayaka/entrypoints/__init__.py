"""The ways to reach the engine from outside the process: the command line and its protocols.

``Conversation`` and ``OutputEvent`` are the format every protocol speaks inside the server.

Public names are resolved from the module that defines them on first access. Nothing here imports
torch: the command line builds an engine, and the engine loads torch, only when a command runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.entrypoints.cli import main
    from ayaka.entrypoints.conversation import Conversation, OutputEvent

__all__ = ["Conversation", "OutputEvent", "main"]

_EXPORTS = {
    "cli": ("main",),
    "conversation": ("Conversation", "OutputEvent"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
