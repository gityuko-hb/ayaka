"""The engine: the loop that turns requests into generated text, and its two facades.

``build_engine`` wires every package together from one ``EngineConfig``; ``EngineCore`` runs the
loop one step at a time; ``InputProcessor`` and ``OutputProcessor`` stand at its two ends; ``LLM``
drives it from the calling thread, and ``AsyncLLM`` runs it on a thread of its own for coroutines,
each reading its request's ``RequestStream``; ``EngineClient`` is the contract the HTTP server
holds with the engine.

Public names are resolved from the module that defines them on first access. Nothing in this
package imports torch when it is imported: ``build_engine`` loads it when it is called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.utils.lazy_exports import build_lazy_exports

if TYPE_CHECKING:
    from ayaka.engine.async_llm import AsyncLLM, RequestStream
    from ayaka.engine.build import EnginePlan, build_engine, plan_engine
    from ayaka.engine.core import EngineCore
    from ayaka.engine.input import InputProcessor
    from ayaka.engine.llm import LLM
    from ayaka.engine.output import OutputProcessor
    from ayaka.engine.recorder import SessionRecorder
    from ayaka.engine.seam import EngineClient, EngineHealth, EngineInfo, EngineState

__all__ = [
    "LLM",
    "AsyncLLM",
    "EngineClient",
    "EngineCore",
    "EngineHealth",
    "EngineInfo",
    "EnginePlan",
    "EngineState",
    "InputProcessor",
    "OutputProcessor",
    "RequestStream",
    "SessionRecorder",
    "build_engine",
    "plan_engine",
]

_EXPORTS = {
    "async_llm": ("AsyncLLM", "RequestStream"),
    "build": ("EnginePlan", "build_engine", "plan_engine"),
    "core": ("EngineCore",),
    "input": ("InputProcessor",),
    "llm": ("LLM",),
    "output": ("OutputProcessor",),
    "recorder": ("SessionRecorder",),
    "seam": ("EngineClient", "EngineHealth", "EngineInfo", "EngineState"),
}

__getattr__, __dir__ = build_lazy_exports(__name__, _EXPORTS, globals())
