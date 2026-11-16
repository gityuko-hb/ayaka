"""Which class builds which architecture.

``config.json`` names an architecture (``"GPT2LMHeadModel"``); :func:`resolve_model`
returns the :class:`~ayaka.models.base.CausalLM` subclass that implements it.

The architectures shipped with Ayaka are listed here by the module that defines
them and are imported only when one is asked for, so resolving ``GPT2LMHeadModel``
imports torch and nothing else does. A class defined elsewhere (a test model, a
plug-in) registers itself with :func:`register_model`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from ayaka.core.errors import ModelLoadError
from ayaka.utils.import_utils import resolve_qualname
from ayaka.utils.validation import require_text

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayaka.models.base import CausalLM

__all__ = ["register_model", "resolve_model"]

#: Architecture name to ``"module:Class"`` of the models in this package.
_BUILTIN: Final = {
    "GPT2LMHeadModel": "ayaka.models.gpt2:GPT2LMHeadModel",
}

_registered: dict[str, type[CausalLM]] = {}


def register_model[M: type[CausalLM]](*architectures: str) -> Callable[[M], M]:
    """Register a model class under one or more architecture names, as a class decorator.

    ::

        @register_model("TinyForCausalLM")
        class TinyModel(CausalLM): ...

    Raises:
        ValueError: From the decorator's call, if no name is given or a name is
            blank; from the decoration, if a name is already taken (by a model of
            this package or an earlier registration), or the class is not a
            :class:`~ayaka.models.base.CausalLM`.
    """
    if not architectures:
        raise ValueError("register_model needs at least one architecture name")
    for name in architectures:
        require_text(name, "architecture")

    def decorate(model: M) -> M:
        from ayaka.models.base import CausalLM

        candidate = cast("object", model)  # a caller may pass anything; the annotation is no check
        if not (isinstance(candidate, type) and issubclass(candidate, CausalLM)):
            raise ValueError(f"{model!r} is not a CausalLM subclass")
        taken = sorted(name for name in architectures if name in _registered or name in _BUILTIN)
        if taken:
            raise ValueError(f"architectures already registered: {taken}")
        for name in architectures:
            _registered[name] = model
        return model

    return decorate


def resolve_model(architecture: str) -> type[CausalLM]:
    """Return the class that implements ``architecture``.

    Raises:
        ModelLoadError: If no model is registered under that name. The message
            lists the supported architectures.
    """
    if (model := _registered.get(architecture)) is not None:
        return model
    if (target := _BUILTIN.get(architecture)) is not None:
        found: type[CausalLM] = resolve_qualname(target)
        return found
    supported = sorted({*_BUILTIN, *_registered})
    raise ModelLoadError(
        f"architecture {architecture!r} is not supported; supported architectures: {supported}",
        context={"architecture": architecture, "supported": supported},
    )
